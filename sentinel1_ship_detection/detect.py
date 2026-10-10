"""Detect vessels in a Sentinel-1 GeoTIFF produced by sentinel1_ship_detection.download.

Uses the pretrained Sentinel-1 vessel detector (Faster R-CNN, checkpoint frcnn_cmp2/3dff445)
and attribute model (ResNet-50, checkpoint attr/c34aa37) from the Allen Institute for AI
(Beukema et al., 2023, arXiv:2312.03207).

Writes next to the input, or into --out:
    <scene>_detections.csv      one row per vessel
    <scene>_detections.geojson  the same, as map points
    <scene>_preview.png         the radar image with detections circled
"""
import argparse
import json
import os
import sys
import time
import types

import cv2
import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw
from pyproj import Transformer

from . import REPO_ROOT

AI2_DIR = os.getenv("AI2_VESSEL_DETECTION_DIR", os.path.join(REPO_ROOT, "third_party", "vessel-detection-sentinels"))
DETECTOR_DIR = os.path.join(AI2_DIR, "src", "model_artifacts", "frcnn_cmp2", "3dff445")
ATTRIBUTE_DIR = os.path.join(AI2_DIR, "src", "model_artifacts", "attr", "c34aa37")


def _load_apply_model():
    """Import apply_model from the Ai2 repository.

    Its pipeline module imports GDAL only for scene warping and coordinate transforms,
    which this package does with rasterio and pyproj instead, so GDAL is stubbed out.
    """
    weights = os.path.join(DETECTOR_DIR, "best.pth")
    if not os.path.isfile(weights) or os.path.getsize(weights) < 1_000_000:
        raise SystemExit(f"Model weights not found in {AI2_DIR}. Run scripts/setup_models.sh first.")
    osgeo = types.ModuleType("osgeo")
    osgeo.gdal = types.ModuleType("osgeo.gdal")
    osgeo.gdal.Transformer = object
    sys.modules.setdefault("osgeo", osgeo)
    sys.modules.setdefault("osgeo.gdal", osgeo.gdal)
    if AI2_DIR not in sys.path:
        sys.path.insert(0, AI2_DIR)
    from src.inference.pipeline import apply_model
    return apply_model


def water_mask(vv, land_threshold=60, downsample=4, min_land_px=35, shore_buffer_px=2):
    """True where a pixel is open water, derived from the radar image itself.

    Calm water is dark (DN ~20-35) and land is bright and textured (DN 100+). A median
    filter on a downsampled copy (~340 m window) ignores individual ships, so only
    large bright areas count as land. Small bright blobs (e.g. tight clusters of
    ships) are kept as water, and land is grown slightly to cover the shoreline.
    """
    h, w = vv.shape
    small = cv2.resize(vv, (max(1, w // downsample), max(1, h // downsample)), interpolation=cv2.INTER_AREA)
    land = (cv2.medianBlur(small, 9) > land_threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(land, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_land_px:
            land[labels == i] = 0
    if shore_buffer_px:
        kernel = np.ones((2 * shore_buffer_px + 1,) * 2, np.uint8)
        land = cv2.dilate(land, kernel)
    land = cv2.resize(land, (w, h), interpolation=cv2.INTER_NEAREST)
    return (land == 0) & (vv > 0)


class PixelToLonLat:
    """Mimics the gdal.Transformer.TransformPoint call apply_model makes."""

    def __init__(self, raster_transform, crs):
        self.affine = raster_transform
        self.to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def TransformPoint(self, _direction, col, row, _z):
        x, y = self.affine * (col + 0.5, row + 0.5)
        lon, lat = self.to_wgs84.transform(x, y)
        return 1, (lon, lat, 0)


def detect(tif_path, out_dir=None, conf=0.5, nms_thresh=20, keep_land=False,
           window_size=2048, padding=400, overlap=20):
    """Run detection on a GeoTIFF and write CSV, GeoJSON and preview outputs."""
    out_dir = out_dir or os.path.dirname(tif_path)
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(tif_path))[0]

    with rasterio.open(tif_path) as src:
        img = src.read()  # (2, h, w) uint8: vh, vv
        transformer = PixelToLonLat(src.transform, src.crs)

    # apply_model needs the image to be at least one window in each direction.
    _, h, w = img.shape
    window_size = min(window_size, h, w)
    padding = min(padding, window_size // 4)

    apply_model = _load_apply_model()
    torch.set_num_threads(os.cpu_count() or 4)
    start = time.time()
    pred = apply_model(
        DETECTOR_DIR,
        torch.as_tensor(img),
        window_size=window_size,
        padding=padding,
        overlap=overlap,
        threshold=conf,
        transformer=transformer,
        nms_thresh=nms_thresh,
        postprocess_model_dir=ATTRIBUTE_DIR,
        out_path=out_dir,
        catalog="sentinel1",
        device=torch.device("cpu"),
    )
    print(f"{len(pred)} candidates in {time.time() - start:.0f}s")

    if not keep_land and len(pred):
        water = water_mask(img[1])
        on_water = water[pred["preprocess_row"].clip(0, h - 1), pred["preprocess_column"].clip(0, w - 1)]
        print(f"Dropped {int((~on_water).sum())} on land or at the shoreline")
        pred = pred[on_water].copy()
    print(f"{len(pred)} vessels")

    heading_cols = [f"heading_bucket_{i}" for i in range(16)]
    if len(pred):
        pred["heading_deg"] = pred[heading_cols].to_numpy().argmax(axis=1) * 22.5
    else:
        pred["heading_deg"] = []
    pred = pred.drop(columns=heading_cols).sort_values("score", ascending=False)
    pred = pred.rename(columns={"preprocess_row": "row", "preprocess_column": "col"})

    csv_path = os.path.join(out_dir, f"{name}_detections.csv")
    pred.to_csv(csv_path, index=False)

    features = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [round(r.lon, 6), round(r.lat, 6)]},
        "properties": {
            "score": round(r.score, 3),
            "length_m": round(r.vessel_length_m, 1),
            "width_m": round(r.vessel_width_m, 1),
            "heading_deg": r.heading_deg,
            "speed_knots": round(r.vessel_speed_k, 1),
            "fishing_probability": round(r.is_fishing_vessel, 3),
        },
    } for r in pred.itertuples()]
    geojson_path = os.path.join(out_dir, f"{name}_detections.geojson")
    with open(geojson_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)

    preview_path = os.path.join(out_dir, f"{name}_preview.png")
    save_preview(img, pred, preview_path)

    print(f"Saved {csv_path}\n      {geojson_path}\n      {preview_path}")
    return pred


def save_preview(img, pred, path, max_width=4000):
    """VV band with a ring around each vessel; larger rings are longer vessels."""
    vv = img[1].astype(np.float32)
    gray = np.clip((vv - 8) / (140 - 8) * 255, 0, 255).astype(np.uint8)
    scale = min(1.0, max_width / vv.shape[1])
    size = (round(vv.shape[1] * scale), round(vv.shape[0] * scale))
    ss = 2  # draw at 2x and downsample for smooth rings
    base = Image.fromarray(gray).resize((size[0] * ss, size[1] * ss), Image.LANCZOS).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for p in pred.itertuples():
        x, y = p.col * scale * ss, p.row * scale * ss
        r = (4 + np.sqrt(p.vessel_length_m) * 0.55) * ss
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 77, 109, 60), outline=(255, 77, 109, 255), width=3)
    Image.alpha_composite(base, overlay).convert("RGB").resize(size, Image.LANCZOS).save(path, optimize=True)


def main():
    import warnings
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(prog="python -m sentinel1_ship_detection.detect", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tif", help="GeoTIFF from sentinel1_ship_detection.download")
    p.add_argument("--out", help="output directory (default: next to the input)")
    p.add_argument("--conf", type=float, default=0.5,
                   help="confidence threshold (default 0.5; Ai2 uses 0.9, which misses the brightest ships here)")
    p.add_argument("--nms", type=int, default=20, help="merge detections closer than this many pixels (default 20, ~190 m)")
    p.add_argument("--keep-land", action="store_true", help="keep detections on land and at the shoreline")
    args = p.parse_args()
    detect(args.tif, args.out, conf=args.conf, nms_thresh=args.nms, keep_land=args.keep_land)


if __name__ == "__main__":
    main()
