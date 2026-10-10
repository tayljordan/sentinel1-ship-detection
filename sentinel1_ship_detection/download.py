"""Download a Sentinel-1 radar scene for an area, prepared for the vessel detector.

The detector was trained on raw GRD amplitude (digital numbers) warped to Web Mercator
at zoom level 13 (~9.55 m/px) and clipped to 0-255. The Sentinel Hub Process API serves
calibrated BETA0 backscatter instead, so it is converted back: for IW GRD products the
beta-nought calibration constant is 237, so DN = 237 * sqrt(beta0).

Output is a 2-band uint8 GeoTIFF (band 1 = VH, band 2 = VV) in EPSG:3857.
"""
import argparse
import json
import math
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import rasterio
import requests
from dotenv import load_dotenv
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from . import REPO_ROOT

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"

WEB_MERCATOR_M = 2 * math.pi * 6378137
PIXEL_SIZE = WEB_MERCATOR_M / 512 / 2**13  # zoom 13, matches the training data
MAX_REQUEST_PX = 2500  # Process API per-request edge limit
BETA0_CONSTANT = 237.0

EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["VH", "VV", "dataMask"] }],
    output: { bands: 3, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  return [s.VH, s.VV, s.dataMask];
}
"""


def _iso(t):
    return t.isoformat().replace("+00:00", "Z")


def get_token():
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    client_id = os.getenv("COPERNICUS_CLIENT_ID")
    client_secret = os.getenv("COPERNICUS_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SystemExit("Set COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in .env (see .env.example).")
    resp = requests.post(TOKEN_URL, data={
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }, timeout=60)
    resp.raise_for_status()
    return resp.json()["access_token"]


def find_scenes(token, bbox, date, lookback_days):
    """Return Sentinel-1 IW dual-polarisation scenes covering bbox, newest first."""
    end = datetime.fromisoformat(date).replace(tzinfo=timezone.utc) + timedelta(days=1)
    start = end - timedelta(days=lookback_days)
    body = {
        "collections": ["sentinel-1-grd"],
        "bbox": bbox,
        "datetime": f"{_iso(start)}/{_iso(end)}",
        "limit": 50,
        "filter": "sar:instrument_mode = 'IW' AND s1:polarization = 'DV'",
        "filter-lang": "cql2-text",
    }
    resp = requests.post(CATALOG_URL, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=60)
    resp.raise_for_status()
    feats = resp.json().get("features", [])
    return sorted(feats, key=lambda f: f["properties"]["datetime"], reverse=True)


def fetch_tile(token, x0, y1, width, height, scene_time):
    """Fetch one tile (at most 2500 px per side) in EPSG:3857 with top-left corner (x0, y1)."""
    t = datetime.fromisoformat(scene_time.replace("Z", "+00:00"))
    body = {
        "input": {
            "bounds": {
                "bbox": [x0, y1 - height * PIXEL_SIZE, x0 + width * PIXEL_SIZE, y1],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/3857"},
            },
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {"from": _iso(t - timedelta(minutes=10)), "to": _iso(t + timedelta(minutes=10))},
                    "acquisitionMode": "IW",
                    "polarization": "DV",
                    "resolution": "HIGH",
                },
                "processing": {"backCoeff": "BETA0", "orthorectify": False},
            }],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }
    resp = requests.post(PROCESS_URL, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(f"Process API returned {resp.status_code}: {resp.text[:500]}")
    with MemoryFile(resp.content) as mem, mem.open() as src:
        return src.read()  # (3, h, w) float32


def download(bbox, date, out_dir, lookback_days=12, scene_index=0):
    """Download the newest Sentinel-1 scene over bbox and return the GeoTIFF path."""
    os.makedirs(out_dir, exist_ok=True)
    token = get_token()

    scenes = find_scenes(token, bbox, date, lookback_days)
    if not scenes:
        raise SystemExit(f"No Sentinel-1 IW scenes over {bbox} in the {lookback_days} days up to {date}.")
    scene = scenes[scene_index]
    scene_id = scene["id"].removesuffix(".SAFE")
    scene_time = scene["properties"]["datetime"]
    print(f"Found {len(scenes)} scenes; using {scene_id} ({scene_time})")

    # Snap the area to the zoom-13 pixel grid.
    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin = to_merc.transform(bbox[0], bbox[1])
    xmax, ymax = to_merc.transform(bbox[2], bbox[3])
    xmin = math.floor(xmin / PIXEL_SIZE) * PIXEL_SIZE
    ymax = math.ceil(ymax / PIXEL_SIZE) * PIXEL_SIZE
    width = math.ceil((xmax - xmin) / PIXEL_SIZE)
    height = math.ceil((ymax - ymin) / PIXEL_SIZE)
    print(f"Area is {width} x {height} px at {PIXEL_SIZE:.2f} m/px")

    mosaic = np.zeros((3, height, width), dtype=np.float32)
    for row in range(0, height, MAX_REQUEST_PX):
        for col in range(0, width, MAX_REQUEST_PX):
            h = min(MAX_REQUEST_PX, height - row)
            w = min(MAX_REQUEST_PX, width - col)
            print(f"  fetching tile at row {row}, col {col} ({w} x {h})")
            mosaic[:, row:row + h, col:col + w] = fetch_tile(
                token, xmin + col * PIXEL_SIZE, ymax - row * PIXEL_SIZE, w, h, scene_time)

    valid = mosaic[2] > 0
    dn = BETA0_CONSTANT * np.sqrt(np.clip(mosaic[:2], 0, None))
    img = np.clip(dn, 0, 255).astype(np.uint8)
    img[:, ~valid] = 0
    print(f"Valid coverage: {valid.mean():.0%}")

    out_path = os.path.join(out_dir, f"{scene_id}.tif")
    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 2, "width": width, "height": height,
        "crs": "EPSG:3857", "transform": from_origin(xmin, ymax, PIXEL_SIZE, PIXEL_SIZE),
        "compress": "deflate", "nodata": 0,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(img)
        dst.set_band_description(1, "vh")
        dst.set_band_description(2, "vv")

    with open(os.path.join(out_dir, f"{scene_id}.json"), "w") as f:
        json.dump({"scene_id": scene_id, "datetime": scene_time, "bbox": bbox, "date": date}, f, indent=2)
    print(f"Saved {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(prog="python -m sentinel1_ship_detection.download", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    p.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat(),
                   help="latest acquisition date, YYYY-MM-DD (default: today)")
    p.add_argument("--lookback-days", type=int, default=12)
    p.add_argument("--scene-index", type=int, default=0, help="0 = newest scene, 1 = next newest, ...")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "output"))
    args = p.parse_args()
    download(args.bbox, args.date, args.out, args.lookback_days, args.scene_index)


if __name__ == "__main__":
    main()
