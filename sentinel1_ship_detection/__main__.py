"""Download the newest Sentinel-1 scene for an area and detect vessels in it.

Examples:
    python -m sentinel1_ship_detection --bbox 103.60 1.15 104.20 1.35 --name singapore
    python -m sentinel1_ship_detection --bbox 18.60 42.38 18.80 42.52 --date 2026-09-01 --name kotor
"""
import argparse
import os
import warnings
from datetime import datetime, timezone

from . import REPO_ROOT
from .detect import detect
from .download import download


def main():
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(prog="python -m sentinel1_ship_detection", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    p.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat(),
                   help="latest acquisition date, YYYY-MM-DD (default: today)")
    p.add_argument("--lookback-days", type=int, default=12)
    p.add_argument("--name", default="area", help="output folder name under output/")
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold (default 0.5)")
    p.add_argument("--keep-land", action="store_true", help="keep detections on land and at the shoreline")
    args = p.parse_args()

    out_dir = os.path.join(REPO_ROOT, "output", args.name)
    tif = download(args.bbox, args.date, out_dir, args.lookback_days)
    pred = detect(tif, out_dir, conf=args.conf, keep_land=args.keep_land)

    if len(pred):
        print(f"\nSummary: {len(pred)} vessels | median length {pred.vessel_length_m.median():.0f} m | "
              f"{(pred.vessel_speed_k > 3).sum()} moving faster than 3 kn")


if __name__ == "__main__":
    main()
