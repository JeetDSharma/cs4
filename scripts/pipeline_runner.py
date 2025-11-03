#!/usr/bin/env python3
import argparse, subprocess, sys
from cs4.utils.config_loader import stamp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["download","unzip","both"], default="both")
    ap.add_argument("--download_cfg", default="configs/download_config.yaml")
    ap.add_argument("--unzip_cfg", default="configs/unzip_config.yaml")
    args = ap.parse_args()

    dl_ts = stamp()
    if args.stage in ("download","both"):
        subprocess.check_call([
            sys.executable, "scripts/download.py",
            "--config", args.download_cfg
        ])
    if args.stage in ("unzip","both"):
        # pass the download timestamp to unzip if needed
        subprocess.check_call([
            sys.executable, "scripts/unzip.py",
            "--config", args.unzip_cfg,
            "--vars", f"download_timestamp={dl_ts}_download"
        ])

if __name__ == "__main__":
    main()
