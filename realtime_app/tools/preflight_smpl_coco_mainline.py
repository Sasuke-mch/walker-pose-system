#!/usr/bin/env python3
"""Preflight the clean PMPose COCO-17 -> male SMPL observation mainline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_app.smpl_coco_observation import inspect_smpl_coco_assets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--regressor", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_smpl_coco_assets(args.model_path, args.regressor)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
