"""Check local WiLoR and InterWild assets without running an experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from pose_app.hand_pose_provider import preflight_backend  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("wilor", "interwild", "all"), default="all", nargs="?")
    args = parser.parse_args()
    names = ("wilor", "interwild") if args.backend == "all" else (args.backend,)
    report = {name: preflight_backend(name) for name in names}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(item["assets_ready"] for item in report.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
