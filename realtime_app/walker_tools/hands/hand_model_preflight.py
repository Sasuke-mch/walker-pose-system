"""Check local WiLoR and InterWild assets without running an experiment."""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
import json
from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parents[2]
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
