#!/usr/bin/env python3
"""Build one double-clickable HTML file from the static walker viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STYLESHEET_TAG = '  <link rel="stylesheet" href="./styles.css">'
MODULE_TAG = '  <script type="module" src="./app.js"></script>'


def build_standalone_html(index: str, css: str, javascript: str, motion_json: str) -> str:
    if STYLESHEET_TAG not in index:
        raise ValueError("viewer index does not contain the expected stylesheet tag")
    if MODULE_TAG not in index:
        raise ValueError("viewer index does not contain the expected module tag")
    payload = json.loads(motion_json)
    if int(payload.get("frame_count", 0)) != len(payload.get("frames") or []):
        raise ValueError("motion data frame_count does not match frames")
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    safe_payload = compact.replace("</", "<\\/")
    html = index.replace(STYLESHEET_TAG, f"  <style>\n{css}\n  </style>")
    embedded = (
        f'  <script id="motion-data" type="application/json">{safe_payload}</script>\n'
        f'  <script type="module">\n{javascript}\n  </script>'
    )
    return html.replace(MODULE_TAG, embedded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer-dir", type=Path, required=True)
    parser.add_argument("--motion-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    viewer = args.viewer_dir.resolve()
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite output: {output}")
    result = build_standalone_html(
        (viewer / "index.html").read_text(encoding="utf-8"),
        (viewer / "styles.css").read_text(encoding="utf-8"),
        (viewer / "app.js").read_text(encoding="utf-8"),
        (args.motion_json.resolve() if args.motion_json else viewer / "data" / "motion.json").read_text(encoding="utf-8"),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result, encoding="utf-8")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
