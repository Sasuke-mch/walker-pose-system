#!/usr/bin/env python3
"""Write a SHA-256 manifest for a completed, parameter-frozen experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file", action="append", type=Path, required=True)
    parser.add_argument("--description", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    records = []
    for path in args.file:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        records.append({
            "path": resolved.relative_to(root).as_posix(),
            "bytes": resolved.stat().st_size,
            "sha256": digest(resolved),
        })
    payload = {
        "schema_version": "frozen_result_manifest_v1",
        "description": args.description,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(records), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
