"""Export run_stereo PMPose rows as raw model-input prediction JSON files.

The stereo runner stores keypoints after inverse rotation in raw fisheye pixel
coordinates.  The historical SMPL fitting entry expects the PMPose model
coordinates and applies its own raw-pixel inverse mapping, so this adapter
performs only that reversible coordinate conversion. Missing detections remain
NaN and are never interpolated or replaced here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def raw_to_model(x: float, y: float, side: str) -> tuple[float, float]:
    if side == "left":  # raw -> left ccw90 model image
        return float(y), float(1919.0 - x)
    if side == "right":  # raw -> right cw90 model image
        return float(1079.0 - y), float(x)
    raise ValueError(side)


def export(input_path: Path, output_path: Path, side: str) -> int:
    rows = []
    for frame_index, line in enumerate(input_path.read_text(encoding="utf-8-sig").splitlines()):
        if not line.strip():
            continue
        payload = json.loads(line)
        persons = payload.get(side, {}).get("persons") or []
        person = persons[0] if persons else None
        if person is None:
            keypoints = [[float("nan"), float("nan"), 0.0] for _ in range(23)]
        else:
            source = person.get("keypoints") or []
            keypoints = []
            for point in source[:23]:
                if len(point) < 3:
                    keypoints.append([float("nan"), float("nan"), 0.0])
                    continue
                x, y = raw_to_model(float(point[0]), float(point[1]), side)
                keypoints.append([x, y, float(point[2])])
            while len(keypoints) < 23:
                keypoints.append([float("nan"), float("nan"), 0.0])
        rows.append({
            "image_id": frame_index,
            "file_name": f"pair_{frame_index:04d}.png",
            "keypoints": [keypoints],
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"images": rows}, ensure_ascii=False), encoding="utf-8")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--left-output", type=Path, required=True)
    parser.add_argument("--right-output", type=Path, required=True)
    args = parser.parse_args()
    left_count = export(args.input, args.left_output, "left")
    right_count = export(args.input, args.right_output, "right")
    if left_count != right_count:
        raise RuntimeError("left/right export frame counts differ")
    print(json.dumps({"frames": left_count, "left": str(args.left_output), "right": str(args.right_output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
