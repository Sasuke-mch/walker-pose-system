"""Render existing C3/D1 lower-body disagreements for visual 2-D review."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


LOWER_BODY_INDICES = {
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}
C3_COLOR = (255, 210, 0)
D1_COLOR = (255, 0, 255)
HIGHLIGHT_COLOR = (0, 255, 255)


def _prediction_map(path: Path) -> dict[str, list[list[float]]]:
    root = json.loads(path.read_text(encoding="utf-8-sig"))
    output = {}
    for record in root["images"]:
        keypoint_sets = record["keypoints"]
        if len(keypoint_sets) != 1:
            raise ValueError(f"{path}: expected one PMPose output per frame.")
        output[record["file_name"]] = keypoint_sets[0]
    return output


def _draw_points(
    image: np.ndarray,
    points: list[list[float]],
    color: tuple[int, int, int],
    label: str,
    divergent: set[str],
) -> None:
    for joint, index in LOWER_BODY_INDICES.items():
        x, y, score = points[index][:3]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        point = (round(float(x)), round(float(y)))
        draw_color = HIGHLIGHT_COLOR if joint in divergent else color
        cv2.circle(image, point, 7 if joint in divergent else 5, draw_color, 2)
        cv2.putText(
            image,
            f"{label}:{joint.replace('_', '')} {float(score):.2f}",
            (point[0] + 7, point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            draw_color,
            1,
            cv2.LINE_AA,
        )


def _side_panel(
    image_path: Path,
    c3_points: list[list[float]],
    d1_points: list[list[float]],
    divergent: set[str],
    side: str,
) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {image_path}.")
    _draw_points(image, c3_points, C3_COLOR, "C3", divergent)
    _draw_points(image, d1_points, D1_COLOR, "D1", divergent)
    cv2.putText(
        image,
        f"{side} | C3 cyan, D1 magenta, divergent joint yellow",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"{side} | C3 cyan, D1 magenta, divergent joint yellow",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return cv2.resize(image, (405, 720), interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render saved C3/D1 divergent lower-body predictions."
    )
    parser.add_argument("--paired-joint-csv", required=True, type=Path)
    parser.add_argument("--c3-left-json", required=True, type=Path)
    parser.add_argument("--c3-right-json", required=True, type=Path)
    parser.add_argument("--d1-left-json", required=True, type=Path)
    parser.add_argument("--d1-right-json", required=True, type=Path)
    parser.add_argument("--left-input-dir", required=True, type=Path)
    parser.add_argument("--right-input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.paired_joint_csv.open(newline="", encoding="utf-8-sig") as handle:
        divergence_rows = [
            row
            for row in csv.DictReader(handle)
            if row["c3_status"] != row["d1_status"]
        ]
    if not divergence_rows:
        raise ValueError("No C3/D1 status divergences to render.")
    joints_by_frame: dict[str, set[str]] = {}
    for row in divergence_rows:
        joints_by_frame.setdefault(row["file_name"], set()).add(row["joint"])

    c3_left = _prediction_map(args.c3_left_json.resolve())
    c3_right = _prediction_map(args.c3_right_json.resolve())
    d1_left = _prediction_map(args.d1_left_json.resolve())
    d1_right = _prediction_map(args.d1_right_json.resolve())
    review_rows = []
    rendered = []
    for file_name in sorted(joints_by_frame):
        divergent = joints_by_frame[file_name]
        left = _side_panel(
            args.left_input_dir.resolve() / file_name,
            c3_left[file_name],
            d1_left[file_name],
            divergent,
            "left",
        )
        right = _side_panel(
            args.right_input_dir.resolve() / file_name,
            c3_right[file_name],
            d1_right[file_name],
            divergent,
            "right",
        )
        panel = np.concatenate([left, right], axis=1)
        title = f"{file_name} | divergent: {', '.join(sorted(divergent))}"
        titled = np.full((760, panel.shape[1], 3), 245, dtype=np.uint8)
        titled[40:, :] = panel
        cv2.putText(
            titled,
            title,
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        output_path = args.output_dir / f"{Path(file_name).stem}_c3_d1_review.jpg"
        if not cv2.imwrite(str(output_path), titled):
            raise RuntimeError(f"Could not write {output_path}.")
        rendered.append(titled)
        review_rows.append(
            {
                "file_name": file_name,
                "divergent_joints": ";".join(sorted(divergent)),
                "review_image": str(output_path),
                "review_status": "pending_visual_review",
                "scope": "saved C3/D1 2-D points only; no pose inference rerun",
            }
        )

    for sheet_index in range(0, len(rendered), 5):
        chunk = rendered[sheet_index : sheet_index + 5]
        sheet = np.concatenate(chunk, axis=0)
        cv2.imwrite(
            str(args.output_dir / f"contact_sheet_{sheet_index // 5 + 1:02d}.jpg"),
            sheet,
        )
    with (args.output_dir / "visual_review_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    print(json.dumps({"frames": len(review_rows), "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
