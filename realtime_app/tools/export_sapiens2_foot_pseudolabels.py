"""Export and visualize Sapiens2-308 ankle, toe, and heel pseudo-labels.

The saved Sapiens2 run contains 308 points although the project's earlier
adapter retained only COCO-17.  This tool reads the original JSON directly,
restores the selected foot points to raw fisheye pixels, and builds an upright
visual-audit pack.  Outputs remain pseudo-labels, not independent truth.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from pose_app.rotation import model_to_raw_point, raw_to_model_point, rotate_image_for_model


POINTS = {
    13: "left_ankle",
    15: "left_big_toe",
    16: "left_small_toe",
    17: "left_heel",
    14: "right_ankle",
    18: "right_big_toe",
    19: "right_small_toe",
    20: "right_heel",
}
LINKS = (
    ("left_heel", "left_ankle"),
    ("left_ankle", "left_big_toe"),
    ("left_ankle", "left_small_toe"),
    ("left_big_toe", "left_small_toe"),
    ("right_heel", "right_ankle"),
    ("right_ankle", "right_big_toe"),
    ("right_ankle", "right_small_toe"),
    ("right_big_toe", "right_small_toe"),
)
LABELS = {
    "left_ankle": "LA",
    "left_big_toe": "LBT",
    "left_small_toe": "LST",
    "left_heel": "LH",
    "right_ankle": "RA",
    "right_big_toe": "RBT",
    "right_small_toe": "RST",
    "right_heel": "RH",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-selection", type=Path, required=True)
    parser.add_argument("--left-predictions", type=Path, required=True)
    parser.add_argument("--right-predictions", type=Path, required=True)
    parser.add_argument("--target-review-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--left-prediction-rotation", default="cw90")
    parser.add_argument("--right-prediction-rotation", default="ccw90")
    parser.add_argument("--left-display-rotation", default="ccw90")
    parser.add_argument("--right-display-rotation", default="cw90")
    parser.add_argument("--usable-score", type=float, default=0.25)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_predictions(path: Path) -> dict[str, dict]:
    source = json.loads(path.read_text(encoding="utf-8-sig"))
    images = source.get("images")
    if not isinstance(images, list):
        raise RuntimeError(f"No images list in {path}")
    by_name = {str(row["file_name"]): row for row in images}
    if len(by_name) != len(images):
        raise RuntimeError(f"Duplicate image names in {path}")
    return by_name


def highest_bbox_instance(image: dict) -> tuple[dict | None, int, float | None]:
    instances = image.get("instances", [])
    if not instances:
        return None, 0, None
    ordered = sorted(instances, key=lambda item: float(item["bbox_score_from_yolo26x"]), reverse=True)
    ratio = None
    if len(ordered) > 1 and float(ordered[0]["bbox_score_from_yolo26x"]) > 0:
        ratio = float(ordered[1]["bbox_score_from_yolo26x"]) / float(ordered[0]["bbox_score_from_yolo26x"])
    return ordered[0], len(instances), ratio


def draw_halo_line(image: np.ndarray, p1: tuple[int, int], p2: tuple[int, int]) -> None:
    cv2.line(image, p1, p2, (255, 255, 255), 8, cv2.LINE_AA)
    cv2.line(image, p1, p2, (0, 0, 0), 3, cv2.LINE_AA)


def draw_halo_point(image: np.ndarray, point: tuple[int, int], label: str) -> None:
    cv2.circle(image, point, 9, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(image, point, 5, (0, 0, 0), -1, cv2.LINE_AA)
    anchor = (point[0] + 7, point[1] - 7)
    cv2.putText(image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    width = max(1, round(image.shape[1] * height / image.shape[0]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_panel(raw_image: np.ndarray, raw_points: dict[str, tuple[float, float, float]], display_rotation: str, title: str) -> np.ndarray:
    upright = rotate_image_for_model(raw_image, display_rotation).copy()
    display_points: dict[str, tuple[int, int]] = {}
    for name, (x, y, _) in raw_points.items():
        dx, dy = raw_to_model_point(x, y, raw_image.shape[1], raw_image.shape[0], display_rotation)
        display_points[name] = (round(dx), round(dy))
    for start, end in LINKS:
        if start in display_points and end in display_points:
            draw_halo_line(upright, display_points[start], display_points[end])
    for name, point in display_points.items():
        draw_halo_point(upright, point, LABELS[name])

    overview = resize_to_height(upright, 720)
    if display_points:
        xy = np.asarray(list(display_points.values()), dtype=np.int32)
        x1, y1 = xy.min(axis=0)
        x2, y2 = xy.max(axis=0)
        margin = max(80, int(max(x2 - x1, y2 - y1) * 0.45))
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(upright.shape[1] - 1, x2 + margin), min(upright.shape[0] - 1, y2 + margin)
        crop = upright[y1 : y2 + 1, x1 : x2 + 1]
    else:
        crop = upright
    zoom = cv2.resize(crop, (720, 720), interpolation=cv2.INTER_CUBIC)
    panel = np.full((770, overview.shape[1] + 720, 3), 255, dtype=np.uint8)
    panel[50:770, : overview.shape[1]] = overview
    panel[50:770, overview.shape[1] :] = zoom
    cv2.putText(panel, title, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 2, cv2.LINE_AA)
    return panel


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    selection = args.input_selection.resolve()
    manifest = read_csv(selection / "selection_manifest.csv")
    if not manifest:
        raise RuntimeError("Selection manifest is empty.")
    required_manifest_fields = {"file_name", "condition"}
    if not required_manifest_fields.issubset(manifest[0]):
        raise RuntimeError(f"Selection manifest must contain {sorted(required_manifest_fields)}.")
    condition_by_name = {row["file_name"]: row["condition"] for row in manifest}
    target_review = {row["image_id"]: row for row in read_csv(args.target_review_csv.resolve())}
    sources = {
        "left": (load_predictions(args.left_predictions.resolve()), args.left_prediction_rotation, args.left_display_rotation),
        "right": (load_predictions(args.right_predictions.resolve()), args.right_prediction_rotation, args.right_display_rotation),
    }
    for side, (records, _, _) in sources.items():
        if set(records) != set(condition_by_name):
            raise RuntimeError(f"{side}: prediction names do not match the selected-pair manifest")

    output.mkdir(parents=True)
    overlay_dir = output / "all_120_upright_overlays"
    overlay_dir.mkdir()
    point_rows: list[dict] = []
    image_rows: list[dict] = []
    panels: dict[str, np.ndarray] = {}
    for side, (records, prediction_rotation, display_rotation) in sources.items():
        for file_name in sorted(records):
            image_id = f"{side}_{Path(file_name).stem}"
            selected, candidate_count, second_ratio = highest_bbox_instance(records[file_name])
            raw_points: dict[str, tuple[float, float, float]] = {}
            scores = []
            for index, name in POINTS.items():
                if selected is None:
                    x_raw = y_raw = None
                    score = 0.0
                else:
                    keypoints = selected["keypoints308"]
                    keypoint_scores = selected["keypoint_scores"]
                    if len(keypoints) != 308 or len(keypoint_scores) != 308:
                        raise RuntimeError(f"{image_id}: expected 308 Sapiens2 points")
                    x_model, y_model = [float(value) for value in keypoints[index]]
                    score = float(keypoint_scores[index])
                    x_raw, y_raw = model_to_raw_point(x_model, y_model, 1920, 1080, prediction_rotation)
                usable = score >= args.usable_score
                if usable and x_raw is not None and y_raw is not None:
                    raw_points[name] = (x_raw, y_raw, score)
                scores.append(score)
                point_rows.append({
                    "image_id": image_id,
                    "file_name": file_name,
                    "side": side,
                    "condition": condition_by_name[file_name],
                    "joint_subject_anatomy": name,
                    "keypoint_index_sapiens308": index,
                    "x_px_raw_fisheye": "" if not usable else f"{x_raw:.6f}",
                    "y_px_raw_fisheye": "" if not usable else f"{y_raw:.6f}",
                    "reference_usable": str(usable).lower(),
                    "sapiens2_confidence": f"{score:.8f}",
                    "reference_source": "Sapiens2-0.4B_keypoints308_historical_CWCCW_input",
                })
            review = target_review.get(image_id, {})
            summary = {
                "image_id": image_id,
                "file_name": file_name,
                "side": side,
                "condition": condition_by_name[file_name],
                "candidate_instance_count": candidate_count,
                "second_to_primary_bbox_score_ratio": "" if second_ratio is None else f"{second_ratio:.8f}",
                "target_ambiguous_original": review.get("needs_manual_target_review", ""),
                "usable_foot_point_count": len(raw_points),
                "min_foot_point_confidence": f"{min(scores):.8f}",
                "median_foot_point_confidence": f"{float(np.median(scores)):.8f}",
            }
            image_rows.append(summary)
            raw_path = selection / "raw_fisheye" / side / file_name
            raw_image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if raw_image is None:
                raise RuntimeError(f"Could not read {raw_path}")
            panel = make_panel(raw_image, raw_points, display_rotation, f"{image_id} | {condition_by_name[file_name]} | usable={len(raw_points)}/8 | min={min(scores):.3f}")
            panels[image_id] = panel
            if not cv2.imwrite(str(overlay_dir / f"{image_id}.jpg"), panel):
                raise RuntimeError(f"Could not write overlay for {image_id}")

    reasons: dict[str, set[str]] = defaultdict(set)
    for row in image_rows:
        if row["target_ambiguous_original"].lower() == "true":
            reasons[row["image_id"]].add("target_identity_reviewed_in_A4")
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in image_rows:
        grouped[(row["condition"], row["side"])].append(row)
    for rows in grouped.values():
        for row in sorted(rows, key=lambda item: float(item["min_foot_point_confidence"]))[:2]:
            reasons[row["image_id"]].add("low_foot_confidence")
    risk_rows = []
    risk_dir = output / "risk_review_overlays"
    risk_dir.mkdir()
    for image_id in sorted(reasons):
        row = next(item for item in image_rows if item["image_id"] == image_id)
        risk_rows.append({**row, "selection_reason": "+".join(sorted(reasons[image_id]))})
        cv2.imwrite(str(risk_dir / f"{image_id}.jpg"), panels[image_id])

    sheets_dir = output / "contact_sheets"
    sheets_dir.mkdir()
    thumb_w, thumb_h = 560, 384
    for sheet_index, start in enumerate(range(0, len(risk_rows), 6), start=1):
        batch = risk_rows[start : start + 6]
        canvas = np.full((2 * thumb_h, 3 * thumb_w, 3), 255, dtype=np.uint8)
        for local_index, row in enumerate(batch):
            thumb = cv2.resize(panels[row["image_id"]], (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            y = (local_index // 3) * thumb_h
            x = (local_index % 3) * thumb_w
            canvas[y : y + thumb_h, x : x + thumb_w] = thumb
        cv2.imwrite(str(sheets_dir / f"sheet_{sheet_index:02d}.jpg"), canvas)

    write_csv(output / "sapiens2_foot_pseudolabels.csv", point_rows)
    write_csv(output / "all_120_image_foot_summary.csv", image_rows)
    write_csv(output / "risk_review_selection.csv", risk_rows)
    usable_counts = defaultdict(int)
    for row in point_rows:
        usable_counts[(row["condition"], row["side"], row["joint_subject_anatomy"])] += row["reference_usable"] == "true"
    image_counts = defaultdict(int)
    for row in image_rows:
        image_counts[(row["condition"], row["side"])] += 1
    summary_rows = [
        {"condition": key[0], "side": key[1], "joint": key[2], "usable_count": value, "image_count": image_counts[(key[0], key[1])], "usable_rate": value / image_counts[(key[0], key[1])]}
        for key, value in sorted(usable_counts.items())
    ]
    write_csv(output / "foot_point_availability_summary.csv", summary_rows)
    metadata = {
        "experiment_id": "E20260829-A6_Sapiens2_308_foot_pseudolabel_export",
        "images": len(image_rows),
        "points_per_image": len(POINTS),
        "prediction_input_rotation": {"left": args.left_prediction_rotation, "right": args.right_prediction_rotation},
        "display_rotation": {"left": args.left_display_rotation, "right": args.right_display_rotation},
        "coordinate_space": "original 1920x1080 raw fisheye pixels after inverse prediction-input rotation",
        "usable_score": args.usable_score,
        "risk_review_images": len(risk_rows),
        "interpretation": "Sapiens2-308 operational foot pseudo-labels; visual review is required before use and no row is independent ground truth.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "point_rows": len(point_rows), "risk_review_images": len(risk_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
