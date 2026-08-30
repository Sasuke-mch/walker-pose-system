"""Build a blinded, stratified lower-limb annotation set from B0 replay data.

No pose model is run here.  The output keeps the unmarked raw stereo images
separate from B0-derived analysis context so a reviewer can label 2-D points
without being steered by the baseline model's success/failure label.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
sys.path.insert(0, str(APP_ROOT))

from pose_app.calibration import StereoCalibration  # noqa: E402


TARGET_PAIRS = 200
RAW_CAPTURE = (
    PROJECT_ROOT
    / "research_records"
    / "raw_captures"
    / "R20260826-01_far_to_near_domain_capture"
    / "approach_far_to_near"
    / "20260826_195459_568"
)
B0_JSONL = (
    PROJECT_ROOT
    / "research_records"
    / "official_runs"
    / "E20260827-B0_replay_baseline"
    / "stereo_results.jsonl"
)
CALIBRATION = APP_ROOT / "calibration" / "results" / "stereo_fisheye.json"
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "research_records"
    / "dataset_preparation"
    / "V20260828-F4_foot_evaluation_set_manifest"
)

LANDMARKS = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_toe_tip",
    "right_toe_tip",
    "left_heel_optional",
    "right_heel_optional",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--target-pairs", type=int, default=TARGET_PAIRS)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="complete a partial review-image write without overwriting existing files",
    )
    return parser.parse_args()


def load_records() -> list[dict]:
    records = []
    with B0_JSONL.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if not isinstance(record.get("pair_id"), int):
                raise ValueError("B0 record lacks an integer pair_id")
            records.append(record)
    if not records:
        raise ValueError("B0 JSONL is empty")
    return records


def person_for_id(record: dict, camera: str, person_id: int | None) -> dict | None:
    people = record[camera].get("persons", [])
    for person in people:
        if person.get("person_id") == person_id:
            return person
    return people[0] if people else None


def ankle_point(person: dict | None) -> tuple[float, float, float] | None:
    if person is None:
        return None
    keypoints = person.get("keypoints", [])
    if len(keypoints) <= 16 or len(keypoints[16]) < 3:
        return None
    x, y, score = (float(value) for value in keypoints[16][:3])
    if not all(np.isfinite((x, y, score))):
        return None
    return x, y, score


def radius_norm(point: tuple[float, float, float] | None, image_size: tuple[int, int]) -> float | None:
    if point is None:
        return None
    width, height = image_size
    return float(np.hypot((point[0] - (width - 1) * 0.5) / (width * 0.5), (point[1] - (height - 1) * 0.5) / (height * 0.5)))


def item_from_record(record: dict, index: int, image_size: tuple[int, int]) -> dict:
    stereo_people = record.get("persons_3d", [])
    stereo_person = stereo_people[0] if stereo_people else None
    right_ankle_3d = None
    if stereo_person is not None:
        for point in stereo_person.get("keypoints_3d", []):
            if point.get("index") == 16:
                right_ankle_3d = point
                break
    if right_ankle_3d is None:
        outcome = "other_failure"
        reason = "no_stereo_person_or_right_ankle_record"
    elif bool(right_ankle_3d.get("valid")):
        outcome = "valid_control"
        reason = "valid"
    elif right_ankle_3d.get("reason") == "high_reprojection_error":
        outcome = "high_reprojection_failure"
        reason = "high_reprojection_error"
    else:
        outcome = "other_failure"
        reason = str(right_ankle_3d.get("reason") or "unknown")
    left_person_id = None if stereo_person is None else stereo_person.get("left_person_id")
    right_person_id = None if stereo_person is None else stereo_person.get("right_person_id")
    left_ankle = ankle_point(person_for_id(record, "left", left_person_id))
    right_ankle = ankle_point(person_for_id(record, "right", right_person_id))
    radii = [value for value in (radius_norm(left_ankle, image_size), radius_norm(right_ankle, image_size)) if value is not None]
    mean_radius = float(np.mean(radii)) if radii else None
    zone = "unknown" if mean_radius is None else ("edge" if mean_radius >= 0.70 else "central")
    return {
        "pair_id": int(record["pair_id"]),
        "sequence_index": index,
        "left_frame_id": int(record["left_frame_id"]),
        "right_frame_id": int(record["right_frame_id"]),
        "pair_timestamp_sec": float(record["pair_timestamp_sec"]),
        "outcome": outcome,
        "right_ankle_b0_reason": reason,
        "right_ankle_b0_valid": None if right_ankle_3d is None else bool(right_ankle_3d.get("valid")),
        "right_ankle_b0_reprojection_error_px": None if right_ankle_3d is None else right_ankle_3d.get("reprojection_error_mean_px"),
        "left_ankle_b0_xy_score": None if left_ankle is None else list(left_ankle),
        "right_ankle_b0_xy_score": None if right_ankle is None else list(right_ankle),
        "mean_right_ankle_radius_norm": mean_radius,
        "image_zone": zone,
    }


def temporal_bin(item: dict, total: int) -> str:
    if item["sequence_index"] < total / 3:
        return "early_far_to_near"
    if item["sequence_index"] < total * 2 / 3:
        return "middle_far_to_near"
    return "late_far_to_near"


def distribute(total: int, buckets: int = 3) -> list[int]:
    base, extra = divmod(total, buckets)
    return [base + (1 if index < extra else 0) for index in range(buckets)]


def select_diverse(items: list[dict], count: int) -> list[dict]:
    """Deterministically interleave high-radius and low-radius candidates."""
    if count <= 0 or not items:
        return []
    edge = sorted((item for item in items if item["image_zone"] == "edge"), key=lambda value: (-float(value["mean_right_ankle_radius_norm"]), value["pair_id"]))
    central = sorted((item for item in items if item["image_zone"] == "central"), key=lambda value: (float(value["mean_right_ankle_radius_norm"]), value["pair_id"]))
    unknown = sorted((item for item in items if item["image_zone"] == "unknown"), key=lambda value: value["pair_id"])
    groups = [edge, central, unknown]
    selected = []
    group_index = 0
    while len(selected) < count and any(groups):
        for _ in range(len(groups)):
            group = groups[group_index % len(groups)]
            group_index += 1
            if group:
                selected.append(group.pop(0))
                break
        else:
            break
    return selected


def select_items(items: list[dict], target_pairs: int) -> tuple[list[dict], dict]:
    buckets = ("high_reprojection_failure", "valid_control", "other_failure")
    available = Counter(item["outcome"] for item in items)
    other_target = min(8, available["other_failure"], target_pairs)
    remaining = target_pairs - other_target
    target_by_outcome = {
        "high_reprojection_failure": min(available["high_reprojection_failure"], remaining // 2),
        "valid_control": min(available["valid_control"], remaining - remaining // 2),
        "other_failure": other_target,
    }
    # Preserve the approximately even valid/failure contrast, then fill only
    # from unselected candidates if an uncommon outcome has too few records.
    for name in ("high_reprojection_failure", "valid_control", "other_failure"):
        if target_by_outcome[name] < 0:
            target_by_outcome[name] = 0
    selected: list[dict] = []
    selected_ids: set[int] = set()
    time_names = ("early_far_to_near", "middle_far_to_near", "late_far_to_near")
    for outcome in buckets:
        candidates = [item for item in items if item["outcome"] == outcome]
        quotas = distribute(target_by_outcome[outcome])
        for time_name, quota in zip(time_names, quotas):
            choices = select_diverse([item for item in candidates if temporal_bin(item, len(items)) == time_name], quota)
            for item in choices:
                if item["pair_id"] not in selected_ids:
                    item["temporal_bin"] = time_name
                    selected.append(item)
                    selected_ids.add(item["pair_id"])
    # A stratum can be short.  Fill from the same category first, ordered by
    # outcome target and then by pair id, without silently duplicating frames.
    for outcome in buckets:
        desired = target_by_outcome[outcome]
        current = sum(item["outcome"] == outcome for item in selected)
        if current >= desired:
            continue
        remaining_items = [item for item in items if item["outcome"] == outcome and item["pair_id"] not in selected_ids]
        for item in select_diverse(remaining_items, desired - current):
            item["temporal_bin"] = temporal_bin(item, len(items))
            selected.append(item)
            selected_ids.add(item["pair_id"])
    if len(selected) < target_pairs:
        for item in sorted((item for item in items if item["pair_id"] not in selected_ids), key=lambda value: (value["outcome"], value["pair_id"])):
            item["temporal_bin"] = temporal_bin(item, len(items))
            selected.append(item)
            selected_ids.add(item["pair_id"])
            if len(selected) == target_pairs:
                break
    selected = sorted(selected[:target_pairs], key=lambda value: value["pair_id"])
    for item in selected:
        item.setdefault("temporal_bin", temporal_bin(item, len(items)))
        item["review_filename"] = f"pair_{item['pair_id']:03d}_L{item['left_frame_id']:04d}_R{item['right_frame_id']:04d}.jpg"
    summary = {
        "available_by_outcome": dict(available),
        "requested_target_pairs": target_pairs,
        "requested_by_outcome": target_by_outcome,
        "selected_by_outcome": dict(Counter(item["outcome"] for item in selected)),
        "selected_by_temporal_bin": dict(Counter(item["temporal_bin"] for item in selected)),
        "selected_by_image_zone": dict(Counter(item["image_zone"] for item in selected)),
        "selected_by_outcome_and_zone": {
            f"{outcome}|{zone}": count
            for (outcome, zone), count in sorted(Counter((item["outcome"], item["image_zone"]) for item in selected).items())
        },
    }
    return selected, summary


def read_frames(video_path: Path, required: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    frames = {}
    index = 0
    try:
        while required and True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in required:
                frames[index] = frame.copy()
                required.remove(index)
            index += 1
    finally:
        capture.release()
    if required:
        raise RuntimeError(f"Missing requested frame IDs in {video_path}: {sorted(required)}")
    return frames


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def annotation_rows(selected: list[dict], image_size: tuple[int, int]) -> list[dict]:
    width, height = image_size
    rows = []
    for item in selected:
        for camera, frame_key in (("left", "left_frame_id"), ("right", "right_frame_id")):
            row = {
                "sample_id": f"pair_{item['pair_id']:03d}_{camera}",
                "pair_id": item["pair_id"],
                "camera": camera,
                "raw_frame_id": item[frame_key],
                "raw_width_px": width,
                "raw_height_px": height,
                "review_filename": item["review_filename"],
                "reviewer_id": "",
                "reviewed_at_bjt": "",
                "scene_tags": "",
                "notes": "",
            }
            for landmark in LANDMARKS:
                row[f"{landmark}_x_px"] = ""
                row[f"{landmark}_y_px"] = ""
                row[f"{landmark}_visibility"] = ""
            rows.append(row)
    return rows


def write_readme(run_dir: Path, image_size: tuple[int, int], selected_count: int) -> None:
    run_dir.joinpath("README.md").write_text(
        "# 脚部专项 2D 评估集：盲标注说明\n\n"
        f"本目录包含 {selected_count} 对 B0 固定重放帧的人工标注清单。图像原始坐标均为 {image_size[0]} x {image_size[1]}。\n\n"
        "## 标注顺序（必须遵守）\n\n"
        "1. 先只打开 `review_pairs/` 和 `annotation_template.csv`；不要在标注阶段查看 `analysis_context.csv`。\n"
        "2. 每个相机图像分别填写左/右髋、膝、踝、脚尖；脚尖定义为鞋/脚在前进方向最远端的可见点。脚跟为可选。\n"
        "3. `visibility` 仅填 `visible`、`occluded`、`outside`、`ambiguous` 之一。看不清时不得猜测坐标。\n"
        "4. 图像未叠加 B0 点。若需像素级复核，使用 `raw_frame_id` 从原始 AVI 提取同一帧；评估坐标始终是原始 1920 x 1080 坐标。\n"
        "5. 完成盲标后，再把人工标注与 `analysis_context.csv` 合并，计算左右 2D 误差、双侧可见率、三角化成功率和重投影误差。\n\n"
        "## 结论边界\n\n"
        "该目录只是评估集与标注模板，尚未包含人工标注或任何误差指标。`analysis_context.csv` 的 B0 成败标签不可作为真值。\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.target_pairs <= 0:
        raise ValueError("target-pairs must be positive")
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing output: {run_dir}")
    if args.resume and (run_dir / "selection_summary.json").exists():
        raise FileExistsError(f"Refusing to resume completed output: {run_dir}")
    calibration = StereoCalibration.load(CALIBRATION)
    if calibration.left_image_size != calibration.right_image_size:
        raise ValueError("This preparation script requires equal left/right image sizes")
    records = load_records()
    items = [item_from_record(record, index, calibration.left_image_size) for index, record in enumerate(records)]
    selected, summary = select_items(items, min(args.target_pairs, len(items)))
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    review_dir = run_dir / "review_pairs"
    review_dir.mkdir(exist_ok=args.resume)
    left_frames = read_frames(RAW_CAPTURE / "left_capture.avi", {item["left_frame_id"] for item in selected})
    right_frames = read_frames(RAW_CAPTURE / "right_capture.avi", {item["right_frame_id"] for item in selected})
    for item in selected:
        left = left_frames[item["left_frame_id"]]
        right = right_frames[item["right_frame_id"]]
        if left.shape != right.shape:
            raise RuntimeError(f"Stereo frame size mismatch for pair {item['pair_id']}")
        combined = np.hstack([left, right])
        output = review_dir / item["review_filename"]
        if output.is_file():
            continue
        if not cv2.imwrite(str(output), combined, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"Could not write {output}")
    analysis_fields = [
        "pair_id", "sequence_index", "temporal_bin", "left_frame_id", "right_frame_id", "pair_timestamp_sec",
        "outcome", "right_ankle_b0_reason", "right_ankle_b0_valid", "right_ankle_b0_reprojection_error_px",
        "left_ankle_b0_xy_score", "right_ankle_b0_xy_score", "mean_right_ankle_radius_norm", "image_zone", "review_filename",
    ]
    write_csv(run_dir / "analysis_context.csv", selected, analysis_fields)
    template = annotation_rows(selected, calibration.left_image_size)
    template_fields = list(template[0]) if template else []
    write_csv(run_dir / "annotation_template.csv", template, template_fields)
    write_readme(run_dir, calibration.left_image_size, len(selected))
    summary.update(
        {
            "classification": "dataset_preparation",
            "validation_id": "V20260828-F4_foot_evaluation_set_manifest",
            "input_capture_id": "R20260826-01_far_to_near_domain_capture",
            "reference_run": "E20260827-B0_replay_baseline",
            "input_pair_count": len(records),
            "selected_pair_count": len(selected),
            "annotation_row_count": len(template),
            "unique_variable": "add fixed stratified blind human-annotation selection only; no model rerun",
            "success_criterion": "approximately 200 unique stereo pairs with blank bilateral hip/knee/ankle/toe labels and separate B0 context",
            "caveat": "No human labels or model-accuracy claims exist yet. Temporal thirds are an acquisition-process proxy, not physical distance ground truth.",
        }
    )
    (run_dir / "selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    run_dir.joinpath("EXPERIMENT.md").write_text(
        "# V20260828-F4 — 脚部专项评估集抽样与盲标注清单\n\n"
        "- 分类：`dataset_preparation`，不运行任何模型。\n"
        "- 输入：B0 的 389 对固定重放记录与原始 AVI。\n"
        "- 唯一变量：增加固定、分层、盲标注优先的人工评估集。\n"
        f"- 结果：选出 {len(selected)} 对，生成 {len(template)} 个单相机标注行及不带 B0 叠加的原始双图。详见 `selection_summary.json`。\n\n"
        "## 结论边界\n\n"
        "该清单没有人工真值，也没有任何 2D/3D 精度结论。时间分层只代表该远—近采集过程的时序位置，不是量测距离。\n",
        encoding="utf-8",
    )
    command = "python .\\tools\\build_foot_evaluation_set.py"
    if args.resume:
        command += " --resume"
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
