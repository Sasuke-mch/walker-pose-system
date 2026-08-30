"""Create a comparable M1 five-model 2-D/3-D summary from saved outputs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from tools.evaluate_offline_stereo_predictions import _records  # noqa: E402


MODELS = ("yolo26x_pose", "pmpose", "probpose", "bboxmaskpose", "sapiens2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-selection", required=True, type=Path)
    parser.add_argument("--five-model-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.25)
    return parser.parse_args()


def top_person_ankle(records: list[dict], threshold: float) -> dict[str, dict[str, bool]]:
    result = {}
    for frame in records:
        people = frame["persons"]
        if not people:
            result[frame["name"]] = {"person": False, "right_ankle": False}
            continue
        primary = max(people, key=lambda person: person.bbox_score)
        result[frame["name"]] = {
            "person": True,
            "right_ankle": len(primary.keypoints) > 16 and primary.keypoints[16][2] >= threshold,
        }
    return result


def main() -> int:
    args = parse_args()
    selection = args.input_selection.resolve()
    root = args.five_model_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    with (selection / "selection_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != 60:
        raise RuntimeError("Expected the fixed 60-pair M1 selection")
    condition_by_name = {row["file_name"]: row["condition"] for row in manifest}
    if set(condition_by_name.values()) != {"far_3m", "mid_2m", "near_1p3m"}:
        raise RuntimeError("Unexpected M1 distance conditions")

    report = {
        "experiment_id": "E20260829-M1_five_model_direct2d_and_fisheye_triangulation",
        "input_selection": str(selection),
        "five_model_root": str(root),
        "models": {},
        "interpretation_boundary": "2-D detection coverage and calibrated-stereo self-consistency only. No manual 2-D annotation or 3-D ground truth is present; rates must not be called accuracy.",
    }
    for model in MODELS:
        left_path = root / "left_cw90" / "run_new_bright_60" / model / "raw_predictions.json"
        right_path = root / "right_ccw90" / "run_new_bright_60" / model / "raw_predictions.json"
        result_path = root / "stereo_geometry" / model / "offline_stereo_results.jsonl"
        left = {row["name"]: row for row in _records(left_path, model, "raw_keypoint")}
        right = {row["name"]: row for row in _records(right_path, model, "raw_keypoint")}
        if set(left) != set(condition_by_name) or set(right) != set(condition_by_name):
            raise RuntimeError(f"{model}: prediction names do not match the fixed selection")
        left_flags = top_person_ankle(list(left.values()), args.threshold)
        right_flags = top_person_ankle(list(right.values()), args.threshold)
        groups: dict[str, dict] = defaultdict(lambda: {
            "pairs": 0, "left_top_person": 0, "right_top_person": 0,
            "left_top_right_ankle_2d": 0, "right_top_right_ankle_2d": 0,
            "bilateral_top_right_ankle_2d": 0, "right_ankle_valid_3d": 0,
            "right_ankle_rejections": Counter(), "right_ankle_reprojection_errors_px": [],
            "valid_3d_keypoints": 0,
        })
        for name in condition_by_name:
            group = groups[condition_by_name[name]]
            group["pairs"] += 1
            group["left_top_person"] += int(left_flags[name]["person"])
            group["right_top_person"] += int(right_flags[name]["person"])
            group["left_top_right_ankle_2d"] += int(left_flags[name]["right_ankle"])
            group["right_top_right_ankle_2d"] += int(right_flags[name]["right_ankle"])
            group["bilateral_top_right_ankle_2d"] += int(left_flags[name]["right_ankle"] and right_flags[name]["right_ankle"])
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            group = groups[condition_by_name[item["file_name"]]]
            for person in item["persons_3d"]:
                group["valid_3d_keypoints"] += int(person["valid_keypoints"])
            if not item["persons_3d"]:
                group["right_ankle_rejections"]["no_stereo_person"] += 1
                continue
            ankle = next((point for point in item["persons_3d"][0]["keypoints_3d"] if point["index"] == 16), None)
            if ankle is None:
                group["right_ankle_rejections"]["missing_right_ankle"] += 1
                continue
            if ankle["valid"]:
                group["right_ankle_valid_3d"] += 1
            else:
                group["right_ankle_rejections"][str(ankle["reason"])] += 1
            if ankle.get("reprojection_error_mean_px") is not None:
                group["right_ankle_reprojection_errors_px"].append(float(ankle["reprojection_error_mean_px"]))
        model_groups = {}
        for condition in ("far_3m", "mid_2m", "near_1p3m"):
            group = groups[condition]
            pairs = group["pairs"]
            model_groups[condition] = {
                "pairs": pairs,
                "left_top_person_rate": group["left_top_person"] / pairs,
                "right_top_person_rate": group["right_top_person"] / pairs,
                "left_top_right_ankle_2d_rate": group["left_top_right_ankle_2d"] / pairs,
                "right_top_right_ankle_2d_rate": group["right_top_right_ankle_2d"] / pairs,
                "bilateral_top_right_ankle_2d_rate": group["bilateral_top_right_ankle_2d"] / pairs,
                "right_ankle_valid_3d_rate": group["right_ankle_valid_3d"] / pairs,
                "right_ankle_rejections": dict(group["right_ankle_rejections"]),
                "right_ankle_mean_reprojection_error_px_when_available": (
                    float(np.mean(group["right_ankle_reprojection_errors_px"]))
                    if group["right_ankle_reprojection_errors_px"] else None
                ),
                "valid_3d_keypoints_total": group["valid_3d_keypoints"],
            }
        report["models"][model] = {"conditions": model_groups}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
