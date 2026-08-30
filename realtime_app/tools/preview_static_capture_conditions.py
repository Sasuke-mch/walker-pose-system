"""Create one unannotated stereo contact sheet for each static distance capture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_ROOT = PROJECT_ROOT / "research_records" / "raw_captures" / "R20260826-01_far_to_near_domain_capture"
SEQUENCES = {
    "far_static": CAPTURE_ROOT / "far_static" / "20260826_195305_083",
    "mid_static": CAPTURE_ROOT / "mid_static" / "20260826_195344_142",
    "near_static": CAPTURE_ROOT / "near_static" / "20260826_195421_326",
}
DEFAULT_RUN_DIR = PROJECT_ROOT / "research_records" / "engineering_validation" / "V20260829-F5_static_capture_coverage_preview"


def frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if count <= 0:
        raise RuntimeError(f"No frames available: {path}")
    return count


def read_frame(path: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok or image is None:
        raise RuntimeError(f"Could not read {path} at frame {index}")
    return image


def label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (430, 56), (0, 0, 0), -1)
    cv2.putText(result, text, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 255), 2, cv2.LINE_AA)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {run_dir}")
    run_dir.mkdir(parents=True)
    rows, pairs = [], []
    for name, folder in SEQUENCES.items():
        left_path, right_path = folder / "left_capture.avi", folder / "right_capture.avi"
        left_count, right_count = frame_count(left_path), frame_count(right_path)
        left_index, right_index = left_count // 2, right_count // 2
        left, right = read_frame(left_path, left_index), read_frame(right_path, right_index)
        if left.shape != right.shape:
            raise RuntimeError(f"Stereo image sizes differ for {name}: {left.shape} vs {right.shape}")
        left_labeled = label(left, f"{name} | left | frame {left_index}/{left_count - 1}")
        right_labeled = label(right, f"{name} | right | frame {right_index}/{right_count - 1}")
        pairs.append(np.hstack([left_labeled, right_labeled]))
        rows.append(
            {
                "sequence": name,
                "left_frame_count": left_count,
                "right_frame_count": right_count,
                "left_mid_frame": left_index,
                "right_mid_frame": right_index,
                "frame_size_px": [int(left.shape[1]), int(left.shape[0])],
                "source_folder": str(folder),
            }
        )
    preview = np.vstack(pairs)
    cv2.imwrite(str(run_dir / "static_stereo_midframe_preview.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
    payload = {
        "classification": "engineering_validation",
        "validation_id": "V20260829-F5_static_capture_coverage_preview",
        "input_capture_id": "R20260826-01_far_to_near_domain_capture",
        "unique_variable": "static distance condition (far/mid/near) visual audit only",
        "frames": rows,
        "success_criterion": "three raw stereo mid-frame pairs are exported for manual foot-field review",
        "caveat": "No pose, depth, calibration residual, physical distance, or 3-D accuracy result was computed.",
    }
    (run_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    run_dir.joinpath("EXPERIMENT.md").write_text(
        "# V20260829-F5 — 静态距离段脚部视场审计\n\n"
        "仅导出 far/mid/near 三段采集的左右中位原始帧，供人工审查脚部可见性与画面位置。"
        "未运行任何模型，不能从此预览推断距离、关键点精度或 3D 精度。\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
