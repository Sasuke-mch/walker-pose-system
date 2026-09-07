"""Build a DA3-only, raw-fisheye ROI candidate against a frozen C3 baseline.

The candidate never remaps image pixels and never shrinks the frozen C3 box.
DA3 therefore changes only a possible additional image-space ROI extent; pose
predictions remain in the standard upright-input coordinates and can still be
inverse-rotated to the original fish-eye pixels for triangulation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--c0-detections", required=True, type=Path)
    parser.add_argument("--c3-detections", required=True, type=Path)
    parser.add_argument("--da3-run-dir", required=True, type=Path)
    parser.add_argument("--condition", required=True, choices=("all", "far_3m", "mid_2m", "near_1p3m"))
    parser.add_argument("--rotation", required=True, choices=("ccw90", "cw90"))
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--visualization-dir", required=True, type=Path)
    return parser.parse_args()


def load_boxes(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, Any]] = {}
    for image in data["images"]:
        detections = image.get("detections", [])
        if len(detections) != 1:
            raise ValueError(f"Expected exactly one detection in {path}: {image['file_name']}")
        result[image["file_name"]] = {
            "box": [float(value) for value in detections[0]["bbox_xyxy"]],
            "score": float(detections[0].get("score", 1.0)),
            "width": int(image["width"]),
            "height": int(image["height"]),
            "image_id": image.get("image_id"),
            "frame_index": image.get("frame_index"),
        }
    return result


def clamp_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(float(width - 1), x1)),
        max(0.0, min(float(height - 1), y1)),
        max(0.0, min(float(width - 1), x2)),
        max(0.0, min(float(height - 1), y2)),
    ]


def union_box(first: list[float], second: list[float], width: int, height: int) -> list[float]:
    return clamp_box(
        [min(first[0], second[0]), min(first[1], second[1]), max(first[2], second[2]), max(first[3], second[3])],
        width,
        height,
    )


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def condition_for_file(file_name: str) -> str:
    index = int(Path(file_name).stem.removeprefix("pair_"))
    if 0 <= index < 20:
        return "far_3m"
    if 20 <= index < 40:
        return "mid_2m"
    if 40 <= index < 60:
        return "near_1p3m"
    raise ValueError(f"D1 expects the fixed 60-pair selection naming, got {file_name}")


def load_upright_depth(npz_path: Path, side_index: int, raw_width: int, raw_height: int, rotation: str) -> np.ndarray:
    with np.load(npz_path) as data:
        depth = np.asarray(data["depth"])[side_index]
    raw = cv2.resize(depth.astype(np.float32), (raw_width, raw_height), interpolation=cv2.INTER_LINEAR)
    return cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE if rotation == "ccw90" else cv2.ROTATE_90_CLOCKWISE)


def dynamic_box(c0: list[float], depth: np.ndarray) -> tuple[list[float], dict[str, float | int | str]]:
    """Find the depth-similar connected component touching the lower C0 band.

    The rule is frozen before inspecting the result: reference = lower 40% of
    C0; threshold = 1.5 robust sigmas around its median; component must touch
    the reference band and extend below the C0 bottom.  No distance label or
    pose output participates in the decision.
    """
    height, width = depth.shape
    x1, y1, x2, y2 = [int(round(value)) for value in clamp_box(c0, width, height)]
    box_height = max(1, y2 - y1)
    ref_start = min(y2, y1 + int(round(0.60 * box_height)))
    reference = depth[ref_start:y2, x1:x2]
    finite = reference[np.isfinite(reference)]
    diagnostics: dict[str, float | int | str] = {
        "reference_count": int(finite.size),
        "status": "no_reference",
        "reference_median": float("nan"),
        "robust_sigma": float("nan"),
        "component_pixels": 0,
    }
    if finite.size < 50:
        return c0, diagnostics
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_sigma = max(1.4826 * mad, 1e-6)
    diagnostics["reference_median"] = median
    diagnostics["robust_sigma"] = robust_sigma
    similar = (np.abs(depth - median) <= 1.5 * robust_sigma) & np.isfinite(depth)
    search_x1 = max(0, x1 - int(round(0.20 * (x2 - x1))))
    search_x2 = min(width, x2 + int(round(0.20 * (x2 - x1))))
    search_y1 = max(0, ref_start)
    search_y2 = min(height, y2 + int(round(1.30 * box_height)))
    restricted = np.zeros_like(similar, dtype=np.uint8)
    restricted[search_y1:search_y2, search_x1:search_x2] = similar[search_y1:search_y2, search_x1:search_x2].astype(np.uint8)
    restricted = cv2.morphologyEx(
        restricted,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(restricted, connectivity=8)
    best: tuple[int, np.ndarray] | None = None
    for label in range(1, count):
        sx, sy, sw, sh, pixels = stats[label]
        touches_reference = sx < x2 and sx + sw > x1 and sy < y2 and sy + sh > ref_start
        extends_below = sy + sh > y2
        if touches_reference and extends_below and pixels >= 50:
            if best is None or pixels > best[1][4]:
                best = (label, stats[label])
    if best is None:
        diagnostics["status"] = "no_connected_extension"
        return c0, diagnostics
    _, stat = best
    sx, sy, sw, sh, pixels = [int(value) for value in stat]
    margin = max(2, int(round(0.03 * box_height)))
    candidate = clamp_box([sx - margin, sy - margin, sx + sw + margin, sy + sh + margin], width, height)
    diagnostics["status"] = "component_extension"
    diagnostics["component_pixels"] = pixels
    return union_box(c0, candidate, width, height), diagnostics


def draw_visualization(image: np.ndarray, depth: np.ndarray, c0: list[float], c3: list[float], da3: list[float], final: list[float], label: str) -> np.ndarray:
    left = image.copy()
    for box, color, name in ((c0, (0, 0, 255), "C0"), (c3, (0, 255, 255), "C3 frozen"), (da3, (255, 0, 0), "DA3 raw"), (final, (0, 180, 0), "D1 safe")):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(left, (x1, y1), (x2, y2), color, 3)
        cv2.putText(left, name, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    depth_vis = cv2.applyColorMap(cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_TURBO)
    depth_vis = cv2.resize(depth_vis, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = np.hstack([left, depth_vis])
    cv2.putText(out, label, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return out


def main() -> int:
    args = parse_args()
    c0_boxes = load_boxes(args.c0_detections.resolve())
    c3_boxes = load_boxes(args.c3_detections.resolve())
    input_dir = args.input_dir.resolve()
    da3_root = args.da3_run_dir.resolve()
    output_json = args.output_json.resolve()
    manifest_csv = args.manifest_csv.resolve()
    visualization_dir = args.visualization_dir.resolve()
    if output_json.exists() or manifest_csv.exists():
        raise FileExistsError("Refusing to overwrite an existing D1 ROI output")
    visualization_dir.mkdir(parents=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for file_name in sorted(c3_boxes):
        condition = condition_for_file(file_name)
        if args.condition != "all" and condition != args.condition:
            continue
        if file_name not in c0_boxes:
            raise ValueError(f"C0 is missing {file_name}")
        image_path = input_dir / file_name
        npz_path = da3_root / condition / file_name.removesuffix(".png") / "exports" / "mini_npz" / "results.npz"
        if not image_path.is_file() or not npz_path.is_file():
            raise FileNotFoundError(f"Missing image or pairwise DA3 output for {file_name}")
        c0 = c0_boxes[file_name]
        c3 = c3_boxes[file_name]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Cannot read {image_path}")
        if (image.shape[1], image.shape[0]) != (c3["width"], c3["height"]):
            raise ValueError(f"Image geometry mismatch for {file_name}")
        side_index = 0 if "left" in output_json.as_posix().lower() else 1
        raw_height, raw_width = image.shape[1], image.shape[0]
        depth = load_upright_depth(npz_path, side_index, raw_width, raw_height, args.rotation)
        da3_box, diagnostics = dynamic_box(c0["box"], depth)
        final_box = union_box(c3["box"], da3_box, c3["width"], c3["height"])
        changed = any(abs(a - b) > 0.5 for a, b in zip(final_box, c3["box"]))
        records.append(
            {
                "image_id": c3["image_id"],
                "frame_index": c3["frame_index"],
                "file_name": file_name,
                "image_path": f"/workspace/input/{file_name}",
                "width": c3["width"],
                "height": c3["height"],
                "detections": [{"class_id": 0, "class_name": "person", "bbox_xyxy": final_box, "score": c0["score"]}],
            }
        )
        rows.append(
            {
                "file_name": file_name,
                "condition": condition,
                "rotation": args.rotation,
                "c0_area": area(c0["box"]),
                "c3_area": area(c3["box"]),
                "da3_raw_area": area(da3_box),
                "d1_safe_area": area(final_box),
                "d1_vs_c3_area_ratio": area(final_box) / area(c3["box"]),
                "changed_from_c3": changed,
                **diagnostics,
            }
        )
        vis = draw_visualization(image, depth, c0["box"], c3["box"], da3_box, final_box, f"{condition} {file_name}")
        cv2.imwrite(str(visualization_dir / file_name.replace(".png", "_d1_roi.jpg")), vis)

    output = {
        "schema_version": "da3_safe_raw_fisheye_roi_v1",
        "condition": "D1_da3_raw_roi_safe_union_c3",
        "description": "DA3-only raw-fisheye ROI candidate, unioned with frozen C3 so it cannot reduce C3 coverage",
        "input_geometry": "upright model input; DA3 depth is independently inferred on the corresponding raw fisheye pair then resized and rotated only for image-space ROI selection",
        "interpretation_boundary": "The output is an image-space ROI. DA3 depth is not used in triangulation and does not establish foot identity, keypoint correctness, or metric depth.",
        "images": records,
    }
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(records)} D1 ROI records; changed from C3: {sum(row['changed_from_c3'] for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
