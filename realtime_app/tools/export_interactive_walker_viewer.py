#!/usr/bin/env python3
"""Export the current unsmoothed ground-frame run for the public web viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def joint_rows(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    people = row.get("persons_3d") or []
    if len(people) != 1:
        return {}
    return {int(point["index"]): point for point in people[0].get("keypoints_3d", [])}


def transform_point(point: np.ndarray, dynamic: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(dynamic["rotation_world_from_left_camera"], dtype=np.float64)
    translation = np.asarray(dynamic["translation_world_from_left_camera_mm"], dtype=np.float64)
    return rotation @ point + translation


def visual_points_ground(
    dynamic: dict[str, Any], strict: dict[str, Any], visual: dict[str, Any]
) -> dict[int, np.ndarray]:
    camera_points: dict[int, np.ndarray] = {}
    for index, item in joint_rows(visual).items():
        xyz = np.asarray(item.get("xyz", []), dtype=np.float64)
        if xyz.shape == (3,) and np.all(np.isfinite(xyz)):
            camera_points[index] = xyz
    for index, item in joint_rows(strict).items():
        xyz = np.asarray(item.get("xyz", []), dtype=np.float64)
        if bool(item.get("valid")) and xyz.shape == (3,) and np.all(np.isfinite(xyz)):
            camera_points[index] = xyz
    return {index: transform_point(point, dynamic) for index, point in camera_points.items()}


def rounded(value: Any, digits: int = 3) -> Any:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [rounded(item, digits) for item in value]
    return round(float(value), digits)


def make_frame(
    dynamic: dict[str, Any],
    strict: dict[str, Any],
    visual: dict[str, Any],
    interaction: dict[str, Any],
) -> dict[str, Any]:
    pair_ids = {
        int(dynamic["pair_id"]), int(strict["pair_id"]),
        int(visual["pair_id"]), int(interaction["pair_id"]),
    }
    if len(pair_ids) != 1:
        raise ValueError(f"unaligned frame streams: {sorted(pair_ids)}")
    points = visual_points_ground(dynamic, strict, visual)
    errors = interaction.get("per_keypoint_reprojection_error_mean_px") or {}
    provenance = interaction.get("per_keypoint_provenance") or {}
    hands = interaction.get("hands") or {}
    walker_ground = interaction.get("walker_ground") or {}
    return {
        "pair": int(dynamic["pair_id"]),
        "stage": dynamic["stage"],
        "points_mm": [rounded(points[index]) if index in points else None for index in range(17)],
        "errors_px": [
            None if str(index) not in errors else rounded(errors[str(index)], 2)
            for index in range(17)
        ],
        "strict": [
            bool((provenance.get(str(index)) or {}).get("strict_valid"))
            for index in range(17)
        ],
        "rotation_world_from_left": rounded(dynamic["rotation_world_from_left_camera"], 7),
        "translation_world_from_left_mm": rounded(dynamic["translation_world_from_left_camera_mm"]),
        "wrist_handle_mm": {
            side: (
                None
                if (hands.get(side) or {}).get("wrist_to_handle_axis_distance_mm") is None
                else rounded(hands[side]["wrist_to_handle_axis_distance_mm"], 1)
            )
            for side in ("left", "right")
        },
        "walker_feet_max_abs_z_mm": (
            None
            if walker_ground.get("maximum_absolute_foot_height_mm") is None
            else rounded(walker_ground["maximum_absolute_foot_height_mm"], 1)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic-ground-jsonl", type=Path, required=True)
    parser.add_argument("--strict-stereo-jsonl", type=Path, required=True)
    parser.add_argument("--visual-stereo-jsonl", type=Path, required=True)
    parser.add_argument("--interaction-jsonl", type=Path, required=True)
    parser.add_argument("--walker-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    streams = [
        read_jsonl(args.dynamic_ground_jsonl),
        read_jsonl(args.strict_stereo_jsonl),
        read_jsonl(args.visual_stereo_jsonl),
        read_jsonl(args.interaction_jsonl),
    ]
    lengths = {len(stream) for stream in streams}
    if len(lengths) != 1 or not streams[0]:
        raise ValueError(f"input streams must be non-empty and aligned, got lengths {sorted(lengths)}")
    walker = json.loads(args.walker_model.read_text(encoding="utf-8-sig"))
    frames = [make_frame(*rows) for rows in zip(*streams)]
    payload = {
        "schema_version": "walker_interactive_viewer_v1",
        "title": "双目助步器人体运动可视化",
        "fps": args.fps,
        "frame_count": len(frames),
        "length_unit": "millimeter",
        "display_smoothing": "none",
        "line_style": "solid_only",
        "coordinate_frame": "fixed_ground_z_up",
        "limitations": (
            "Engineering visualization without external trajectory truth. Hand and foot states are geometric proximity candidates, not tactile or load contact."
        ),
        "skeleton_edges": [
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12],
            [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 1],
            [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
        ],
        "walker": {
            "nodes_left_camera_mm": walker["nodes_left_camera_mm"],
            "edges": walker["edges"],
            "handle_segments": walker["handle_segments"],
            "foot_nodes": walker["foot_nodes"],
            "camera_centers_left_camera_mm": walker["camera_centers_left_camera_mm"],
            "dimensions_mm": walker["dimensions_mm"],
            "fit_audit": {
                "horizontal_residual_median_mm": walker["fit_audit"]["horizontal_residual_median_mm"],
                "horizontal_residual_max_mm": walker["fit_audit"]["horizontal_residual_max_mm"],
            },
        },
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve()), "frames": len(frames)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
