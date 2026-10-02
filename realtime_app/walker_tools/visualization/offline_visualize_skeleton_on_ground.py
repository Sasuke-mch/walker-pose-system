#!/usr/bin/env python3
"""Render strict stereo skeletons above one fixed ChArUco-derived floor plane."""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
from walker_tools._compat import APP_ROOT as _tool_app_root
_tool_prepare_imports()

import argparse
import bisect
import json
from pathlib import Path
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


APP_DIR = _tool_app_root

from pose_app.static_ground_reference import StaticGroundReference  # noqa: E402


COCO_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)
LEFT_JOINTS = {5, 7, 9, 11, 13, 15}
RIGHT_JOINTS = {6, 8, 10, 12, 14, 16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-stereo-jsonl", type=Path, required=True)
    parser.add_argument(
        "--display-completion-jsonl",
        type=Path,
        help=(
            "Optional force-all stereo results used only to complete the drawing. "
            "Missing points after that are filled temporally and remain labelled as display-only."
        ),
    )
    parser.add_argument("--ground-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--stationary-setup-confirmed",
        action="store_true",
        help="Required acknowledgement that the walker and cameras did not move after board capture.",
    )
    args = parser.parse_args()
    if not args.stationary_setup_confirmed:
        parser.error("--stationary-setup-confirmed is required for a fixed ground frame")
    if args.fps <= 0 or args.frame_step < 1:
        parser.error("fps must be positive and frame-step must be at least one")
    return args


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("strict stereo input is empty")
    return rows


def read_image(path: Path) -> np.ndarray | None:
    """Read a rendered image without relying on OpenCV's Windows Unicode path support."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None


def transform_row(row: dict, reference: StaticGroundReference) -> dict:
    people = row.get("persons_3d") or []
    if len(people) != 1:
        points = []
        status = "no_single_person"
    else:
        points = []
        for point in people[0].get("keypoints_3d", []):
            xyz = np.asarray(point.get("xyz", []), dtype=np.float64)
            if not bool(point.get("valid")) or xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
                continue
            ground = reference.transform(xyz)
            points.append({
                "index": int(point["index"]),
                "name": str(point.get("name", point["index"])),
                "xyz_ground_mm": ground.tolist(),
                "height_above_ground_mm": float(ground[2]),
                "score": point.get("score"),
                "reprojection_error_mean_px": point.get("reprojection_error_mean_px"),
                "provenance": "strict_valid",
                "display_completion_only": False,
            })
        status = "rendered" if points else "no_valid_3d_keypoints"
    return {
        "pair_id": int(row["pair_id"]),
        "pair_timestamp_sec": row.get("pair_timestamp_sec"),
        "status": status,
        "coordinate_frame": "fixed_ground",
        "length_unit": "millimeter",
        "points": points,
    }


def _finite_point_map(row: dict, *, require_valid: bool) -> dict[int, dict]:
    people = row.get("persons_3d") or []
    if len(people) != 1:
        return {}
    result: dict[int, dict] = {}
    for point in people[0].get("keypoints_3d", []):
        xyz = np.asarray(point.get("xyz", []), dtype=np.float64)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            continue
        if require_valid and not bool(point.get("valid")):
            continue
        result[int(point["index"])] = point
    return result


def complete_display_records(
    strict_rows: list[dict],
    completion_rows: list[dict],
    reference: StaticGroundReference,
) -> list[dict]:
    """Build a fully connected display sequence without upgrading geometric evidence."""
    if len(strict_rows) != len(completion_rows):
        raise ValueError("strict and display-completion row counts differ")
    records: list[dict] = []
    for strict_row, completion_row in zip(strict_rows, completion_rows):
        pair_id = int(strict_row["pair_id"])
        if pair_id != int(completion_row["pair_id"]):
            raise ValueError("strict and display-completion pair IDs differ")
        strict_map = _finite_point_map(strict_row, require_valid=True)
        completion_map = _finite_point_map(completion_row, require_valid=True)
        points: list[dict] = []
        for joint_index in range(17):
            source = strict_map.get(joint_index)
            provenance = "strict_valid"
            if source is None:
                source = completion_map.get(joint_index)
                provenance = "force_all_candidate"
            if source is None:
                continue
            ground = reference.transform(source["xyz"])
            points.append({
                "index": joint_index,
                "name": str(source.get("name", joint_index)),
                "xyz_ground_mm": ground.tolist(),
                "height_above_ground_mm": float(ground[2]),
                "score": source.get("score"),
                "reprojection_error_mean_px": source.get("reprojection_error_mean_px"),
                "provenance": provenance,
                "display_completion_only": provenance != "strict_valid",
            })
        records.append({
            "pair_id": pair_id,
            "pair_timestamp_sec": strict_row.get("pair_timestamp_sec"),
            "status": "display_completion_pending",
            "coordinate_frame": "fixed_ground",
            "length_unit": "millimeter",
            "strict_valid_joint_count": len(strict_map),
            "points": points,
        })

    for joint_index in range(17):
        available = [
            frame_index
            for frame_index, record in enumerate(records)
            if any(point["index"] == joint_index for point in record["points"])
        ]
        if not available:
            raise ValueError(f"joint {joint_index} has no finite 3-D source in the sequence")
        for frame_index, record in enumerate(records):
            if any(point["index"] == joint_index for point in record["points"]):
                continue
            position = bisect.bisect_left(available, frame_index)
            if position == 0:
                left_index = right_index = available[0]
                provenance = "nearest_temporal_extrapolation"
            elif position == len(available):
                left_index = right_index = available[-1]
                provenance = "nearest_temporal_extrapolation"
            else:
                left_index, right_index = available[position - 1], available[position]
                provenance = "linear_temporal_interpolation"
            left_point = next(
                point for point in records[left_index]["points"] if point["index"] == joint_index
            )
            right_point = next(
                point for point in records[right_index]["points"] if point["index"] == joint_index
            )
            if left_index == right_index:
                xyz = np.asarray(left_point["xyz_ground_mm"], dtype=np.float64)
            else:
                weight = (frame_index - left_index) / (right_index - left_index)
                xyz = (
                    (1.0 - weight) * np.asarray(left_point["xyz_ground_mm"], dtype=np.float64)
                    + weight * np.asarray(right_point["xyz_ground_mm"], dtype=np.float64)
                )
            record["points"].append({
                "index": joint_index,
                "name": str(left_point.get("name", right_point.get("name", joint_index))),
                "xyz_ground_mm": xyz.tolist(),
                "height_above_ground_mm": float(xyz[2]),
                "score": None,
                "reprojection_error_mean_px": None,
                "provenance": provenance,
                "display_completion_only": True,
            })

    for record in records:
        record["points"].sort(key=lambda point: point["index"])
        if len(record["points"]) != 17:
            raise RuntimeError(f"pair {record['pair_id']} is not complete after display filling")
        record["status"] = "rendered_complete_display"
    return records


def axis_limits(records: list[dict]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    points = np.asarray(
        [point["xyz_ground_mm"] for record in records for point in record["points"]],
        dtype=np.float64,
    )
    if not len(points):
        raise ValueError("no valid 3-D keypoints to render")
    x_min, y_min = np.percentile(points[:, :2], 1, axis=0) - 250.0
    x_max, y_max = np.percentile(points[:, :2], 99, axis=0) + 250.0
    z_min = min(-50.0, float(np.percentile(points[:, 2], 1) - 100.0))
    z_max = max(500.0, float(np.percentile(points[:, 2], 99) + 150.0))
    return (float(x_min), float(x_max)), (float(y_min), float(y_max)), (z_min, z_max)


def render_frame(
    record: dict,
    history: list[dict],
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    path: Path,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(8.0, 6.0), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    x_lim, y_lim, z_lim = limits
    gx = np.linspace(x_lim[0], x_lim[1], 12)
    gy = np.linspace(y_lim[0], y_lim[1], 12)
    xx, yy = np.meshgrid(gx, gy)
    ax.plot_surface(xx, yy, np.zeros_like(xx), color="#d9d9d9", alpha=0.42, linewidth=0)
    for x in gx:
        ax.plot([x, x], y_lim, [0, 0], color="#a6a6a6", linewidth=0.45, alpha=0.6)
    for y in gy:
        ax.plot(x_lim, [y, y], [0, 0], color="#a6a6a6", linewidth=0.45, alpha=0.6)

    point_by_index = {point["index"]: point for point in record["points"]}
    by_index = {
        index: np.asarray(point["xyz_ground_mm"])
        for index, point in point_by_index.items()
    }
    for start, end in COCO_EDGES:
        if start in by_index and end in by_index:
            segment = np.vstack([by_index[start], by_index[end]])
            completed = (
                point_by_index[start].get("display_completion_only", False)
                or point_by_index[end].get("display_completion_only", False)
            )
            ax.plot(
                segment[:, 0], segment[:, 1], segment[:, 2],
                color="#777777" if completed else "#222222",
                linewidth=2.0 if completed else 2.4,
                linestyle="--" if completed else "-",
            )
    for index, xyz in by_index.items():
        color = "#0072B2" if index in LEFT_JOINTS else "#D55E00" if index in RIGHT_JOINTS else "#222222"
        completed = point_by_index[index].get("display_completion_only", False)
        ax.scatter(
            *xyz,
            color=color if not completed else "white",
            edgecolors=color,
            linewidths=1.4,
            s=28,
            depthshade=False,
        )

    for joint_index, color in ((15, "#0072B2"), (16, "#D55E00")):
        trail = []
        for old in history[-90:]:
            point_map = {point["index"]: point for point in old["points"]}
            if joint_index in point_map:
                xyz = point_map[joint_index]["xyz_ground_mm"]
                trail.append([xyz[0], xyz[1], 2.0])
        if len(trail) >= 2:
            trail_array = np.asarray(trail)
            ax.plot(trail_array[:, 0], trail_array[:, 1], trail_array[:, 2], color=color, linewidth=1.5, alpha=0.75)

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_xlabel("Ground X (mm)")
    ax.set_ylabel("Ground Y (mm)")
    ax.set_zlabel("Height (mm)")
    if "strict_valid_joint_count" in record:
        title = (
            f"pair {record['pair_id']:04d}   complete skeleton 17/17   "
            f"strict {record['strict_valid_joint_count']}/17"
        )
    else:
        title = f"pair {record['pair_id']:04d}   valid joints {len(record['points'])}/17"
    ax.set_title(title)
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((x_lim[1] - x_lim[0], y_lim[1] - y_lim[0], z_lim[1] - z_lim[0]))
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True)
    reference = StaticGroundReference.load(args.ground_reference)
    source_rows = load_rows(args.strict_stereo_jsonl)
    if args.display_completion_jsonl is None:
        transformed = [transform_row(row, reference) for row in source_rows]
    else:
        completion_rows = load_rows(args.display_completion_jsonl)
        transformed = complete_display_records(source_rows, completion_rows, reference)
    limits = axis_limits(transformed)
    selected = transformed[:: args.frame_step]
    frame_paths = []
    for index, record in enumerate(selected):
        path = frames_dir / f"frame_{index:06d}_pair_{record['pair_id']:06d}.png"
        source_index = index * args.frame_step
        render_frame(record, transformed[: source_index + 1], limits, path, args.dpi)
        frame_paths.append(path)
    first = read_image(frame_paths[0])
    if first is None:
        raise RuntimeError("failed to read rendered frame")
    video_name = (
        "skeleton_on_fixed_ground_complete.mp4"
        if args.display_completion_jsonl is not None
        else "skeleton_on_fixed_ground.mp4"
    )
    video_path = output / video_name
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps / args.frame_step,
        (first.shape[1], first.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError("failed to open MP4 writer")
    try:
        for path in frame_paths:
            image = read_image(path)
            if image is None or image.shape != first.shape:
                raise RuntimeError(f"invalid rendered frame: {path}")
            writer.write(image)
    finally:
        writer.release()
    with (output / "ground_relative_skeleton.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in transformed:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    joint_counts = [len(record["points"]) for record in transformed]
    ankle_heights = [
        point["height_above_ground_mm"]
        for record in transformed
        for point in record["points"]
        if point["index"] in (15, 16)
    ]
    provenance_counts: dict[str, int] = {}
    for record in transformed:
        for point in record["points"]:
            provenance = str(point.get("provenance", "unknown"))
            provenance_counts[provenance] = provenance_counts.get(provenance, 0) + 1
    summary = {
        "schema_version": (
            "fixed_ground_complete_display_visualization_v1"
            if args.display_completion_jsonl is not None
            else "fixed_ground_skeleton_visualization_v1"
        ),
        "source_frames": len(source_rows),
        "rendered_frames": len(selected),
        "frame_step": args.frame_step,
        "valid_joint_count_median": float(np.median(joint_counts)),
        "all_frames_complete_17_joints": all(count == 17 for count in joint_counts),
        "display_point_provenance_counts": provenance_counts,
        "display_completion_source": (
            str(args.display_completion_jsonl.resolve())
            if args.display_completion_jsonl is not None
            else None
        ),
        "valid_ankle_observations": len(ankle_heights),
        "ankle_height_median_mm": float(np.median(ankle_heights)) if ankle_heights else None,
        "ground_reference": str(args.ground_reference.resolve()),
        "stationary_setup_confirmed_by_operator": True,
        "video": str(video_path),
        "interpretation": (
            "Complete display: strict points are solid; force-all or temporally filled points are "
            "display-only and dashed. Filled points are not upgraded to accepted geometry."
            if args.display_completion_jsonl is not None
            else "Visualization of accepted stereo keypoints in one fixed floor frame. "
            "Ankle height is not a shoe-sole contact measurement and no gait event is assigned."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
