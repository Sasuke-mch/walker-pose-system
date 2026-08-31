#!/usr/bin/env python3
"""Export qualitative DA3-relative point-cloud galleries from saved D5 NPZ files.

The exported merged clouds use DA3's own predicted world-to-camera poses. They
are relative model-coordinate visualizations, not calibrated fisheye geometry
or metric scene reconstructions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAIR_SPECS = (
    ("far_3m", "pair_005"),
    ("far_3m", "pair_015"),
    ("mid_2m", "pair_025"),
    ("mid_2m", "pair_035"),
    ("near_1p3m", "pair_045"),
    ("near_1p3m", "pair_055"),
)
VIEW_NAMES = ("left", "right")
VIEW_COLORS = np.asarray(((51, 153, 255), (255, 102, 102)), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4)
    return parser.parse_args()


def percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("No finite numeric values")
    return float(np.percentile(finite, q))


def camera_to_world(points_camera: np.ndarray, w2c_3x4: np.ndarray) -> np.ndarray:
    rotation = np.asarray(w2c_3x4[:, :3], dtype=np.float64)
    translation = np.asarray(w2c_3x4[:, 3], dtype=np.float64)
    return (points_camera - translation[None, :]) @ rotation


def backproject(depth: np.ndarray, intrinsics: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    zz = depth[yy, xx].astype(np.float64)
    valid = np.isfinite(zz) & (zz > 0.0)
    if not np.any(valid):
        raise RuntimeError("No positive finite sampled depth values")
    xx = xx[valid].astype(np.float64)
    yy = yy[valid].astype(np.float64)
    zz = zz[valid]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise RuntimeError(f"Invalid focal lengths fx={fx}, fy={fy}")
    points = np.column_stack(((xx - cx) * zz / fx, (yy - cy) * zz / fy, zz))
    return points, zz


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if points.shape != colors.shape or points.shape[1] != 3:
        raise ValueError("PLY points/colors must both have shape (N, 3)")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join((
        "ply", "format ascii 1.0", f"element vertex {len(points)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue", "end_header",
    ))
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(header + "\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def set_equal_axes(axis: plt.Axes, points: np.ndarray) -> None:
    lo = np.min(points, axis=0)
    hi = np.max(points, axis=0)
    center = (lo + hi) / 2.0
    radius = max(float(np.max(hi - lo)) / 2.0, 1e-4)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def render_gallery(path: Path, merged: np.ndarray, view_ids: np.ndarray, pair_label: str) -> None:
    figure = plt.figure(figsize=(14, 4.8), constrained_layout=True)
    views = ((20, -65, "front-right"), (18, 25, "front-left"), (82, -90, "top"))
    for index, (elevation, azimuth, label) in enumerate(views, 1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        for view_id, name in enumerate(VIEW_NAMES):
            mask = view_ids == view_id
            axis.scatter(
                merged[mask, 0], merged[mask, 1], merged[mask, 2],
                s=0.45, c=(VIEW_COLORS[view_id] / 255.0)[None, :],
                alpha=0.62, linewidths=0, label=name,
            )
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(label)
        axis.set_xlabel("X (DA3 relative unit)")
        axis.set_ylabel("Y (DA3 relative unit)")
        axis.set_zlabel("Z (DA3 relative unit)")
        set_equal_axes(axis, merged)
        if index == 1:
            axis.legend(loc="upper left")
    figure.suptitle(f"{pair_label}: DA3 predicted-coordinate relative point cloud")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_readme(path: Path) -> None:
    path.write_text(
        "# DA3 正向鱼眼相对点云图库\n\n"
        "本目录从 D5 已保存的 DA3 `results.npz` 导出点云。每对图像包含左右相机各自的相机坐标点云、"
        "以及使用 DA3 自己预测 world-to-camera 位姿变换后的合并点云。\n\n"
        "`*_left_camera.ply` 和 `*_right_camera.ply` 分别在各自相机坐标系中。"
        "`*_da3_predicted_world_merged.ply` 在 DA3 预测坐标系中，蓝色为左视图、红色为右视图。"
        "`*_three_views.png` 给出同一合并点云的三个观察方向。\n\n"
        "深度单位为 DA3 相对单位；点云没有使用本项目鱼眼标定，也不是米制点云。DA3 的内部 `conf` "
        "未用于筛点或概率解释，仅在 `pointcloud_summary.csv` 中保留原始分位数。\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    total_points = 0
    for condition, pair_name in PAIR_SPECS:
        npz_path = input_root / condition / pair_name / "exports" / "mini_npz" / "results.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        archive = np.load(npz_path)
        required = {"depth", "conf", "intrinsics", "extrinsics"}
        if set(archive.files) != required:
            raise RuntimeError(f"Unexpected NPZ keys at {npz_path}: {archive.files}")
        depth = np.asarray(archive["depth"])
        conf = np.asarray(archive["conf"])
        intrinsics = np.asarray(archive["intrinsics"])
        extrinsics = np.asarray(archive["extrinsics"])
        if depth.shape != (2, 504, 280) or conf.shape != depth.shape or intrinsics.shape != (2, 3, 3) or extrinsics.shape != (2, 3, 4):
            raise RuntimeError(f"Unexpected DA3 array shapes at {npz_path}")
        pair_output = output_root / condition / pair_name
        per_view_world: list[np.ndarray] = []
        for view_id, view_name in enumerate(VIEW_NAMES):
            camera_points, sampled_depth = backproject(depth[view_id], intrinsics[view_id], args.stride)
            world_points = camera_to_world(camera_points, extrinsics[view_id])
            color = np.repeat(VIEW_COLORS[view_id][None, :], len(camera_points), axis=0)
            write_ply(pair_output / f"{pair_name}_{view_name}_camera.ply", camera_points, color)
            per_view_world.append(world_points)
            rows.append({
                "condition": condition, "pair": pair_name, "view": view_name,
                "sample_stride": args.stride, "point_count": len(camera_points),
                "depth_p01": percentile(sampled_depth, 1), "depth_p50": percentile(sampled_depth, 50), "depth_p99": percentile(sampled_depth, 99),
                "conf_p01": percentile(conf[view_id], 1), "conf_p50": percentile(conf[view_id], 50), "conf_p99": percentile(conf[view_id], 99),
                "coordinate_note": "camera coordinates for this individual DA3 view",
            })
        merged = np.vstack(per_view_world)
        view_ids = np.concatenate((np.zeros(len(per_view_world[0]), dtype=np.int8), np.ones(len(per_view_world[1]), dtype=np.int8)))
        merged_colors = VIEW_COLORS[view_ids]
        write_ply(pair_output / f"{pair_name}_da3_predicted_world_merged.ply", merged, merged_colors)
        render_gallery(pair_output / f"{pair_name}_three_views.png", merged, view_ids, f"{condition}/{pair_name}")
        total_points += len(merged)
    csv_path = output_root / "pointcloud_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "source": str(input_root),
        "pairs": [{"condition": condition, "pair": pair} for condition, pair in PAIR_SPECS],
        "sample_stride": args.stride,
        "total_points": total_points,
        "depth_interpretation": "DA3 relative depth only",
        "extrinsics_interpretation": "DA3-predicted world-to-camera only",
        "merged_cloud_interpretation": "qualitative relative point cloud in DA3 predicted coordinates; not calibrated fisheye, metric reconstruction, ground truth, or a joint-fusion input",
        "confidence_interpretation": "DA3 internal output recorded only as raw quantiles; not a probability and not used to filter points",
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(output_root / "README.md")
    print(json.dumps({"pairs": len(PAIR_SPECS), "views": len(rows), "total_points": total_points, "output_root": str(output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
