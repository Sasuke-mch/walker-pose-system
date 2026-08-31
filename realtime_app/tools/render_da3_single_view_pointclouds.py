#!/usr/bin/env python3
"""Render separate three-view visualizations for existing DA3 camera-coordinate PLY clouds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VIEW_SPECS = ((20, -65, "front-right"), (18, 25, "front-left"), (82, -90, "top"))
COLOR_BY_VIEW = {
    "left": (51 / 255, 153 / 255, 255 / 255),
    "right": (255 / 255, 102 / 255, 102 / 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-ply", type=Path, required=True)
    parser.add_argument("--right-ply", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_ascii_ply(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="ascii").splitlines()
    try:
        end_header = lines.index("end_header")
    except ValueError as exc:
        raise RuntimeError(f"Missing end_header in {path}") from exc
    vertex_line = next((line for line in lines[:end_header] if line.startswith("element vertex ")), None)
    if vertex_line is None:
        raise RuntimeError(f"Missing vertex count in {path}")
    expected = int(vertex_line.split()[-1])
    values = np.loadtxt(lines[end_header + 1 :], dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or values.shape[0] != expected:
        raise RuntimeError(f"Unexpected PLY table at {path}: {values.shape}, expected ({expected}, 6)")
    if not np.isfinite(values[:, :3]).all():
        raise RuntimeError(f"Non-finite coordinates in {path}")
    return values[:, :3]


def set_equal_axes(axis: plt.Axes, points: np.ndarray) -> None:
    lo = np.min(points, axis=0)
    hi = np.max(points, axis=0)
    center = (lo + hi) / 2.0
    radius = max(float(np.max(hi - lo)) / 2.0, 1e-4)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def render(path: Path, points: np.ndarray, view_name: str) -> None:
    figure = plt.figure(figsize=(15, 5.1), layout="constrained")
    color = COLOR_BY_VIEW[view_name]
    for index, (elevation, azimuth, title) in enumerate(VIEW_SPECS, 1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.8, c=[color], alpha=0.78, linewidths=0)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("X (DA3 relative unit)", fontsize=9)
        axis.set_ylabel("Y (DA3 relative unit)", fontsize=9)
        axis.set_zlabel("Z (DA3 relative unit)", fontsize=9)
        set_equal_axes(axis, points)
    figure.suptitle(
        f"near_1p3m/pair_045 — {view_name} camera cloud only ({len(points)} points; DA3 relative camera coordinates)",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    left_path = args.left_ply.resolve()
    right_path = args.right_ply.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    left_points = read_ascii_ply(left_path)
    right_points = read_ascii_ply(right_path)
    if len(left_points) != 8820 or len(right_points) != 8820:
        raise RuntimeError(f"Expected 8820 points per cloud, got {len(left_points)} and {len(right_points)}")
    render(output_root / "near_1p3m" / "pair_045" / "pair_045_left_camera_three_views.png", left_points, "left")
    render(output_root / "near_1p3m" / "pair_045" / "pair_045_right_camera_three_views.png", right_points, "right")
    readme = (
        "# pair_045 左右单独 DA3 点云图\n\n"
        "左图只包含左相机输入得到的蓝色点，右图只包含右相机输入得到的红色点。"
        "每张 PNG 内的三幅子图只是同一份单相机点云的三个观察方向。\n\n"
        "两个单相机点云分别处于各自 DA3 相机坐标中，深度单位为相对单位，"
        "未使用本项目鱼眼标定；因此两张图之间不能量距离、量尺度或直接判断标定误差。\n"
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    metadata = {
        "source_left_ply": str(left_path),
        "source_right_ply": str(right_path),
        "pair": "near_1p3m/pair_045",
        "left_points": int(len(left_points)),
        "right_points": int(len(right_points)),
        "coordinate_interpretation": "separate DA3 relative camera coordinates; not calibrated fisheye or metric geometry",
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"left_points": len(left_points), "right_points": len(right_points), "png": 2, "output_root": str(output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
