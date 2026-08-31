#!/usr/bin/env python3
"""Render readable DA3 relative-depth diagnostics from saved D5 outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


PAIR_SPECS = (
    ("far_3m", "pair_005"),
    ("far_3m", "pair_015"),
    ("mid_2m", "pair_025"),
    ("mid_2m", "pair_035"),
    ("near_1p3m", "pair_045"),
    ("near_1p3m", "pair_055"),
)
VIEW_SPECS = (("left", "left_ccw90", 0), ("right", "right_cw90", 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--da3-root", type=Path, required=True)
    parser.add_argument("--input-selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def percentile_pair(values: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("No finite values available for normalization")
    lo, hi = np.percentile(finite, (low, high)).astype(np.float32)
    if not float(hi) > float(lo):
        hi = lo + np.float32(1e-6)
    return float(lo), float(hi)


def normalize_percentile(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    lo, hi = percentile_pair(values)
    normalized = np.clip((values.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return normalized, lo, hi


def colorize(normalized: np.ndarray, palette: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    image = cv2.applyColorMap(np.rint(normalized * 255.0).astype(np.uint8), palette)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_map(values: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(values.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)


def depth_edges(depth_normalized: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(depth_normalized, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth_normalized, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(dx * dx + dy * dy)
    edge_norm, _, _ = normalize_percentile(magnitude)
    return edge_norm


def lower_region(height: int, width: int) -> tuple[int, int, int, int]:
    return 0, int(round(height * 0.52)), width, height


def blend(rgb: np.ndarray, colored: np.ndarray, alpha: float = 0.46) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) * (1.0 - alpha) + colored.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def add_axis_image(axis: plt.Axes, image: np.ndarray, title: str) -> None:
    axis.imshow(image)
    axis.set_title(title, fontsize=10)
    axis.set_axis_off()


def render_view(
    path: Path,
    condition: str,
    pair_name: str,
    view_name: str,
    rgb: np.ndarray,
    full_depth: np.ndarray,
    full_conf: np.ndarray,
) -> dict[str, float | int | str]:
    height, width = rgb.shape[:2]
    depth_norm, depth_lo, depth_hi = normalize_percentile(full_depth)
    conf_norm, conf_lo, conf_hi = normalize_percentile(full_conf)
    depth_color = colorize(depth_norm)
    conf_color = colorize(conf_norm, cv2.COLORMAP_VIRIDIS)
    full_overlay = blend(rgb, depth_color)
    edge_color = colorize(depth_edges(depth_norm), cv2.COLORMAP_MAGMA)
    x0, y0, x1, y1 = lower_region(height, width)
    lower_rgb = rgb[y0:y1, x0:x1]
    lower_depth = depth_norm[y0:y1, x0:x1]
    lower_depth_color = depth_color[y0:y1, x0:x1]
    lower_band = np.floor(lower_depth * 5.0).clip(0, 4).astype(np.float32) / 4.0
    lower_band_color = colorize(lower_band, cv2.COLORMAP_TURBO)
    lower_overlay = blend(lower_rgb, lower_band_color, alpha=0.52)

    figure, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    axes[0, 0].imshow(rgb)
    axes[0, 0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#00ff66", linewidth=3))
    axes[0, 0].set_title("DA3 input: upright raw fisheye", fontsize=10)
    axes[0, 0].set_axis_off()
    add_axis_image(axes[0, 1], depth_color, "DA3 relative-depth value")
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(vmin=depth_lo, vmax=depth_hi)),
        ax=axes[0, 1], fraction=0.046, pad=0.02,
    )
    colorbar.set_label("relative depth value (not metres)", fontsize=8)
    add_axis_image(axes[0, 2], full_overlay, "raw image + relative depth")
    add_axis_image(axes[0, 3], conf_color, "DA3 internal conf map (not probability)")
    add_axis_image(axes[1, 0], lower_rgb, "lower-image region from green box")
    add_axis_image(axes[1, 1], lower_depth_color, "lower-region relative depth")
    add_axis_image(axes[1, 2], lower_overlay, "lower-region depth bands over raw")
    add_axis_image(axes[1, 3], edge_color[y0:y1, x0:x1], "lower-region relative-depth edges")
    figure.suptitle(
        f"{condition}/{pair_name}/{view_name} — DA3 diagnostic; raw fisheye input, relative output",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    return {
        "condition": condition,
        "pair": pair_name,
        "view": view_name,
        "input_width": width,
        "input_height": height,
        "depth_p02": depth_lo,
        "depth_p50": float(np.percentile(full_depth, 50)),
        "depth_p98": depth_hi,
        "conf_p02": conf_lo,
        "conf_p50": float(np.percentile(full_conf, 50)),
        "conf_p98": conf_hi,
        "lower_region": f"x=[{x0},{x1}), y=[{y0},{y1})",
    }


def write_readme(path: Path) -> None:
    path.write_text(
        "# DA3 原始鱼眼深度诊断图组\n\n"
        "每张 `*_depth_diagnostics.png` 对应一张已经转正后送入 DA3 的原始鱼眼图。"
        "上排依次为原图、DA3 相对深度、原图与相对深度叠加、DA3 内部 conf 图；"
        "下排为原图下部区域、该区域的相对深度、五档相对深度带叠加和相对深度边缘。\n\n"
        "彩色深度图表示当前图像内部的相对深度数值，颜色不表示米制距离；不同图之间也不能直接比较颜色或数值。"
        "绿色框仅为固定的图像下部检查区域，不是人体或脚部检测框。`conf` 是 DA3 内部输出，"
        "不是概率，也没有用于筛选点或判断关键点是否正确。\n\n"
        "这些图可用于观察前后分层、深度边缘和可能的遮挡位置，不能替代鱼眼标定、左右对应、双目三角化或三维关节精度评估。\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    da3_root = args.da3_root.resolve()
    input_root = args.input_selection_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {output_root}")
    output_root.mkdir(parents=True)
    rows: list[dict[str, float | int | str]] = []
    for condition, pair_name in PAIR_SPECS:
        npz_path = da3_root / condition / pair_name / "exports" / "mini_npz" / "results.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        with np.load(npz_path) as archive:
            if set(archive.files) != {"depth", "conf", "intrinsics", "extrinsics"}:
                raise RuntimeError(f"Unexpected NPZ keys at {npz_path}: {archive.files}")
            depth = np.asarray(archive["depth"], dtype=np.float32)
            conf = np.asarray(archive["conf"], dtype=np.float32)
        if depth.shape != (2, 504, 280) or conf.shape != depth.shape:
            raise RuntimeError(f"Unexpected DA3 depth/conf shapes at {npz_path}: {depth.shape}, {conf.shape}")
        pair_output = output_root / condition / pair_name
        for view_name, image_directory, view_index in VIEW_SPECS:
            image_path = input_root / image_directory / f"{pair_name}.png"
            rgb = load_rgb(image_path)
            height, width = rgb.shape[:2]
            full_depth = resize_map(depth[view_index], width, height)
            full_conf = resize_map(conf[view_index], width, height)
            row = render_view(
                pair_output / f"{pair_name}_{view_name}_depth_diagnostics.png",
                condition, pair_name, view_name, rgb, full_depth, full_conf,
            )
            rows.append(row)
    csv_path = output_root / "depth_diagnostic_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "source_da3_root": str(da3_root),
        "source_upright_input_root": str(input_root),
        "pairs": [{"condition": condition, "pair": pair_name} for condition, pair_name in PAIR_SPECS],
        "views": ["left", "right"],
        "output_interpretation": "per-image qualitative relative-depth diagnostics only",
        "depth_interpretation": "relative depth value; not metres and not comparable across images",
        "confidence_interpretation": "raw DA3 internal output visualized only; not a probability and not used for filtering",
        "lower_region_interpretation": "fixed lower-image diagnostic area, not a person or foot detector",
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(output_root / "README.md")
    print(json.dumps({"pairs": len(PAIR_SPECS), "views": len(rows), "png": len(rows), "output_root": str(output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
