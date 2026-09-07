#!/usr/bin/env python3
"""基于A10数据集的DA3深度引导选择性去畸变实验

对比方案：
- C0 (A10基线): 原始YOLO框 + 无去畸变
- C1 (A10扩展): 固定扩展框(15%水平+25%底部) + 无去畸变
- C2 (保守DA3): 轻度DA3引导框扩展 + 轻度选择性去畸变(intensity=0.4)
- C3 (激进DA3): 强力DA3引导框扩展 + 强力选择性去畸变(intensity=0.8)

目标：验证DA3深度先验是否能在中近距离进一步改善下肢关键点识别
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
    parser.add_argument("--a10-root", type=Path, required=True,
                        help="A10实验根目录")
    parser.add_argument("--da3-root", type=Path, required=True,
                        help="DA3深度图根目录")
    parser.add_argument("--calibration", type=Path, required=True,
                        help="鱼眼标定JSON文件")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="输出目录")
    parser.add_argument("--mode", choices=["conservative", "aggressive"], required=True,
                        help="实验模式：conservative(C2)或aggressive(C3)")
    parser.add_argument("--distance", choices=["far_3m", "mid_2m", "near_1p3m", "all"], default="all",
                        help="处理哪个距离条件")
    return parser.parse_args()


def load_calibration(path: Path) -> dict[str, Any]:
    """加载标定参数"""
    with path.open("r", encoding="utf-8-sig") as f:
        calib = json.load(f)

    # 加载cam0和cam1的内参
    calib_dir = path.parent
    cam0_path = calib_dir / Path(calib["cam0_intrinsics"]).name
    cam1_path = calib_dir / Path(calib["cam1_intrinsics"]).name

    with cam0_path.open("r", encoding="utf-8-sig") as f:
        cam0 = json.load(f)
    with cam1_path.open("r", encoding="utf-8-sig") as f:
        cam1 = json.load(f)

    return {
        "left_K": np.asarray(cam0["K"], dtype=np.float64),
        "left_D": np.asarray(cam0["D"], dtype=np.float64).reshape(-1, 1),
        "right_K": np.asarray(cam1["K"], dtype=np.float64),
        "right_D": np.asarray(cam1["D"], dtype=np.float64).reshape(-1, 1),
        "image_size": tuple(calib["image_size"]),
    }


def create_undistort_maps(K: np.ndarray, D: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """创建鱼眼去畸变映射"""
    width, height = size
    identity = np.eye(3, dtype=np.float64)
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        K, D, identity, K, (width, height), cv2.CV_32FC1
    )
    return map_x, map_y


def load_da3_depth(da3_root: Path, distance: str) -> tuple[np.ndarray, np.ndarray]:
    """加载DA3深度图（左右视图）

    DA3输出格式：depth shape (2, 280, 504)，需要resize到原图尺寸1920x1080
    """
    npz_path = da3_root / distance / "exports" / "mini_npz" / "results.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"DA3深度文件不存在: {npz_path}")

    data = np.load(npz_path)
    depth = data["depth"]  # shape: (2, 280, 504)

    # Resize到原图尺寸
    left_depth = cv2.resize(depth[0], (1920, 1080), interpolation=cv2.INTER_LINEAR)
    right_depth = cv2.resize(depth[1], (1920, 1080), interpolation=cv2.INTER_LINEAR)

    return left_depth, right_depth


def compute_depth_guided_mask(
    depth: np.ndarray,
    bbox: list[float],
    mode: str,
) -> np.ndarray:
    """计算深度引导的去畸变mask

    Args:
        depth: DA3深度图 (H, W)
        bbox: 检测框 [x1, y1, x2, y2]
        mode: "conservative" 或 "aggressive"

    Returns:
        mask: 去畸变权重图 (H, W)，范围[0, 1]
    """
    height, width = depth.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    box_height = y2 - y1

    # 扩展到下方（脚部区域）
    if mode == "conservative":
        # 保守：只扩展30%高度
        y2_extended = min(height, y2 + int(box_height * 0.3))
        threshold_percentile = 65  # 更严格的前景阈值
    else:  # aggressive
        # 激进：扩展50%高度
        y2_extended = min(height, y2 + int(box_height * 0.5))
        threshold_percentile = 55  # 更宽松的前景阈值

    # 采样扩展区域内的深度
    roi_depth = depth[y1:y2_extended, x1:x2]
    valid_depth = roi_depth[np.isfinite(roi_depth)]

    if len(valid_depth) < 10:
        # 深度数据不足，返回基于几何的fallback mask
        return create_geometric_mask(depth.shape, bbox, mode)

    # 计算深度阈值（前景物体通常深度值较小）
    depth_threshold = np.percentile(valid_depth, threshold_percentile)

    # 创建二值mask
    binary_mask = np.zeros(depth.shape, dtype=np.float32)
    foreground = (depth < depth_threshold) & np.isfinite(depth)
    binary_mask[foreground] = 1.0

    # 形态学闭运算填充空洞
    kernel_size = 21 if mode == "conservative" else 31
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # 高斯模糊创建平滑过渡
    blur_size = 51 if mode == "conservative" else 71
    smooth_mask = cv2.GaussianBlur(binary_mask, (blur_size, blur_size), 15)

    # 重点关注框下半部分和下方
    lower_y_start = y1 + int(box_height * 0.5)  # 从50%高度开始
    region_mask = np.zeros(depth.shape, dtype=np.float32)

    # 创建渐变：上方权重低，下方权重高
    for y in range(lower_y_start, y2_extended):
        if y >= height:
            break
        progress = (y - lower_y_start) / max(1, y2_extended - lower_y_start)
        region_mask[y, x1:x2] = progress

    return smooth_mask * region_mask


def create_geometric_mask(shape: tuple[int, int], bbox: list[float], mode: str) -> np.ndarray:
    """当DA3深度不可用时的fallback mask"""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.float32)

    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    box_height = y2 - y1

    # 重点关注下半部分
    lower_y_start = y1 + int(box_height * 0.5)
    extend_ratio = 0.3 if mode == "conservative" else 0.5
    lower_y_end = min(height, y2 + int(box_height * extend_ratio))

    # 创建渐变mask
    for y in range(lower_y_start, lower_y_end):
        weight = (y - lower_y_start) / max(1, lower_y_end - lower_y_start)
        mask[y, x1:x2] = weight

    blur_size = 51 if mode == "conservative" else 71
    return cv2.GaussianBlur(mask, (blur_size, blur_size), 15)


def expand_bbox_with_depth(
    bbox: list[float],
    depth: np.ndarray,
    mode: str,
) -> list[float]:
    """基于深度信息动态扩展检测框

    Args:
        bbox: 原始YOLO检测框 [x1, y1, x2, y2]
        depth: DA3深度图
        mode: "conservative" 或 "aggressive"

    Returns:
        expanded_bbox: 扩展后的检测框
    """
    height, width = depth.shape
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1

    # 基础水平和顶部扩展（与C1相同）
    new_x1 = max(0.0, x1 - 0.15 * bw)
    new_y1 = max(0.0, y1 - 0.10 * bh)
    new_x2 = min(float(width - 1), x2 + 0.15 * bw)

    # 底部扩展：基于深度连续性
    x1_int = max(0, int(x1))
    x2_int = min(width, int(x2))
    y2_int = min(height, int(y2))

    # 采样框内下部的深度作为人体参考
    ref_y_start = max(0, int(y1 + bh * 0.6))
    ref_depth_roi = depth[ref_y_start:y2_int, x1_int:x2_int]
    valid_ref = ref_depth_roi[np.isfinite(ref_depth_roi)]

    if len(valid_ref) > 10:
        ref_depth_median = np.median(valid_ref)
        ref_depth_std = np.std(valid_ref) + 1e-6

        # 向下搜索深度相近的区域
        if mode == "conservative":
            max_extension = min(int(bh * 0.35), height - y2_int)  # 最多35%
            std_threshold = 1.5  # 1.5个标准差内
        else:  # aggressive
            max_extension = min(int(bh * 0.6), height - y2_int)  # 最多60%
            std_threshold = 2.5  # 2.5个标准差内

        bottom_extension = 0
        for dy in range(0, max_extension, 5):
            search_y = y2_int + dy
            if search_y >= height:
                break

            search_roi = depth[search_y, x1_int:x2_int]
            valid_search = search_roi[np.isfinite(search_roi)]

            if len(valid_search) > 5:
                search_median = np.median(valid_search)
                if abs(search_median - ref_depth_median) < std_threshold * ref_depth_std:
                    bottom_extension = dy
                else:
                    break

        # 至少扩展25%（与C1一致），最多到深度连续位置
        bottom_extension = max(int(bh * 0.25), bottom_extension)
        new_y2 = min(float(height - 1), y2 + bottom_extension)
    else:
        # 深度数据不足，使用固定扩展
        if mode == "conservative":
            new_y2 = min(float(height - 1), y2 + 0.30 * bh)
        else:
            new_y2 = min(float(height - 1), y2 + 0.40 * bh)

    return [new_x1, new_y1, new_x2, new_y2]


def apply_selective_undistort(
    image: np.ndarray,
    mask: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    intensity: float,
) -> np.ndarray:
    """应用选择性去畸变"""
    # 完全去畸变的图像
    undistorted = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    # 根据mask和intensity混合
    effective_mask = (mask * intensity)[:, :, np.newaxis]
    result = (
        image.astype(np.float32) * (1.0 - effective_mask) +
        undistorted.astype(np.float32) * effective_mask
    ).astype(np.uint8)

    return result


def visualize_comparison(
    original: np.ndarray,
    depth: np.ndarray,
    bbox_c0: list[float],
    bbox_c1: list[float],
    bbox_c2: list[float],
    mask: np.ndarray,
    undistorted: np.ndarray,
    pair_name: str,
    side: str,
) -> np.ndarray:
    """创建C0/C1/C2/C3对比可视化"""
    h, w = original.shape[:2]

    # 绘制三种检测框
    img_with_boxes = original.copy()
    for bbox, color, label in [
        (bbox_c0, (0, 0, 255), "C0_original"),
        (bbox_c1, (0, 255, 255), "C1_fixed"),
        (bbox_c2, (0, 255, 0), "C2/C3_DA3")
    ]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_with_boxes, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 深度可视化
    depth_vis = np.zeros((h, w, 3), dtype=np.uint8)
    valid_depth = depth[np.isfinite(depth)]
    if len(valid_depth) > 0:
        depth_norm = np.clip(depth, np.percentile(valid_depth, 1), np.percentile(valid_depth, 99))
        depth_norm = ((depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min() + 1e-8) * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        depth_vis[~np.isfinite(depth)] = 0

    # mask可视化
    mask_vis = (mask * 255).astype(np.uint8)
    mask_vis = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)

    # 创建2x2拼图
    scale = 0.4
    new_h, new_w = int(h * scale), int(w * scale)

    top = np.hstack([
        cv2.resize(img_with_boxes, (new_w, new_h)),
        cv2.resize(depth_vis, (new_w, new_h))
    ])
    bottom = np.hstack([
        cv2.resize(mask_vis, (new_w, new_h)),
        cv2.resize(undistorted, (new_w, new_h))
    ])

    result = np.vstack([top, bottom])

    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, f"{pair_name} {side} - BBox Comparison", (10, 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(result, "DA3 Depth", (new_w + 10, 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(result, "Undistort Mask", (10, new_h + 25), font, 0.7, (255, 255, 255), 2)
    cv2.putText(result, "Result", (new_w + 10, new_h + 25), font, 0.7, (255, 255, 255), 2)

    return result


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在: {output_dir}")

    output_dir.mkdir(parents=True)

    # 确定实验参数
    if args.mode == "conservative":
        condition_name = "C2_conservative_da3"
        undistort_intensity = 0.4
        description = "保守DA3引导：轻度框扩展(最多35%) + 轻度去畸变(0.4)"
    else:  # aggressive
        condition_name = "C3_aggressive_da3"
        undistort_intensity = 0.8
        description = "激进DA3引导：强力框扩展(最多60%) + 强力去畸变(0.8)"

    print(f"实验模式: {condition_name}")
    print(f"去畸变强度: {undistort_intensity}")

    # 加载标定
    calib = load_calibration(args.calibration)

    # 创建去畸变映射
    left_map_x, left_map_y = create_undistort_maps(
        calib["left_K"], calib["left_D"], calib["image_size"]
    )
    right_map_x, right_map_y = create_undistort_maps(
        calib["right_K"], calib["right_D"], calib["image_size"]
    )

    # 确定要处理的距离条件
    if args.distance == "all":
        distances = ["far_3m", "mid_2m", "near_1p3m"]
    else:
        distances = [args.distance]

    # A10使用M2的输入图像（直立后的图像）
    # M2目录结构：input_selection/left_ccw90, input_selection/right_cw90
    m2_root = Path("D:/my_works/walker_pose_system/research_records/screening/E20260829-M2_bright_60pair_upright_2d")

    # 从M1获取selection manifest（有距离条件信息）
    m1_root = Path("D:/my_works/walker_pose_system/research_records/screening/E20260829-M1_bright_60pair_5model")
    selection_csv = m1_root / "input_selection" / "selection_manifest.csv"

    if not selection_csv.exists():
        raise FileNotFoundError(f"M1选择清单不存在: {selection_csv}")

    # 读取清单
    pairs_by_distance = {"far_3m": [], "mid_2m": [], "near_1p3m": []}
    with selection_csv.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            condition = row["condition"]
            if condition in pairs_by_distance:
                pairs_by_distance[condition].append(row)

    a10_root = args.a10_root.resolve()

    results = []

    for distance in distances:
        print(f"\n处理距离条件: {distance}")

        # 加载DA3深度
        try:
            left_depth, right_depth = load_da3_depth(args.da3_root, distance)
        except FileNotFoundError as e:
            print(f"  跳过 {distance}: {e}")
            continue

        # 加载C0和C1的检测框
        c0_left_json = a10_root / "left" / "detections" / "c0_yolo_top1.json"
        c1_left_json = a10_root / "left" / "detections" / "c1_yolo_expanded.json"
        c0_right_json = a10_root / "right" / "detections" / "c0_yolo_top1.json"
        c1_right_json = a10_root / "right" / "detections" / "c1_yolo_expanded.json"

        with c0_left_json.open("r", encoding="utf-8") as f:
            c0_left_data = json.load(f)
        with c1_left_json.open("r", encoding="utf-8") as f:
            c1_left_data = json.load(f)
        with c0_right_json.open("r", encoding="utf-8") as f:
            c0_right_data = json.load(f)
        with c1_right_json.open("r", encoding="utf-8") as f:
            c1_right_data = json.load(f)

        # 创建文件名到检测框的映射
        c0_left_boxes = {img["file_name"]: img["detections"][0]["bbox_xyxy"] if img["detections"] else None
                         for img in c0_left_data["images"]}
        c1_left_boxes = {img["file_name"]: img["detections"][0]["bbox_xyxy"] if img["detections"] else None
                         for img in c1_left_data["images"]}
        c0_right_boxes = {img["file_name"]: img["detections"][0]["bbox_xyxy"] if img["detections"] else None
                          for img in c0_right_data["images"]}
        c1_right_boxes = {img["file_name"]: img["detections"][0]["bbox_xyxy"] if img["detections"] else None
                          for img in c1_right_data["images"]}

        # 处理该距离的所有pair
        for pair_info in pairs_by_distance[distance]:
            pair_name = pair_info["file_name"]
            print(f"  处理 {pair_name}")

            # 加载M2的直立图像（已旋转）
            # 注意：这些是直立后的图像，需要反向旋转回鱼眼坐标
            left_img_path = m2_root / "input_selection" / "left_ccw90" / pair_name
            right_img_path = m2_root / "input_selection" / "right_cw90" / pair_name

            if not left_img_path.exists() or not right_img_path.exists():
                print(f"    警告: 图像文件不存在")
                continue

            # 读取直立图像
            left_img_upright = cv2.imread(str(left_img_path))
            right_img_upright = cv2.imread(str(right_img_path))

            # 反向旋转回原始鱼眼方向（left是逆时针90°旋转的，需要顺时针90°转回）
            left_img = cv2.rotate(left_img_upright, cv2.ROTATE_90_CLOCKWISE)
            right_img = cv2.rotate(right_img_upright, cv2.ROTATE_90_COUNTERCLOCKWISE)

            pair_result = {"pair_name": pair_name, "distance": distance, "views": {}}

            for side, img, depth, map_x, map_y, c0_boxes, c1_boxes in [
                ("left", left_img, left_depth, left_map_x, left_map_y, c0_left_boxes, c1_left_boxes),
                ("right", right_img, right_depth, right_map_x, right_map_y, c0_right_boxes, c1_right_boxes)
            ]:
                bbox_c0 = c0_boxes.get(pair_name)
                bbox_c1 = c1_boxes.get(pair_name)

                if bbox_c0 is None or bbox_c1 is None:
                    print(f"    {side}: 无检测框")
                    continue

                # 步骤1: DA3引导的框扩展
                bbox_c2 = expand_bbox_with_depth(bbox_c0, depth, args.mode)

                # 步骤2: 计算深度引导的去畸变mask
                mask = compute_depth_guided_mask(depth, bbox_c2, args.mode)

                # 步骤3: 应用选择性去畸变
                undistorted = apply_selective_undistort(
                    img, mask, map_x, map_y, undistort_intensity
                )

                # 保存去畸变后的图像（用于后续PMPose/Sapiens2处理）
                side_dir = output_dir / side / "undistorted_images"
                side_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(side_dir / pair_name), undistorted)

                # 保存检测框JSON（用于后续姿态估计）
                bbox_json_dir = output_dir / side / "detections"
                bbox_json_dir.mkdir(parents=True, exist_ok=True)

                # 创建可视化
                vis_dir = output_dir / side / "visualizations"
                vis_dir.mkdir(parents=True, exist_ok=True)
                vis = visualize_comparison(
                    img, depth, bbox_c0, bbox_c1, bbox_c2, mask, undistorted,
                    pair_name, side
                )
                cv2.imwrite(str(vis_dir / pair_name.replace(".png", "_process.jpg")), vis)

                # 记录统计
                bbox_c0_area = (bbox_c0[2] - bbox_c0[0]) * (bbox_c0[3] - bbox_c0[1])
                bbox_c1_area = (bbox_c1[2] - bbox_c1[0]) * (bbox_c1[3] - bbox_c1[1])
                bbox_c2_area = (bbox_c2[2] - bbox_c2[0]) * (bbox_c2[3] - bbox_c2[1])

                pair_result["views"][side] = {
                    "bbox_c0": bbox_c0,
                    "bbox_c1": bbox_c1,
                    "bbox_c2": bbox_c2,
                    "bbox_c0_area": float(bbox_c0_area),
                    "bbox_c1_area": float(bbox_c1_area),
                    "bbox_c2_area": float(bbox_c2_area),
                    "c2_vs_c0_area_ratio": float(bbox_c2_area / bbox_c0_area),
                    "c2_vs_c1_area_ratio": float(bbox_c2_area / bbox_c1_area),
                    "mask_coverage": float(mask.sum() / mask.size),
                }

            results.append(pair_result)

    # 保存检测框JSON（统一格式，用于后续实验）
    for side in ["left", "right"]:
        bbox_json_path = output_dir / side / "detections" / f"{condition_name}_bbox.json"
        bbox_json_path.parent.mkdir(parents=True, exist_ok=True)

        images_list = []
        for result in results:
            if side in result["views"]:
                view = result["views"][side]
                images_list.append({
                    "file_name": result["pair_name"],
                    "distance": result["distance"],
                    "detections": [{
                        "class_name": "person",
                        "bbox_xyxy": view["bbox_c2"],
                        "bbox_area": view["bbox_c2_area"],
                    }]
                })

        bbox_json_data = {
            "schema_version": "a10_da3_guided_v1",
            "condition": condition_name,
            "description": description,
            "undistort_intensity": undistort_intensity,
            "images": images_list,
        }

        with bbox_json_path.open("w", encoding="utf-8") as f:
            json.dump(bbox_json_data, f, ensure_ascii=False, indent=2)

    # 保存实验元数据
    metadata = {
        "classification": "engineering_validation",
        "experiment_id": f"A10_{condition_name}_da3_guided_selective_undistort",
        "parent_experiment": "E20260830-A10_upright_yolo_box_expansion_control",
        "description": description,
        "comparison": {
            "C0": "A10原始YOLO框 + 无去畸变（基线）",
            "C1": "A10固定扩展框(15%水平+25%底部) + 无去畸变",
            f"{condition_name}": f"DA3引导框扩展 + 选择性去畸变(intensity={undistort_intensity})"
        },
        "method": {
            "step1": "基于DA3深度连续性动态扩展检测框",
            "step2": "使用深度先验识别人体下肢区域mask",
            "step3": f"对mask区域应用{undistort_intensity}强度的去畸变"
        },
        "parameters": {
            "mode": args.mode,
            "undistort_intensity": undistort_intensity,
            "conservative_max_extension": "35% of box height",
            "aggressive_max_extension": "60% of box height",
        },
        "processed_pairs": len(results),
        "results_summary": results,
    }

    with (output_dir / "experiment_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 创建EXPERIMENT.md
    (output_dir / "EXPERIMENT.md").write_text(
        f"""# {condition_name} - DA3深度引导的选择性去畸变实验

## 分类
`engineering_validation`

## 父实验
E20260830-A10_upright_yolo_box_expansion_control

## 对比条件
- **C0 (A10基线)**: 原始YOLO框 + 无去畸变
- **C1 (A10固定扩展)**: 15%水平+25%底部固定扩展 + 无去畸变
- **{condition_name}**: {description}

## 核心假设
A10证明了框扩展在中近距离有显著改善（误差下降35-61%），但远距离略有退化（+10-15%）。
本实验验证：DA3深度先验能否实现自适应扩展，并通过选择性去畸变进一步改善关节点几何。

## 方法
1. **深度引导的框扩展**:
   - 采样框内下部深度作为人体参考
   - 向下搜索深度相近区域（连续性判断）
   - 保守模式：最多35%高度，1.5σ阈值
   - 激进模式：最多60%高度，2.5σ阈值

2. **选择性去畸变**:
   - 基于深度mask识别下肢区域
   - 仅对mask权重区域应用去畸变
   - 保守模式：intensity=0.4（轻度）
   - 激进模式：intensity=0.8（强力）

3. **与C0/C1的区别**:
   - C0/C1：在原始鱼眼图上检测
   - {condition_name}：在选择性去畸变图上检测

## 参数
- 模式: {args.mode}
- 去畸变强度: {undistort_intensity}
- 处理的pair数: {len(results)}

## 输出
- `left/undistorted_images/`, `right/undistorted_images/`: 去畸变后的图像
- `left/detections/{condition_name}_bbox.json`: 检测框定义
- `left/visualizations/`, `right/visualizations/`: C0/C1/C2对比可视化

## 下一步
1. 在去畸变图像上运行PMPose/ProbPose/Sapiens2（使用{condition_name}_bbox.json中的框）
2. 与A10的C0/C1结果进行相对一致性评估（对比Sapiens2伪标签）
3. 分析各距离条件的改善/退化情况
4. 检查保守vs激进模式的权衡

## 结论边界
- DA3深度为相对深度，不是米制真值
- 评估基于Sapiens2伪标签，不是人工真值
- 去畸变改善几何但引入插值误差
- 需要在实际双目三角化中验证3D一致性
""",
        encoding="utf-8"
    )

    # 创建运行命令记录
    (output_dir / "command.txt").write_text(
        f"python .\\tools\\experiment_da3_selective_undistort_a10.py \\\n"
        f"  --a10-root \"{args.a10_root}\" \\\n"
        f"  --da3-root \"{args.da3_root}\" \\\n"
        f"  --calibration \"{args.calibration}\" \\\n"
        f"  --output-dir \"{output_dir}\" \\\n"
        f"  --mode {args.mode}\n",
        encoding="utf-8"
    )

    print(f"\n实验完成！")
    print(f"输出目录: {output_dir}")
    print(f"处理了 {len(results)} 个pairs")
    print(f"\n下一步：在去畸变图像上运行姿态估计，然后与A10的C0/C1进行对比评估")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
