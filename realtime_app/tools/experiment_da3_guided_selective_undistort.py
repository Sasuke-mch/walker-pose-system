#!/usr/bin/env python3
"""基于DA3深度图引导的选择性去畸变实验

核心思想：
1. 利用DA3深度图识别人体下肢区域（特别是脚部）
2. 对关键区域进行局部自适应去畸变
3. 保持其他区域的原始鱼眼特性
4. 配合动态检测框扩展

与全图去畸变（已验证为负结果）的区别：
- 仅对人体下肢/脚部区域去畸变，避免全图插值损失
- 基于深度先验动态确定去畸变区域
- 保持背景区域的鱼眼投影，减少计算开销
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo-capture-dir", required=True, type=Path,
                        help="双目采集目录")
    parser.add_argument("--calibration", required=True, type=Path,
                        help="双目标定JSON文件")
    parser.add_argument("--da3-root", required=True, type=Path,
                        help="DA3深度图根目录")
    parser.add_argument("--yolo-detection", required=True, type=Path,
                        help="YOLO初步检测结果JSON")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="输出目录")
    parser.add_argument("--pair-ids", type=int, nargs="+", required=True,
                        help="要处理的pair ID列表")
    parser.add_argument("--undistort-intensity", type=float, default=0.8,
                        help="去畸变强度，0=无去畸变，1=完全去畸变（默认0.8）")
    parser.add_argument("--depth-threshold-percentile", type=float, default=60,
                        help="深度阈值百分位，用于分离前景人体（默认60）")
    return parser.parse_args()


def load_calibration(path: Path) -> dict[str, Any]:
    """加载标定参数"""
    with path.open("r", encoding="utf-8") as f:
        calib = json.load(f)
    return {
        "left_K": np.asarray(calib["left"]["K"], dtype=np.float64),
        "left_D": np.asarray(calib["left"]["D"], dtype=np.float64).reshape(-1, 1),
        "right_K": np.asarray(calib["right"]["K"], dtype=np.float64),
        "right_D": np.asarray(calib["right"]["D"], dtype=np.float64).reshape(-1, 1),
        "image_size": tuple(calib["left"]["image_size"]),
    }


def create_undistort_maps(K: np.ndarray, D: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """创建鱼眼去畸变映射"""
    width, height = size
    identity = np.eye(3, dtype=np.float64)
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        K, D, identity, K, (width, height), cv2.CV_32FC1
    )
    return map_x, map_y


def compute_depth_mask(
    depth: np.ndarray,
    bbox: list[float],
    threshold_percentile: float,
) -> np.ndarray:
    """基于DA3深度图计算人体下肢区域mask

    策略：
    1. 在检测框及其下方区域采样深度
    2. 使用百分位阈值分离前景（人体）和背景
    3. 生成平滑的权重mask
    """
    height, width = depth.shape
    x1, y1, x2, y2 = bbox

    # 扩展检测框到下方（脚部可能被截断的区域）
    x1_int = max(0, int(x1))
    y1_int = max(0, int(y1))
    x2_int = min(width, int(x2))
    y2_int = min(height, int(y2))

    # 向下扩展50%高度以覆盖可能的脚部区域
    box_height = y2_int - y1_int
    y2_extended = min(height, y2_int + int(box_height * 0.5))

    # 在扩展区域内采样有效深度值
    roi_depth = depth[y1_int:y2_extended, x1_int:x2_int]
    valid_depth = roi_depth[np.isfinite(roi_depth)]

    if len(valid_depth) < 10:
        # 深度数据不足，返回基于几何的简单mask
        return create_geometric_mask(depth.shape, bbox)

    # 计算深度阈值（前景物体通常深度值较小）
    depth_threshold = np.percentile(valid_depth, threshold_percentile)

    # 创建二值mask：深度小于阈值的区域
    binary_mask = np.zeros(depth.shape, dtype=np.float32)
    foreground = (depth < depth_threshold) & np.isfinite(depth)
    binary_mask[foreground] = 1.0

    # 形态学操作：闭运算填充空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # 高斯模糊创建平滑过渡
    smooth_mask = cv2.GaussianBlur(binary_mask, (51, 51), 15)

    # 限制mask范围在扩展检测框内及下方
    region_mask = np.zeros(depth.shape, dtype=np.float32)
    region_mask[y1_int:y2_extended, x1_int:x2_int] = 1.0

    return smooth_mask * region_mask


def create_geometric_mask(shape: tuple[int, int], bbox: list[float]) -> np.ndarray:
    """当DA3深度不可用时，创建基于几何的简单mask"""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.float32)

    x1, y1, x2, y2 = bbox
    x1_int = max(0, int(x1))
    y1_int = max(0, int(y1))
    x2_int = min(width, int(x2))
    y2_int = min(height, int(y2))

    # 重点关注检测框下半部分和下方区域（下肢/脚部）
    box_height = y2_int - y1_int
    lower_y_start = y1_int + int(box_height * 0.4)  # 从40%高度开始
    lower_y_end = min(height, y2_int + int(box_height * 0.5))  # 向下扩展50%

    # 创建渐变mask：上方较弱，下方较强
    for y in range(lower_y_start, lower_y_end):
        weight = (y - lower_y_start) / (lower_y_end - lower_y_start)
        mask[y, x1_int:x2_int] = weight

    # 平滑过渡
    return cv2.GaussianBlur(mask, (51, 51), 15)


def apply_selective_undistort(
    image: np.ndarray,
    mask: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    intensity: float,
) -> np.ndarray:
    """应用选择性去畸变

    Args:
        image: 原始鱼眼图像
        mask: 去畸变权重mask（0-1范围）
        map_x, map_y: 去畸变映射
        intensity: 去畸变强度（0-1）

    Returns:
        选择性去畸变后的图像
    """
    # 完全去畸变的图像
    undistorted = cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    # 根据mask和intensity混合原图和去畸变图像
    effective_mask = (mask * intensity)[:, :, np.newaxis]
    result = (
        image.astype(np.float32) * (1.0 - effective_mask) +
        undistorted.astype(np.float32) * effective_mask
    ).astype(np.uint8)

    return result


def expand_bbox_with_depth(
    bbox: list[float],
    depth: np.ndarray,
    image_size: tuple[int, int],
) -> list[float]:
    """基于深度信息动态扩展检测框

    策略：如果检测框下方存在与人体深度相近的区域，扩展框以包含它们
    """
    width, height = image_size
    x1, y1, x2, y2 = bbox

    # 基础扩展（水平15%，顶部10%）
    bw, bh = x2 - x1, y2 - y1
    new_x1 = max(0.0, x1 - 0.15 * bw)
    new_y1 = max(0.0, y1 - 0.10 * bh)
    new_x2 = min(float(width - 1), x2 + 0.15 * bw)

    # 动态底部扩展：检查深度连续性
    x1_int = max(0, int(x1))
    x2_int = min(depth.shape[1], int(x2))
    y2_int = min(depth.shape[0], int(y2))

    # 采样框内中下部的深度作为人体参考
    ref_y_start = max(0, int(y1 + bh * 0.6))
    ref_depth_roi = depth[ref_y_start:y2_int, x1_int:x2_int]
    valid_ref = ref_depth_roi[np.isfinite(ref_depth_roi)]

    if len(valid_ref) > 10:
        ref_depth_median = np.median(valid_ref)
        ref_depth_std = np.std(valid_ref)

        # 向下搜索深度相近的区域
        max_extension = min(int(bh * 0.5), depth.shape[0] - y2_int)
        bottom_extension = 0

        for dy in range(0, max_extension, 5):
            search_y = y2_int + dy
            if search_y >= depth.shape[0]:
                break

            search_roi = depth[search_y, x1_int:x2_int]
            valid_search = search_roi[np.isfinite(search_roi)]

            if len(valid_search) > 5:
                search_median = np.median(valid_search)
                # 如果深度在2个标准差内，认为是连续的人体区域
                if abs(search_median - ref_depth_median) < 2.0 * ref_depth_std:
                    bottom_extension = dy
                else:
                    break

        # 至少扩展25%，最多扩展到深度连续的位置
        bottom_extension = max(int(bh * 0.25), bottom_extension)
        new_y2 = min(float(height - 1), y2 + bottom_extension)
    else:
        # 深度数据不足，使用固定25%扩展
        new_y2 = min(float(height - 1), y2 + 0.25 * bh)

    return [new_x1, new_y1, new_x2, new_y2]


def visualize_process(
    original: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    undistorted: np.ndarray,
    bbox_original: list[float],
    bbox_expanded: list[float],
) -> np.ndarray:
    """创建处理过程可视化"""
    h, w = original.shape[:2]

    # 归一化深度图用于显示
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

    # 绘制检测框
    img_with_boxes = original.copy()
    for bbox, color, label in [
        (bbox_original, (0, 0, 255), "Original"),
        (bbox_expanded, (0, 255, 0), "Expanded")
    ]:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img_with_boxes, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 创建2x2拼图
    top = np.hstack([
        cv2.resize(img_with_boxes, (w // 2, h // 2)),
        cv2.resize(depth_vis, (w // 2, h // 2))
    ])
    bottom = np.hstack([
        cv2.resize(mask_vis, (w // 2, h // 2)),
        cv2.resize(undistorted, (w // 2, h // 2))
    ])

    result = np.vstack([top, bottom])

    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, "Original + Boxes", (10, 30), font, 1.0, (255, 255, 255), 2)
    cv2.putText(result, "DA3 Depth", (w // 2 + 10, 30), font, 1.0, (255, 255, 255), 2)
    cv2.putText(result, "Undistort Mask", (10, h // 2 + 30), font, 1.0, (255, 255, 255), 2)
    cv2.putText(result, "Result", (w // 2 + 10, h // 2 + 30), font, 1.0, (255, 255, 255), 2)

    return result


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在: {output_dir}")

    output_dir.mkdir(parents=True)

    # 加载标定参数
    calib = load_calibration(args.calibration)

    # 创建去畸变映射
    left_map_x, left_map_y = create_undistort_maps(
        calib["left_K"], calib["left_D"], calib["image_size"]
    )
    right_map_x, right_map_y = create_undistort_maps(
        calib["right_K"], calib["right_D"], calib["image_size"]
    )

    # 加载YOLO检测结果
    with args.yolo_detection.open("r", encoding="utf-8") as f:
        yolo_data = json.load(f)

    results = []

    for pair_id in args.pair_ids:
        print(f"\n处理 pair_id={pair_id}")

        # 加载DA3深度（这里需要根据实际DA3输出格式调整）
        # 假设DA3输出为 .npz 格式
        da3_file = args.da3_root / f"pair_{pair_id:03d}" / "results.npz"
        if not da3_file.exists():
            print(f"  警告: DA3深度文件不存在: {da3_file}")
            continue

        da3_data = np.load(da3_file)
        left_depth = da3_data["depth"][0]  # 左图深度
        right_depth = da3_data["depth"][1]  # 右图深度

        # 获取对应的YOLO检测框
        # 这里需要根据实际YOLO输出格式调整
        pair_detection = next((d for d in yolo_data["pairs"] if d["pair_id"] == pair_id), None)
        if pair_detection is None:
            print(f"  警告: 未找到pair_id={pair_id}的检测结果")
            continue

        # 加载原始图像
        stereo_capture = args.stereo_capture_dir
        # 这里需要根据实际采集格式调整路径
        left_img_path = stereo_capture / "left" / f"frame_{pair_id:06d}.png"
        right_img_path = stereo_capture / "right" / f"frame_{pair_id:06d}.png"

        left_img = cv2.imread(str(left_img_path))
        right_img = cv2.imread(str(right_img_path))

        if left_img is None or right_img is None:
            print(f"  警告: 无法加载图像")
            continue

        pair_result = {"pair_id": pair_id, "views": {}}

        for side, img, depth, map_x, map_y, det in [
            ("left", left_img, left_depth, left_map_x, left_map_y, pair_detection.get("left")),
            ("right", right_img, right_depth, right_map_x, right_map_y, pair_detection.get("right"))
        ]:
            if det is None or not det.get("bbox"):
                print(f"  {side}: 无检测框")
                continue

            bbox_original = det["bbox"]

            # 步骤1: 基于深度动态扩展检测框
            bbox_expanded = expand_bbox_with_depth(
                bbox_original, depth, calib["image_size"]
            )

            # 步骤2: 计算深度引导的去畸变mask
            mask = compute_depth_mask(
                depth, bbox_expanded, args.depth_threshold_percentile
            )

            # 步骤3: 应用选择性去畸变
            undistorted = apply_selective_undistort(
                img, mask, map_x, map_y, args.undistort_intensity
            )

            # 保存结果
            output_prefix = output_dir / f"pair_{pair_id:03d}_{side}"
            cv2.imwrite(str(output_prefix) + "_undistorted.png", undistorted)

            # 创建可视化
            vis = visualize_process(
                img, depth, mask, undistorted,
                bbox_original, bbox_expanded
            )
            cv2.imwrite(str(output_prefix) + "_process.jpg", vis)

            pair_result["views"][side] = {
                "bbox_original": bbox_original,
                "bbox_expanded": bbox_expanded,
                "expansion_ratio": {
                    "width": (bbox_expanded[2] - bbox_expanded[0]) / (bbox_original[2] - bbox_original[0]),
                    "height": (bbox_expanded[3] - bbox_expanded[1]) / (bbox_original[3] - bbox_original[1])
                },
                "mask_coverage": float(mask.sum() / mask.size),
            }

            print(f"  {side}: 完成 (框扩展比例 {pair_result['views'][side]['expansion_ratio']['height']:.2f})")

        results.append(pair_result)

    # 保存实验元数据
    metadata = {
        "classification": "engineering_validation",
        "experiment_id": "DA3_guided_selective_undistort",
        "description": "基于DA3深度图引导的选择性去畸变，重点优化下肢/脚部区域",
        "method": {
            "step1": "基于DA3深度动态扩展检测框以包含完整脚部",
            "step2": "使用深度先验识别人体下肢区域",
            "step3": "对关键区域应用选择性去畸变，保持背景原始投影"
        },
        "parameters": {
            "undistort_intensity": args.undistort_intensity,
            "depth_threshold_percentile": args.depth_threshold_percentile,
        },
        "comparison_to_previous": {
            "vs_B0": "B0在原始鱼眼图上直接检测，本方法对下肢区域去畸变",
            "vs_full_undistort": "全图去畸变为负结果，本方法仅对关键区域去畸变"
        },
        "results": results,
    }

    with (output_dir / "experiment_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 创建EXPERIMENT.md
    (output_dir / "EXPERIMENT.md").write_text(
        f"""# DA3深度引导的选择性去畸变实验

## 分类
`engineering_validation`

## 动机
- B0基线：在原始鱼眼图上直接检测，中近距离脚部易被框截断
- 全图去畸变：已验证为负结果（精度损失大于畸变收益）
- 本方法：仅对下肢/脚部关键区域进行去畸变，保持背景原始投影

## 方法
1. **动态框扩展**: 基于DA3深度连续性自适应扩展检测框底部
2. **区域识别**: 利用深度先验识别人体下肢mask
3. **选择性去畸变**: 根据mask权重混合原图和去畸变图像

## 参数
- 去畸变强度: {args.undistort_intensity}
- 深度阈值百分位: {args.depth_threshold_percentile}

## 处理的pair数量
{len(results)}

## 下一步
1. 在去畸变后的图像上运行PMPose/Sapiens2
2. 对比B0基线的脚部关键点检出率
3. 检查三角化后的3D下肢链连续性

## 结论边界
- DA3深度为相对深度，不是米制真值
- 去畸变改善几何但引入插值误差
- 需要独立人工标注验证脚部关键点质量
""",
        encoding="utf-8"
    )

    print(f"\n实验完成，输出目录: {output_dir}")
    print(f"处理了 {len(results)} 个pairs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
