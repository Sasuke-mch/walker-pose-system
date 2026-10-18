#!/usr/bin/env python3
"""Fit a coarse complete walking-frame model from catalog and offline evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from pose_app.calibration import StereoCalibration  # noqa: E402
from pose_app.parametric_walker_model import WalkerDimensions, build_coarse_frame, transform_nodes  # noqa: E402
from pose_app.rotation import raw_to_model_point  # noqa: E402


SEMANTIC_ANCHORS = {
    "kp_01": "rear_right_foot",
    "kp_02": "rear_left_foot",
    "kp_03": "front_top_center",
    "kp_04": "rear_left_mid",
    "kp_05": "rear_right_mid",
    "kp_06": "right_body_handle_midpoint",
    "kp_07": "rear_right_lower_leg",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def estimate_dimensions(
    anchors_ground: dict[str, np.ndarray],
    camera_height_mm: float,
    camera_center_above_front_rail_mm: float,
) -> WalkerDimensions:
    height = float(np.median([anchors_ground["kp_03"][2], anchors_ground["kp_06"][2]]))
    height = float(np.clip(height, 750.0, 930.0))
    middle = float(np.median([anchors_ground["kp_04"][2], anchors_ground["kp_05"][2]]))
    front_rail = float(camera_height_mm - camera_center_above_front_rail_mm)
    return WalkerDimensions(
        outer_width_mm=450.0,
        depth_mm=300.0,
        height_mm=height,
        middle_rail_height_mm=middle,
        front_rail_height_mm=front_rail,
    )


def anchor_points_walker(model: dict, anchors_ground: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    nodes = {name: np.asarray(value, dtype=np.float64) for name, value in model["nodes_walker_mm"].items()}
    left_top = nodes["front_left_top"]
    rear_left_top = nodes["rear_left_top"]
    return {
        "kp_01": nodes["rear_right_foot"],
        "kp_02": nodes["rear_left_foot"],
        "kp_03": nodes["front_top_center"],
        "kp_04": nodes["rear_left_mid"],
        "kp_05": nodes["rear_right_mid"],
        "kp_06": 0.5 * (left_top + rear_left_top),
        "kp_07": np.array([nodes["rear_right_foot"][0], nodes["rear_right_foot"][1], anchors_ground["kp_07"][2]]),
    }


def fit_ground_yaw_translation(
    source_walker: dict[str, np.ndarray], target_ground: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict]:
    labels = [label for label in SEMANTIC_ANCHORS if label in source_walker and label in target_ground]
    source = np.asarray([source_walker[label][:2] for label in labels])
    target = np.asarray([target_ground[label][:2] for label in labels])
    source_center, target_center = np.mean(source, axis=0), np.mean(target, axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation_2d = vt.T @ u.T
    if np.linalg.det(rotation_2d) < 0:
        vt[-1] *= -1
        rotation_2d = vt.T @ u.T
    translation_2d = target_center - rotation_2d @ source_center
    rotation = np.eye(3)
    rotation[:2, :2] = rotation_2d
    translation = np.array([translation_2d[0], translation_2d[1], 0.0])
    predicted = (rotation @ np.asarray([source_walker[label] for label in labels]).T).T + translation
    residuals = {
        label: {
            "semantic_assignment": SEMANTIC_ANCHORS[label],
            "observed_ground_mm": target_ground[label].tolist(),
            "model_ground_mm": predicted[index].tolist(),
            "xyz_residual_mm": float(np.linalg.norm(predicted[index] - target_ground[label])),
            "horizontal_residual_mm": float(np.linalg.norm(predicted[index, :2] - target_ground[label][:2])),
        }
        for index, label in enumerate(labels)
    }
    horizontal = np.asarray([item["horizontal_residual_mm"] for item in residuals.values()])
    xyz = np.asarray([item["xyz_residual_mm"] for item in residuals.values()])
    audit = {
        "semantic_assignment_source": "manual visual interpretation of pair_0000 plus product topology",
        "yaw_deg": float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
        "horizontal_residual_median_mm": float(np.median(horizontal)),
        "horizontal_residual_max_mm": float(np.max(horizontal)),
        "xyz_residual_median_mm": float(np.median(xyz)),
        "xyz_residual_max_mm": float(np.max(xyz)),
        "per_anchor": residuals,
    }
    return rotation, translation, audit


def project_overlay(
    image: np.ndarray,
    nodes_camera: dict[str, list[float]],
    edges: list[list[str]],
    calibration: StereoCalibration,
    side: str,
) -> np.ndarray:
    canvas = image.copy()
    names = list(nodes_camera)
    xyz = np.asarray([nodes_camera[name] for name in names], dtype=np.float64)
    raw = calibration.project_left(xyz) if side == "left" else calibration.project_right(xyz)
    rotation = "ccw90" if side == "left" else "cw90"
    upright = {
        name: np.asarray(raw_to_model_point(point[0], point[1], 1920, 1080, rotation))
        for name, point in zip(names, raw)
    }
    for start_name, end_name in edges:
        start, end = upright[start_name], upright[end_name]
        cv2.line(canvas, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)), (255, 190, 0), 6, cv2.LINE_AA)
    for name in ("front_right_foot", "front_left_foot", "rear_right_foot", "rear_left_foot"):
        cv2.circle(canvas, tuple(np.rint(upright[name]).astype(int)), 9, (0, 180, 255), -1, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo-anchors", type=Path, required=True)
    parser.add_argument("--ground-reference", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--left-image", type=Path, required=True)
    parser.add_argument("--right-image", type=Path, required=True)
    parser.add_argument("--product-reference", type=Path, required=True)
    parser.add_argument(
        "--camera-center-above-front-rail-mm",
        type=float,
        default=30.0,
        help="Coarse vertical offset from the front mounting rail to the left camera optical center.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    anchor_data = read_json(args.stereo_anchors)
    ground = read_json(args.ground_reference)
    calibration = StereoCalibration.load(args.calibration)
    rotation_ground_camera = np.asarray(ground["rotation_ground_from_left"], dtype=np.float64)
    translation_ground_camera = np.asarray(ground["translation_ground_from_left_mm"], dtype=np.float64)
    camera_height_mm = float(translation_ground_camera[2])
    if not 10.0 <= args.camera_center_above_front_rail_mm <= 100.0:
        raise ValueError("camera-to-front-rail vertical offset must be between 10 and 100 mm")
    anchors_ground = {
        item["label"]: rotation_ground_camera @ np.asarray(item["xyz_left_camera_mm"], dtype=np.float64) + translation_ground_camera
        for item in anchor_data["anchors"]
        if item["status"] == "candidate" and item["label"] in SEMANTIC_ANCHORS
    }
    dimensions = estimate_dimensions(
        anchors_ground,
        camera_height_mm,
        args.camera_center_above_front_rail_mm,
    )
    model = build_coarse_frame(dimensions)
    source_anchors = anchor_points_walker(model, anchors_ground)
    rotation_ground_walker, translation_ground_walker, fit_audit = fit_ground_yaw_translation(source_anchors, anchors_ground)
    nodes_ground = transform_nodes(model["nodes_walker_mm"], rotation_ground_walker, translation_ground_walker)
    rotation_camera_ground = rotation_ground_camera.T
    rotation_camera_walker = rotation_camera_ground @ rotation_ground_walker
    translation_camera_walker = rotation_camera_ground @ (translation_ground_walker - translation_ground_camera)
    nodes_camera = transform_nodes(model["nodes_walker_mm"], rotation_camera_walker, translation_camera_walker)
    right_camera_center_left = -calibration.R.T @ calibration.T
    foot_heights = [nodes_ground[name][2] for name in model["foot_nodes"]]
    result = {
        "schema_version": "coarse_parametric_walker_model_v2",
        "status": "offline_coarse_model_candidate",
        "product_reference": {
            "image": str(args.product_reference.resolve()),
            "brand": "可孚",
            "model": "611",
            "catalog_outer_width_mm": 450.0,
            "catalog_depth_mm": 300.0,
            "catalog_inner_width_mm": 320.0,
            "catalog_height_range_mm": [750.0, 930.0],
            "catalog_lower_adjustment_range_mm": [430.0, 610.0],
        },
        "dimensions_mm": {
            "outer_width": dimensions.outer_width_mm,
            "depth": dimensions.depth_mm,
            "height": dimensions.height_mm,
            "middle_rail_height": dimensions.middle_rail_height_mm,
            "front_rail_height": dimensions.front_rail_height_mm,
            "camera_height": camera_height_mm,
            "camera_center_above_front_rail": args.camera_center_above_front_rail_mm,
            "height_source": "median of geometry-compatible upper crossbar and handle anchors, clamped to catalog range",
            "front_rail_height_source": "measured initial left-camera height minus configurable optical-center mounting offset",
        },
        "coordinate_frame": "walker_rigid_frame; x=body-left direction in ground display, y=open/user side, z=up",
        "nodes_walker_mm": model["nodes_walker_mm"],
        "edges": model["edges"],
        "handle_segments": model["handle_segments"],
        "foot_nodes": model["foot_nodes"],
        "rotation_initial_ground_from_walker": rotation_ground_walker.tolist(),
        "translation_initial_ground_from_walker_mm": translation_ground_walker.tolist(),
        "rotation_left_camera_from_walker": rotation_camera_walker.tolist(),
        "translation_left_camera_from_walker_mm": translation_camera_walker.tolist(),
        "nodes_initial_ground_mm": nodes_ground,
        "nodes_left_camera_mm": nodes_camera,
        "camera_centers_left_camera_mm": {
            "left": [0.0, 0.0, 0.0],
            "right": right_camera_center_left.tolist(),
        },
        "camera_display_interpretation": (
            "optical centers from the stereo calibration; markers are display-only and do not alter pose estimation"
        ),
        "fit_audit": fit_audit,
        "initial_foot_signed_height_mm": foot_heights,
        "ground_contact_interpretation": "all four catalog feet are constrained to the measured initial z=0 plane by model construction; this is not four independently observed contacts",
        "hand_contact_interpretation": "distance to catalog-derived handle segments is a visual proximity candidate, not tactile or load-bearing truth",
        "topology_boundary": "complete coarse topology comes from the product reference; image evidence positions and checks the model but does not observe every member",
    }
    model_path = output / "coarse_walker_model.json"
    model_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for side, image_path in (("left", args.left_image), ("right", args.right_image)):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        overlay = project_overlay(image, nodes_camera, model["edges"], calibration, side)
        cv2.imwrite(str(output / f"pair_0000_{side}_coarse_model_projection.jpg"), overlay)
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(model_path), "dimensions_mm": result["dimensions_mm"], "fit_audit": fit_audit}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
