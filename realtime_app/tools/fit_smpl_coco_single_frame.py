#!/usr/bin/env python3
"""Fit male SMPL to one frozen stereo PMPose COCO-17 frame.

Only per-view 2-D keypoints, scores, fisheye calibration, the official male
SMPL asset and the fixed Pose2Mesh COCO regressor are read.  Stored 3-D points,
old fit parameters, temporal state and contact data are intentionally ignored.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_app.fisheye_camera import (  # noqa: E402
    fisheye_project_torch,
    load_stereo_fisheye,
)
from pose_app.smpl_coco_observation import (  # noqa: E402
    COCO17_NAMES,
    load_coco17_regressor,
    load_smpl_male,
    regress_coco17_torch,
)


RIGHT_KNEE_INDEX = 14
TORSO_INDICES = (5, 6, 11, 12)
SMPL_BODY_JOINT_NAMES = (
    "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
)
# Rotations after the last observed point on a limb cannot be identified from
# COCO-17. Freeze ankles/feet and wrists/hands instead of letting the optimizer
# move surface proxy points through unobservable terminal rotations. Keep all
# three axis-angle components for every remaining joint: a fixed component is
# not a fixed anatomical hinge axis in the posed SMPL kinematic chain.
FROZEN_POSE_JOINTS = (6, 7, 9, 10, 19, 20, 21, 22)
FULL_ACTIVE_POSE_JOINTS = tuple(
    index for index in range(len(SMPL_BODY_JOINT_NAMES))
    if index not in FROZEN_POSE_JOINTS
)
ACTIVE_POSE_COMPONENTS = tuple(
    joint * 3 + axis for joint in FULL_ACTIVE_POSE_JOINTS for axis in range(3)
)
COCO_SKELETON = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
)


def read_pair(path: Path, pair_id: int) -> dict:
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("pair_id", -1)) == pair_id:
                if row.get("max_matches") != 1:
                    raise ValueError(f"pair {pair_id} does not preserve max_matches=1")
                if len(row["left"]["persons"]) != 1 or len(row["right"]["persons"]) != 1:
                    raise ValueError(f"pair {pair_id} does not contain exactly one person per view")
                return row
    raise KeyError(f"pair {pair_id} not found in {path}")


def pixel_ray(pixel: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    xd = (float(pixel[0]) - K[0, 2]) / K[0, 0]
    yd = (float(pixel[1]) - K[1, 2]) / K[1, 1]
    radius_distorted = math.hypot(xd, yd)
    if radius_distorted < 1e-12:
        return np.asarray([0.0, 0.0, 1.0])
    theta = radius_distorted
    k1, k2, k3, k4 = np.asarray(D).reshape(-1)[:4]
    for _ in range(10):
        theta2 = theta * theta
        value = theta * (
            1.0 + k1 * theta2 + k2 * theta2**2 + k3 * theta2**3 + k4 * theta2**4
        ) - radius_distorted
        derivative = (
            1.0 + 3.0 * k1 * theta2 + 5.0 * k2 * theta2**2
            + 7.0 * k3 * theta2**3 + 9.0 * k4 * theta2**4
        )
        theta -= value / max(float(derivative), 1e-8)
    radius = math.tan(theta)
    ray = np.asarray([xd / radius_distorted * radius, yd / radius_distorted * radius, 1.0])
    return ray / np.linalg.norm(ray)


def initial_root_from_left_2d(keypoints: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    points = []
    for index in (5, 6, 11, 12):
        ray = pixel_ray(keypoints[index, :2], K, D)
        points.append(ray / max(ray[2], 1e-6))
    left_shoulder, right_shoulder, left_hip, right_hip = points
    x_axis = left_shoulder - right_shoulder
    y_axis = 0.5 * (left_shoulder + right_shoulder) - 0.5 * (left_hip + right_hip)
    x_axis /= max(np.linalg.norm(x_axis), 1e-8)
    y_axis -= x_axis * np.dot(x_axis, y_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-8)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(np.linalg.norm(z_axis), 1e-8)
    matrix = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(matrix) < 0:
        matrix[:, 2] *= -1.0
    return Rotation.from_matrix(matrix).as_rotvec().astype(np.float32)


def valid_2d(keypoints: np.ndarray, width: int, height: int, minimum_score: float) -> np.ndarray:
    valid = (
        np.isfinite(keypoints).all(axis=1)
        & (keypoints[:, 2] >= minimum_score)
        & (keypoints[:, 0] >= 0.0)
        & (keypoints[:, 0] < width)
        & (keypoints[:, 1] >= 0.0)
        & (keypoints[:, 1] < height)
    )
    valid[RIGHT_KNEE_INDEX] = False
    return valid


def huber_vector(torch: "object", residual: "object", delta: float) -> "object":
    norm = torch.linalg.vector_norm(residual, dim=-1)
    return torch.where(norm <= delta, 0.5 * norm.square(), delta * (norm - 0.5 * delta))


def metrics(
    projection_left: np.ndarray,
    projection_right: np.ndarray,
    observed_left: np.ndarray,
    observed_right: np.ndarray,
    valid_left: np.ndarray,
    valid_right: np.ndarray,
    depth_left: np.ndarray,
    depth_right: np.ndarray,
) -> dict:
    error_left = np.linalg.norm(projection_left - observed_left[:, :2], axis=1)
    error_right = np.linalg.norm(projection_right - observed_right[:, :2], axis=1)
    used = np.concatenate([error_left[valid_left], error_right[valid_right]])
    jointwise = {}
    for index, name in enumerate(COCO17_NAMES):
        jointwise[name] = {
            "left_px": float(error_left[index]),
            "right_px": float(error_right[index]),
            "left_supervised": bool(valid_left[index]),
            "right_supervised": bool(valid_right[index]),
        }
    return {
        "median_px": float(np.median(used)),
        "p95_px": float(np.percentile(used, 95)),
        "left_median_px": float(np.median(error_left[valid_left])),
        "left_p95_px": float(np.percentile(error_left[valid_left], 95)),
        "right_median_px": float(np.median(error_right[valid_right])),
        "right_p95_px": float(np.percentile(error_right[valid_right], 95)),
        "supervised_left": int(valid_left.sum()),
        "supervised_right": int(valid_right.sum()),
        "negative_depth_points_left": int((depth_left <= 0.0).sum()),
        "negative_depth_points_right": int((depth_right <= 0.0).sum()),
        "jointwise": jointwise,
    }


def draw_overlay(
    input_path: Path,
    recover_rotation: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(input_path).convert("RGB")
    if recover_rotation == "cw":
        image = image.transpose(Image.Transpose.ROTATE_270)
    elif recover_rotation == "ccw":
        image = image.transpose(Image.Transpose.ROTATE_90)
    if image.size != (1920, 1080):
        raise ValueError(f"recovered raw image has unexpected size {image.size}: {input_path}")
    draw = ImageDraw.Draw(image)
    for a, b in COCO_SKELETON:
        draw.line((*observed[a, :2], *observed[b, :2]), fill=(255, 210, 0), width=3)
        draw.line((*predicted[a], *predicted[b]), fill=(0, 255, 255), width=3)
    for index, (obs, pred) in enumerate(zip(observed, predicted)):
        draw.line((*obs[:2], *pred), fill=(255, 255, 255), width=2)
        radius = 6
        draw.ellipse((obs[0]-radius, obs[1]-radius, obs[0]+radius, obs[1]+radius), fill=(255, 210, 0))
        draw.ellipse((pred[0]-radius, pred[1]-radius, pred[0]+radius, pred[1]+radius), fill=(0, 255, 255))
        draw.text((pred[0]+7, pred[1]+4), str(index), fill=(0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255))
    image.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stereo-jsonl", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--regressor", type=Path, required=True)
    parser.add_argument("--left-upright-image", type=Path, required=True)
    parser.add_argument("--right-upright-image", type=Path, required=True)
    parser.add_argument("--pair-id", type=int, default=75)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-score", type=float, default=0.20)
    parser.add_argument("--camera-iterations", type=int, default=250)
    parser.add_argument("--body-iterations", type=int, default=800)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--skip-overlays",
        action="store_true",
        help="Do not render raw-image overlays (used by the fixed window audit).",
    )
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    row = read_pair(args.stereo_jsonl.resolve(), args.pair_id)
    left = np.asarray(row["left"]["persons"][0]["keypoints"], dtype=np.float64)
    right = np.asarray(row["right"]["persons"][0]["keypoints"], dtype=np.float64)
    if left.shape != (17, 3) or right.shape != (17, 3):
        raise ValueError(f"expected PMPose COCO-17, got {left.shape} and {right.shape}")
    cal = load_stereo_fisheye(args.calibration_dir)
    valid_left = valid_2d(left, cal.image_width, cal.image_height, args.minimum_score)
    valid_right = valid_2d(right, cal.image_width, cal.image_height, args.minimum_score)

    import torch

    model = load_smpl_male(args.model_path, batch_size=1, device=args.device)
    regressor = load_coco17_regressor(args.regressor)
    betas = torch.zeros(1, 10, dtype=torch.float32, device=args.device)
    active_pose = torch.nn.Parameter(
        torch.zeros(1, len(ACTIVE_POSE_COMPONENTS), dtype=torch.float32, device=args.device)
    )
    active_pose_indices = torch.tensor(
        ACTIVE_POSE_COMPONENTS,
        dtype=torch.long,
        device=args.device,
    )
    root_seed = initial_root_from_left_2d(left, cal.K0, cal.D0)
    root = torch.nn.Parameter(torch.tensor(root_seed[None], device=args.device))

    pelvis_pixel = 0.5 * (left[11, :2] + left[12, :2])
    pelvis_ray = pixel_ray(pelvis_pixel, cal.K0, cal.D0)
    observation_left = torch.tensor(left[:, :2], dtype=torch.float32, device=args.device)
    observation_right = torch.tensor(right[:, :2], dtype=torch.float32, device=args.device)
    weight_left = torch.tensor(valid_left * np.square(left[:, 2]), dtype=torch.float32, device=args.device)
    weight_right = torch.tensor(valid_right * np.square(right[:, 2]), dtype=torch.float32, device=args.device)
    rotation_right = torch.tensor(cal.R_cam0_to_cam1, dtype=torch.float32, device=args.device)
    translation_right = torch.tensor(cal.T_cam0_to_cam1_mm / 1000.0, dtype=torch.float32, device=args.device)

    def expand_pose(current_active_pose: "object") -> "object":
        full_pose = torch.zeros(1, 69, dtype=current_active_pose.dtype, device=current_active_pose.device)
        return full_pose.index_copy(1, active_pose_indices, current_active_pose)

    def forward(current_root: "object", current_active_pose: "object", current_translation: "object") -> tuple:
        current_pose = expand_pose(current_active_pose)
        result = model(
            betas=betas,
            body_pose=current_pose,
            global_orient=current_root,
            transl=current_translation,
            return_verts=True,
        )
        coco_left = regress_coco17_torch(result.vertices, regressor)
        coco_right = torch.einsum("ij,bkj->bki", rotation_right, coco_left) + translation_right
        projection_left = fisheye_project_torch(coco_left, cal.K0, cal.D0)
        projection_right = fisheye_project_torch(coco_right, cal.K1, cal.D1)
        return result, coco_left, coco_right, projection_left, projection_right

    def reprojection_loss(proj_l: "object", proj_r: "object", mask: tuple[int, ...] | None, delta: float) -> "object":
        if mask is None:
            indices = torch.arange(17, device=args.device)
        else:
            indices = torch.tensor(mask, device=args.device)
        wl = weight_left[indices]
        wr = weight_right[indices]
        left_loss = (huber_vector(torch, proj_l[0, indices] - observation_left[indices], delta) * wl).sum()
        right_loss = (huber_vector(torch, proj_r[0, indices] - observation_right[indices], delta) * wr).sum()
        return (left_loss + right_loss) / (wl.sum() + wr.sum()).clamp_min(1.0)

    candidates = []
    zero_pose = torch.zeros_like(active_pose)
    for depth in (0.45, 0.70, 1.00, 1.50):
        candidate_root = torch.nn.Parameter(torch.tensor(root_seed[None], device=args.device))
        point = pelvis_ray * (depth / max(float(pelvis_ray[2]), 1e-6))
        candidate_translation = torch.nn.Parameter(torch.tensor(point[None], dtype=torch.float32, device=args.device))
        optimizer = torch.optim.Adam([candidate_root, candidate_translation], lr=0.01)
        with torch.no_grad():
            _, _, _, pl0, pr0 = forward(candidate_root, zero_pose, candidate_translation)
            initial_loss = float(reprojection_loss(pl0, pr0, TORSO_INDICES, 100.0))
        for step in range(args.camera_iterations):
            _, _, _, projection_left, projection_right = forward(
                candidate_root, zero_pose, candidate_translation
            )
            delta = 100.0 if step < args.camera_iterations // 2 else 30.0
            loss = reprojection_loss(
                projection_left, projection_right, TORSO_INDICES, delta
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                candidate_translation[:, 0:2].clamp_(-3.0, 3.0)
                candidate_translation[:, 2].clamp_(0.25, 3.0)
        with torch.no_grad():
            _, _, _, pl, pr = forward(candidate_root, zero_pose, candidate_translation)
            final_loss = float(reprojection_loss(pl, pr, None, 30.0))
        candidates.append({
            "depth_seed_m": depth,
            "initial_torso_loss": initial_loss,
            "final_all_joint_loss": final_loss,
            "root": candidate_root.detach().clone(),
            "translation": candidate_translation.detach().clone(),
        })

    best = min(candidates, key=lambda item: item["final_all_joint_loss"])
    with torch.no_grad():
        root.copy_(best["root"])
    translation = torch.nn.Parameter(best["translation"].clone())
    optimizer = torch.optim.Adam(
        [
            {"params": [root, active_pose], "lr": 0.008},
            {"params": [translation], "lr": 0.004},
        ]
    )
    history = []
    with torch.no_grad():
        _, coco_l, coco_r, projection_left, projection_right = forward(root, active_pose, translation)
        initial_metrics = metrics(
            projection_left[0].cpu().numpy(), projection_right[0].cpu().numpy(),
            left, right, valid_left, valid_right,
            coco_l[0, :, 2].cpu().numpy(), coco_r[0, :, 2].cpu().numpy(),
        )
    for step in range(args.body_iterations):
        result, coco_l, coco_r, projection_left, projection_right = forward(
            root, active_pose, translation
        )
        delta = 50.0 if step < args.body_iterations // 3 else 20.0
        loss_2d = reprojection_loss(projection_left, projection_right, None, delta)
        pose_prior = active_pose.square().mean()
        loss = loss_2d + 5.0 * pose_prior
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            active_pose.clamp_(-2.5, 2.5)
            translation[:, 0:2].clamp_(-3.0, 3.0)
            translation[:, 2].clamp_(0.25, 3.0)
        if step in (0, 99, 199, 399, args.body_iterations - 1):
            history.append({
                "iteration": step + 1,
                "loss": float(loss.detach()),
                "loss_2d": float(loss_2d.detach()),
                "pose_prior": float(pose_prior.detach()),
            })

    with torch.no_grad():
        result, coco_l, coco_r, projection_left, projection_right = forward(
            root, active_pose, translation
        )
    vertices = result.vertices[0].cpu().numpy()
    coco_left = coco_l[0].cpu().numpy()
    coco_right = coco_r[0].cpu().numpy()
    projected_left = projection_left[0].cpu().numpy()
    projected_right = projection_right[0].cpu().numpy()
    final_metrics = metrics(
        projected_left, projected_right, left, right, valid_left, valid_right,
        coco_left[:, 2], coco_right[:, 2],
    )
    candidate_report = [
        {key: value for key, value in item.items() if key not in {"root", "translation"}}
        for item in candidates
    ]
    full_pose = expand_pose(active_pose).detach()
    frozen_pose_joints = [SMPL_BODY_JOINT_NAMES[index] for index in FROZEN_POSE_JOINTS]
    report = {
        "schema_version": "pmpose_smpl_coco_single_frame_v1",
        "pair_id": args.pair_id,
        "input_whitelist": [
            str(args.stereo_jsonl.resolve()),
            str(args.calibration_dir.resolve()),
            str(args.model_path.resolve()),
            str(args.regressor.resolve()),
        ],
        "forbidden_inputs_confirmed_absent": [
            "persons_3d", "old SMPL-X parameters", "old SMPL-X mesh",
            "temporal prefit", "old beta", "old pose", "old translation",
            "contact", "soft-3D",
        ],
        "model": "official_male_SMPL_6890",
        "observation_model": "Pose2Mesh_COCO17_17x6890",
        "betas_fixed_zero": True,
        "right_knee_supervision_weight": 0,
        "full_active_pose_joints": [
            SMPL_BODY_JOINT_NAMES[index] for index in FULL_ACTIVE_POSE_JOINTS
        ],
        "pose_parameterization": "three_axis_except_unobservable_terminal_rotations",
        "frozen_pose_joints": frozen_pose_joints,
        "camera_initialization_candidates": candidate_report,
        "selected_depth_seed_m": best["depth_seed_m"],
        "initial": initial_metrics,
        "final": final_metrics,
        "history": history,
        "parameters": {
            "root_orient": root.detach().cpu().numpy().reshape(-1).tolist(),
            "translation_m": translation.detach().cpu().numpy().reshape(-1).tolist(),
            "body_pose": full_pose.cpu().numpy().reshape(-1).tolist(),
        },
        "conclusion_boundary": (
            "Reprojection agreement is an internal consistency check against PMPose, "
            "not true 2-D or 3-D accuracy and not physical contact validation."
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output / "result.npz",
        vertices_m=vertices.astype(np.float32),
        faces=np.asarray(model.faces, dtype=np.uint32),
        coco_left_m=coco_left.astype(np.float32),
        coco_right_m=coco_right.astype(np.float32),
        projected_left_px=projected_left.astype(np.float32),
        projected_right_px=projected_right.astype(np.float32),
        observed_left=left.astype(np.float32),
        observed_right=right.astype(np.float32),
        valid_left=valid_left,
        valid_right=valid_right,
        root_orient=root.detach().cpu().numpy().astype(np.float32),
        body_pose=full_pose.cpu().numpy().astype(np.float32),
        translation_m=translation.detach().cpu().numpy().astype(np.float32),
        betas=np.zeros((1, 10), dtype=np.float32),
    )
    if not args.skip_overlays:
        draw_overlay(
            args.left_upright_image.resolve(), "cw", left, projected_left,
            output / "overlay_left_raw.png",
        )
        draw_overlay(
            args.right_upright_image.resolve(), "ccw", right, projected_right,
            output / "overlay_right_raw.png",
        )
    print(json.dumps({"initial": initial_metrics, "final": final_metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
