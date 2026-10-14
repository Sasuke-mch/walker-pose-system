"""First-frame human 2D references and a 2D-only SMPL-H initialization refinement."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from pose_app.fisheye_camera import fisheye_project_torch
from pose_app.smplh_motion_tracker import MotionPose


LABEL_JOINTS = {"left_shoulder": 5, "right_shoulder": 6, "left_knee": 13, "right_knee": 14,
                "left_hand": 9, "leftt_hand": 9, "right_hand": 10, "left_foot": 15, "right_foot": 16}


def apply_first_frame_labels(labels: Path, input_pairs: Path, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, dict]:
    """Hand/foot semantics are user-confirmed wrist/ankle; no fuzzy name matching."""
    manual = np.zeros((2, 12), dtype=bool)
    record = {"hand_foot_semantics": "user_confirmed_wrist_ankle", "coordinates": "upright_pixels_inverse_rotated_to_raw_fisheye", "points": []}
    for v, side, rows, folder in ((0, "left", left, "left_ccw90"), (1, "right", right, "right_cw90")):
        source = labels / f"{side}_pair_0000.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        if (data["imageWidth"], data["imageHeight"]) != (1080, 1920):
            raise ValueError("labels must be on the 1080x1920 upright image")
        labeled_image = (source.parent / data["imagePath"]).resolve()
        original_image = input_pairs / folder / "pair_0000.png"
        a = cv2.imdecode(np.fromfile(labeled_image, dtype=np.uint8), cv2.IMREAD_COLOR)
        b = cv2.imdecode(np.fromfile(original_image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if a is None or b is None or a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError("label image is not the selected first-frame image")
        seen = set()
        for shape in data["shapes"]:
            label = shape["label"]
            if label not in LABEL_JOINTS or shape["shape_type"] != "point" or len(shape["points"]) != 1:
                raise ValueError(f"unsupported annotation: {label}")
            joint = LABEL_JOINTS[label]
            if joint in seen:
                raise ValueError(f"duplicate anatomical joint label: {label}")
            seen.add(joint)
            u, y = np.asarray(shape["points"][0], dtype=float)
            if not np.isfinite([u, y]).all() or not (0 <= u < 1080 and 0 <= y < 1920):
                raise ValueError(f"invalid label pixel: {label}")
            raw = np.array([1919-y, u]) if side == "left" else np.array([y, 1079-u])
            original = rows[0, joint].copy()
            rows[0, joint] = [raw[0], raw[1], 1.]
            manual[v, joint-5] = True
            record["points"].append({"camera": side, "label": label, "joint_coco17": joint,
                                      "upright_xy": [float(u), float(y)], "raw_xy": raw.tolist(),
                                      "original_auto_xy": original[:2].tolist(),
                                      "source": str(source.resolve())})
    record["manual_points"] = int(manual.sum())
    record["score_semantics"] = "1 means explicit inclusion, not a calibrated detector/manual probability"
    return manual, record


def refine_first_frame_2d(model, cal, pixels: np.ndarray, manual: np.ndarray, output: Path, steps: int = 500) -> Path:
    """Only model parameters and frame-0 pixel residuals; never triangulates."""
    import torch
    from smplx.lbs import batch_rodrigues
    output.mkdir(parents=True)
    z = torch.nn.Parameter(torch.tensor(model.initial_pose.latent[None], dtype=torch.float32, device=model.device))
    root = torch.nn.Parameter(torch.tensor(Rotation.from_matrix(model.initial_pose.root_rotation).as_rotvec()[None], dtype=torch.float32, device=model.device))
    translation = torch.nn.Parameter(torch.tensor(model.initial_pose.translation_m[None], dtype=torch.float32, device=model.device))
    beta = torch.nn.Parameter(model.betas.clone())
    optimizer = torch.optim.Adam([{"params": [z], "lr": .02}, {"params": [root, translation], "lr": .005}, {"params": [beta], "lr": .01}])
    valid = np.isfinite(pixels).all(axis=2) & (pixels[..., 2] >= .2)
    valid &= ((pixels[..., 0] >= 0) & (pixels[..., 0] < cal.image_width)
              & (pixels[..., 1] >= 0) & (pixels[..., 1] < cal.image_height))
    target = torch.tensor(np.nan_to_num(pixels[..., :2]), dtype=torch.float32, device=model.device)
    mask = torch.tensor(valid, device=model.device)
    sigma = torch.tensor(np.where(manual, 4., 15.), dtype=torch.float32, device=model.device)
    R01 = torch.tensor(cal.R_cam0_to_cam1, dtype=torch.float32, device=model.device)
    t01 = torch.tensor(cal.T_cam0_to_cam1_mm/1000, dtype=torch.float32, device=model.device)
    def decode():
        body = model.vposer.decode(z)["pose_body_matrot"].reshape(1, 21, 3, 3).clone()
        body[:, model.fixed_body_indices] = model.fixed_body[list(model.fixed_body_indices)]
        rotations = batch_rodrigues(root)
        mesh = model.model(betas=beta, transl=translation, global_orient=rotations, body_pose=body,
                           left_hand_pose=model.hand_rotations[0], right_hand_pose=model.hand_rotations[1])
        points = torch.einsum("jv,bvc->bjc", model.regressor, mesh.vertices)[0]
        right = points @ R01.T + t01
        projected = torch.stack((fisheye_project_torch(points[5:17], cal.K0, cal.D0),
                                 fisheye_project_torch(right[5:17], cal.K1, cal.D1)))
        return mesh, points, projected, body, right
    history = []
    for i in range(steps):
        optimizer.zero_grad()
        mesh, points, projected, body, other = decode()
        if bool((points[5:17, 2] <= 1e-4).any() or (other[5:17, 2] <= 1e-4).any()):
            raise ValueError("first-frame optimization left the projection domain")
        norm = torch.linalg.vector_norm(projected-target, dim=2) / sigma
        data_loss = torch.where(norm <= 3., .5*norm.square(), 3.*(norm-1.5))[mask].sum()
        loss = data_loss + .5*z.square().sum()/1.5**2 + .5*beta.square().sum()/2.**2
        if not bool(torch.isfinite(loss)):
            raise ValueError("non-finite first-frame 2D objective")
        loss.backward()
        optimizer.step()
        if i % 50 == 0 or i == steps-1:
            history.append({"step": i, "loss": float(loss.detach()), "data": float(data_loss.detach())})
    model.betas = beta.detach().clone()
    model.initial_pose = MotionPose(translation.detach().cpu().numpy()[0].astype(float),
                                    Rotation.from_rotvec(root.detach().cpu().numpy()[0]).as_matrix(),
                                    z.detach().cpu().numpy()[0].astype(float))
    final = model.evaluate(model.initial_pose, mesh=True)
    body_pose = Rotation.from_matrix(final["body_rotations"][0]).as_rotvec().reshape(1, 63)
    hands = [Rotation.from_matrix(h[0].cpu().numpy()).as_rotvec().reshape(1, 45) for h in model.hand_rotations]
    path = output / "result.npz"
    np.savez_compressed(path, pair_id=np.array([0]), vertices=final["vertices_m"], faces=model.faces,
                        predicted_coco=final["coco_m"], betas=model.betas.cpu().numpy(),
                        global_orient=root.detach().cpu().numpy(), transl=translation.detach().cpu().numpy(),
                        vposer_latent=z.detach().cpu().numpy(), body_pose=body_pose,
                        left_hand_pose=hands[0], right_hand_pose=hands[1])
    with torch.no_grad():
        _, _, projected, _, _ = decode()
        errors = torch.linalg.vector_norm(projected-target, dim=2).cpu().numpy()
    report = {"steps": steps, "objective": "frame0_2d_pixels_only_plus_VPoser_and_beta_priors", "manual_points": int(manual.sum()),
              "manual_sigma_px": 4., "auto_sigma_px": 15., "manual_reprojection_px": errors[manual].tolist(),
              "all_reprojection_px": errors.tolist(), "history": history, "shape_frozen_after_this_fit": True}
    (output/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return path
