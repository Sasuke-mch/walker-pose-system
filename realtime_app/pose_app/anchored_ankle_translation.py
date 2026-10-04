"""Camera translation from fixed operational ankle anchors and current rays.

This is an independent observation route, not a relaxed 3D triangulation gate.
"""

import numpy as np


def solve_ankle_bearing_translation(calibration, rotation, anchors, observations):
    if calibration.length_unit not in ("mm", "millimeter"):
        raise ValueError("ankle bearings require millimeter calibration")
    rotation = np.asarray(rotation, float).reshape(3, 3)
    if (
        not np.isfinite(rotation).all()
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
        or np.linalg.det(rotation) < 0
    ):
        raise ValueError("invalid camera rotation")
    normal = np.zeros((3, 3))
    rhs = np.zeros(3)
    links = []
    center_right = -calibration.R.T @ calibration.T.reshape(3)
    for foot, joint in [("left", 15), ("right", 16)]:
        if foot not in anchors:
            continue
        for view in ["left", "right"]:
            point = np.asarray(observations[view][joint], float)
            size = (
                calibration.left_image_size
                if view == "left"
                else calibration.right_image_size
            )
            if (
                not np.isfinite(point).all()
                or point[2] < 0.25
                or not (0 <= point[0] <= size[0] - 1 and 0 <= point[1] <= size[1] - 1)
            ):
                continue
            ray = np.r_[calibration.undistort_normalized(point[None, :2], view)[0], 1.0]
            offset = np.zeros(3)
            if view == "right":
                ray = calibration.R.T @ ray
                offset = rotation @ center_right
            ray = rotation @ ray
            ray /= np.linalg.norm(ray)
            projector = np.eye(3) - np.outer(ray, ray)
            weight = min(1.0, point[2])
            normal += weight * projector
            rhs += weight * projector @ (np.asarray(anchors[foot]) - offset)
            links.append((foot, view, point[:2]))
    base = {
        "ray_count": len(links),
        "source": "current_2d_ankles_and_existing_world_anchors",
        "triangulation_mask_modified": False,
    }
    if (
        len(links) < 3
        or len({x[0] for x in links}) < 2
        or len({x[1] for x in links}) < 2
    ):
        return None, dict(
            base, status="unavailable", reason="insufficient_two_foot_two_view_rays"
        )
    eig = np.linalg.eigvalsh(normal)
    base["normal_eigenvalues"] = eig.tolist()
    if eig[0] < 1e-5:
        return None, dict(base, status="rejected", reason="ill_conditioned_ankle_rays")
    translation = np.linalg.solve(normal, rhs)
    errors = []
    for foot, view, pixel in links:
        xyz = rotation.T @ (np.asarray(anchors[foot]) - translation)
        view_xyz = (
            xyz if view == "left" else calibration.R @ xyz + calibration.T.reshape(3)
        )
        if view_xyz[2] <= 0:
            return None, dict(
                base, status="rejected", reason="ankle_anchor_nonpositive_depth"
            )
        projected = (
            calibration.project_left if view == "left" else calibration.project_right
        )(xyz[None])[0]
        errors.append(float(np.linalg.norm(projected - pixel)))
    base["reprojection_errors_px"] = [
        {"foot": f, "view": v, "error": e} for (f, v, _), e in zip(links, errors)
    ]
    base["maximum_reprojection_px"] = max(errors)
    if max(errors) > 10:
        return None, dict(
            base, status="rejected", reason="ankle_anchor_reprojection_above_10px"
        )
    return translation, dict(base, status="accepted", reason=None)
