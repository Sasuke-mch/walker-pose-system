"""Shape-independent manual wrist references, never constructed grasp parameters."""
import json
from pathlib import Path

import numpy as np


def load_reference(path, calibration_dir, allow_diagnostic=False):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "annotated_wrist_targets_v1" or data.get("finger_parameters_used") is not False:
        raise ValueError("expected independent annotated wrist reference")
    if data.get("assumption") != "wrist_fixed_relative_to_walker_with_rigid_camera_mount":
        raise ValueError("wrist reference requires camera-rigid walker assumption")
    if Path(data["calibration_dir"]).resolve() != Path(calibration_dir).resolve():
        raise ValueError("wrist reference calibration differs from current fit")
    positions, states = [], {}
    for side in ("left", "right"):
        hand = data["hands"][side]
        accepted = hand["state"] == "accepted_manual_reference_geometry"
        if not accepted and not allow_diagnostic:
            raise ValueError("diagnostic wrist reference requires explicit opt-in")
        # Direct multi-frame annotation solver output in camera0 metres.
        # No beta, PCA, mesh, or optimized constructed-grasp wrist is read.
        p = np.asarray(hand["diagnostic_wrist_left_camera_m"], dtype=np.float32)
        if p.shape != (3,) or not np.isfinite(p).all() or p[2] <= 0:
            raise ValueError("invalid independent wrist coordinate")
        positions.append(p)
        states[side] = {"state": hand["state"], "reasons": hand.get("reasons", [])}
    return np.stack(positions), {"source": str(Path(path).resolve()),
        "source_annotations": data["source_annotations"], "source_status": data["status"],
        "hands": states, "coordinate_frame": "left_camera", "length_unit": "m",
        "semantic": "manual visible wrist reference applied softly to SMPLH internal joints 20/21",
        "assumption": data["assumption"], "allow_diagnostic": allow_diagnostic,
        "beta_or_constructed_grasp_consumed": False}


def wrist_position_loss(joints, targets):
    """Mean squared Euclidean error in metres, both wrists and all frames."""
    return (joints[:, [20, 21]] - targets[None]).square().sum(-1).mean()
