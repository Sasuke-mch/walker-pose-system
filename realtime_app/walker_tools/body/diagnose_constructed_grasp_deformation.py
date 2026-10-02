"""One-frame controlled SMPL-H hand deformation audit; not a fitting route.

Disable corrective blend shapes / neutralize wrist rotations only as causal
diagnostics. Never export these ablations as reconstructed or accepted poses.
"""
from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
from walker_tools._compat import PROJECT_ROOT as _tool_project_root
_tool_prepare_imports()

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rodrigues
from smplx.utils import Struct

ROOT = _tool_project_root
from pose_app.smpl_coco_observation import _install_legacy_smpl_pickle_compatibility
from pose_app.constructed_grasp_refinement import load_grasp, wrist_rotations
from walker_tools.hands.audit_fixed_mano_grip import crossing_pairs


def main():
    ap = argparse.ArgumentParser()
    for name in ("source-result", "refined-result", "grasp", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("refuse existing diagnostic output")
    _install_legacy_smpl_pickle_compatibility()
    torch.set_num_threads(2)
    source = np.load(args.source_result, allow_pickle=False)
    refined = np.load(args.refined_result, allow_pickle=False)
    grasp = load_grasp(args.grasp)
    mesh = np.load(args.grasp.with_suffix(".npz"), allow_pickle=False)
    frame = int(refined["pair_id"][0])
    model_path = ROOT / "third_party/WiLoR/mano_data/models/SMPLH_male.pkl"
    with model_path.open("rb") as f:
        asset = pickle.load(f, encoding="latin1")
    data = dict(asset)
    data.update(hands_componentsl=np.eye(45), hands_componentsr=np.eye(45), hands_meanl=np.zeros(45), hands_meanr=np.zeros(45))
    model = smplx.SMPLHLayer(str(model_path), data_struct=Struct(**data), use_pca=False, flat_hand_mean=True)
    def t(a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float32)
    def matrices(aa):
        return batch_rodrigues(t(aa).reshape(-1, 3)).reshape(1, -1, 3, 3)
    zero = torch.eye(3).expand(1, 21, 3, 3).clone()
    body_final = matrices(refined["body_pose"][0])
    body_source = matrices(source["body_pose"][frame])
    fingers = [matrices(grasp["hands"][s]["hand_pose_axis_angle"]) for s in ("left", "right")]
    dom = np.asarray(asset["weights"]).argmax(1)
    posedirs = model.posedirs.detach().clone()
    cases = [("neutral_body_both_hands", zero, False),
             ("source_body_constructed_fingers", body_source, False),
             ("refined_body", body_final, False),
             ("refined_body_without_pose_correctives", body_final, True)]
    wrist_neutral = body_final.clone()
    wrist_neutral[:, [19, 20]] = torch.eye(3)
    cases.append(("refined_body_neutral_local_wrists", wrist_neutral, False))
    wrist_only = zero.clone()
    wrist_only[:, [19, 20]] = body_final[:, [19, 20]]
    cases.append(("only_refined_local_wrists", wrist_only, False))
    records = {}
    for name, body, no_correctives in cases:
        with torch.no_grad():
            model.posedirs.copy_(torch.zeros_like(posedirs) if no_correctives else posedirs)
            out = model(betas=t(source["betas"]), body_pose=body,
                global_orient=torch.eye(3).reshape(1, 1, 3, 3),
                left_hand_pose=fingers[0], right_hand_pose=fingers[1], transl=torch.zeros(1, 3))
            wr = wrist_rotations(torch.eye(3)[None], body, model.parents)[0]
        row = {}
        for j, side in enumerate(("left", "right")):
            idx = mesh[f"{side}_vertex_indices"]
            faces = mesh[f"{side}_faces"]
            q = (out.vertices[0, idx] - out.joints[0, 20 + j]).numpy() @ wr[j].numpy()
            hits = crossing_pairs(q, faces)
            labels = []
            for a, b in hits:
                la = Counter(dom[idx[faces[a]]]).most_common(1)[0][0]
                lb = Counter(dom[idx[faces[b]]]).most_common(1)[0][0]
                labels.append("/".join(map(str, sorted((int(la), int(lb))))))
            error = np.linalg.norm(q - mesh[f"{side}_vertices_local_m"], axis=-1) * 1000
            row[side] = {"crossing_count": len(hits), "crossing_joint_pairs": dict(Counter(labels)),
                "local_vertex_difference_mm": {"median": float(np.median(error)), "max": float(error.max())},
                "local_wrist_axis_angle_deg": np.rad2deg(Rotation.from_matrix(body[0, 19 + j].numpy()).as_rotvec()).tolist(),
                "crossing_pairs": hits}
        records[name] = row
        print(name, {s: row[s]["crossing_count"] for s in row}, flush=True)
    model.posedirs.copy_(posedirs)
    args.output.write_text(json.dumps({"status": "controlled_deformation_diagnostic_not_a_solution",
        "pair_id": frame, "inputs": {k: str(v.resolve()) for k, v in vars(args).items()},
        "cases": records, "limits": "one frame causal probes; neutral wrists and disabled correctives are not proposed fitting fixes"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
