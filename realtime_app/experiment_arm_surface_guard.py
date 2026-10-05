"""Uniform step-scale sensitivity with full-mesh guards; no per-frame fallback."""
import argparse
import json
from pathlib import Path

import numpy as np
from compare_capture_arm_bias import RECORDS, ROOT, read, stats
from experiment_forearm_residual_parts import FITS
from experiment_local_arm_pose import load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("refuse_existing_output")
    args.output.mkdir(parents=True)
    source = RECORDS / "G20261005_local_arm_pose_v1"
    scales = [0., 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.]
    sets_path = RECORDS / "G20260930_smplh_contact_set_fix_v1_run3/contact_vertex_sets.json"
    protocol = {
        "input_candidate": str(source), "sources": {k: str(v) for k, v in FITS.items() if "wrist_off" in k},
        "code": "realtime_app/experiment_arm_surface_guard.py", "scales": scales,
        "sets": str(sets_path), "variable": "Uniform sequence-wide scale of candidate axis-angle delta; no optimization",
        "guards": "Every frame sole vertex displacement<=1mm, palm<=10mm, lower COCO<=1mm; weighted body SSE no increase",
        "success": "Find whether any nonzero uniform step meets every guard, report rejected frames and metrics for every scale",
        "stop": "Scale0 original replay or scale1 candidate replay >2e-6m abort; no retuning guards",
        "limits": "Displacement protects original geometry, not correct contact; no old/new mesh mixing or frame-specific selection"
    }
    protocol["objective_baseline"] = "Scale-zero mesh replay, same batch/device arithmetic as candidates; frozen original mesh separately parity checked"
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    import torch
    from pose_app.smpl_coco_observation import (
        load_coco17_regressor,
        regress_coco17_torch,
    )

    model = load_model(ROOT / "third_party/WiLoR/mano_data/models")
    reg = load_coco17_regressor(ROOT / "models/smpl/J_regressor_coco.npy")
    sets = read(sets_path)["sets"]
    groups = {part: sorted({i for side in ("left", "right") for group in sets[f"{side}_{part}_surface_candidate"].values() for i in group})
              for part in ("sole", "palm")}
    summaries = {}
    for name in read(source / "summary.json"):
        with np.load(FITS[name]) as data, np.load(source / (name + ".npz")) as cand:
            a = {k: data[k] for k in ("body_pose", "betas", "global_orient", "transl", "left_hand_pose", "right_hand_pose",
                                     "vertices", "predicted_coco", "body_quality", "body_accepted", "raw_triangulated_points")}
            bpose, bmesh = cand["body_pose"], cand["vertices"]
        n = len(a["body_pose"])
        weights = np.where(a["body_accepted"], a["body_quality"], 0)
        target = np.where(a["body_accepted"][..., None], a["raw_triangulated_points"] / 1000, 0)
        old_sse = np.sum(np.sum((a["predicted_coco"] - target) ** 2, axis=-1) * weights, axis=1)
        rows = []
        for scale in scales:
            chunks = []
            with torch.no_grad():
                for start in range(0, n, 32):
                    sl = slice(start, min(start + 32, n))
                    def tensor(x):
                        return torch.as_tensor(x, dtype=torch.float32, device="cuda")
                    pose = a["body_pose"][sl] + scale * (bpose[sl] - a["body_pose"][sl])
                    out = model(betas=tensor(a["betas"]).expand(len(pose), -1), body_pose=tensor(pose),
                                return_verts=True, **{k: tensor(a[k][sl]) for k in ("global_orient", "transl", "left_hand_pose", "right_hand_pose")})
                    coco = regress_coco17_torch(out.vertices, reg).cpu().numpy()
                    mesh = out.vertices.cpu().numpy()
                    reference = a["vertices"][sl] if scale == 0 else bmesh[sl] if scale == 1 else None
                    if reference is not None and np.max(np.abs(mesh - reference)) > 2e-6:
                        raise ValueError("endpoint_replay_failed")
                    movement = np.linalg.norm(mesh - a["vertices"][sl], axis=-1) * 1000
                    error = np.linalg.norm(coco * 1000 - a["raw_triangulated_points"][sl], axis=-1)
                    sse = np.sum(np.sum((coco - target[sl]) ** 2, axis=-1) * weights[sl], axis=1)
                    if scale == 0:
                        # Comparing rounded archive COCO to a new GEMM creates
                        # false objective-increase warnings even at zero step.
                        old_sse[sl] = sse
                    lower = np.linalg.norm(coco[:, 11:] - a["predicted_coco"][sl, 11:], axis=-1).max(axis=1) * 1000
                    for i in range(len(pose)):
                        sole, palm = (float(movement[i, groups[part]].max()) for part in ("sole", "palm"))
                        reasons = []
                        if sole > 1: reasons.append("sole_over_1mm")
                        if palm > 10: reasons.append("palm_over_10mm")
                        if lower[i] > 1: reasons.append("lower_coco_over_1mm")
                        if sse[i] > old_sse[start + i] + 1e-8: reasons.append("body_sse_increased")
                        chunks.append({"frame": start + i, "sole_max_mm": sole, "palm_max_mm": palm,
                                       "lower_max_mm": float(lower[i]), "reasons": reasons,
                                       "arm_errors_mm": error[i, 7:11].tolist()})
            rows.append({"scale": scale, "passed_frames": sum(not r["reasons"] for r in chunks), "total_frames": n,
                         "sole_max_mm": max(r["sole_max_mm"] for r in chunks), "palm_max_mm": max(r["palm_max_mm"] for r in chunks),
                         "arm_errors_mm": stats([e for r in chunks for e in r["arm_errors_mm"]])})
            (args.output / f"{name}_scale{scale}.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
        summaries[name] = rows
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(name, "finished", flush=True)


if __name__ == "__main__":
    main()
