"""Bounded direct upper-arm pose candidate; frozen mainline archives stay untouched."""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from compare_capture_arm_bias import RECORDS, ROOT, read, stats
from experiment_forearm_residual_parts import FITS

BODY_INDICES = [12, 13, 15, 16, 17, 18]  # native 13/14 collars,16/17 shoulders,18/19 elbows


def load_model(assets):
    """Use the fitter's compatibility setup and native SMPLH/MANO assets."""
    from pose_app.smplh_fitting import pipeline  # noqa: F401

    # isort: split
    import smplx
    from smplx.utils import Struct

    with (assets / "SMPLH_male.pkl").open("rb") as stream:
        data = pickle.load(stream, encoding="latin1")
    for short, side in (("l", "LEFT"), ("r", "RIGHT")):
        with (assets / f"MANO_{side}.pkl").open("rb") as stream:
            mano = pickle.load(stream, encoding="latin1")
        data["hands_components" + short] = np.asarray(mano["hands_components"], np.float32)
        data["hands_mean" + short] = np.asarray(mano["hands_mean"], np.float32)
    model = smplx.SMPLH(str(assets / "SMPLH_male.pkl"), data_struct=Struct(**data), gender="male",
                        use_pca=False, flat_hand_mean=True, batch_size=1).cuda()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--surface-penalty", type=float, default=0.)
    args = ap.parse_args()
    if not np.isfinite(args.surface_penalty) or args.surface_penalty < 0:
        raise ValueError("invalid_surface_penalty")
    if args.output.exists():
        raise ValueError("refuse_existing_output")
    args.output.mkdir(parents=True)
    sources = {k: v for k, v in FITS.items() if k in ("old_pmpose_wrist_off", "new_pmpose_wrist_off")}
    assets = ROOT / "third_party/WiLoR/mano_data/models"
    protocol = {
        "inputs": {k: str(v) for k, v in sources.items()}, "assets": str(assets),
        "regressor": str(ROOT / "models/smpl/J_regressor_coco.npy"),
        "code": "realtime_app/experiment_local_arm_pose.py",
        "variable": "Release direct collar/shoulder/elbow axis-angle around frozen pose, 0.15rad per component box",
        "body_pose_indices": BODY_INDICES, "steps": 100, "lr": .005, "batch": 32,
        "controls": "Same body quality, mask, COCO regressor, full SMPLH and frozen beta/root/transl/legs/fingers",
        "objective": "Original all-body weighted squared observation residual; no new labels or detector",
        "success": "Initial full-mesh replay<=2e-6m, finite candidate, lower-body COCO displacement<=1mm; report full residual and pose/motion changes",
        "stop": "Replay mismatch/nonfinite abort, preserve failed protocol/log; no retuning",
        "limits": "Candidate leaves VPoser manifold; no physical contact or anatomical accuracy certification; not accepted for main fit"
    }
    protocol["surface_penalty"] = {
        "weight": args.surface_penalty, "on": "per-frame maximum point displacement squared hinge",
        "soft_margins_m": {"sole": .0008, "palm": .008, "lower_coco": .0008},
        "hard_audit_m": {"sole": .001, "palm": .010, "lower_coco": .001},
        "sets": str(RECORDS / "G20260930_smplh_contact_set_fix_v1_run3/contact_vertex_sets.json"),
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    import torch
    from pose_app.smpl_coco_observation import (
        load_coco17_regressor,
        regress_coco17_torch,
    )

    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for registered experiment")
    model = load_model(assets)
    reg = load_coco17_regressor(protocol["regressor"])
    sets = read(Path(protocol["surface_penalty"]["sets"]))["sets"] if args.surface_penalty else {}
    surface_indices = {part: sorted({i for side in ("left", "right")
                                    for group in sets[f"{side}_{part}_surface_candidate"].values() for i in group})
                       for part in ("sole", "palm")} if sets else {}
    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32, device="cuda")
    summaries = {}
    for name, path in sources.items():
        with np.load(path) as archive:
            src = {k: archive[k] for k in ("pair_id", "vertices", "predicted_coco", "body_pose", "betas", "global_orient", "transl",
                   "left_hand_pose", "right_hand_pose", "raw_triangulated_points", "body_accepted", "body_quality")}
        n = len(src["pair_id"])
        if not np.array_equal(src["pair_id"], np.arange(n)):
            raise ValueError("frame mismatch")
        poses, cocos, vertices, histories = [], [], [], []
        start_time = time.perf_counter()
        parity_max = 0.
        for start in range(0, n, 32):
            end = min(start + 32, n)
            initial = tensor(src["body_pose"][start:end]).reshape(-1, 21, 3)
            delta = torch.nn.Parameter(torch.zeros(end - start, len(BODY_INDICES), 3, device="cuda"))
            selector = torch.zeros(21, len(BODY_INDICES), device="cuda")
            selector[BODY_INDICES, torch.arange(len(BODY_INDICES))] = 1
            fixed = {k: tensor(src[k][start:end]) for k in ("global_orient", "transl", "left_hand_pose", "right_hand_pose")}
            beta = tensor(src["betas"]).expand(end - start, -1)
            mask = src["body_accepted"][start:end]
            target = tensor(np.where(mask[..., None], src["raw_triangulated_points"][start:end] / 1000, 0))
            weights = tensor(np.where(mask, src["body_quality"][start:end], 0))
            original_mesh = tensor(src["vertices"][start:end]) if args.surface_penalty else None
            original_coco = tensor(src["predicted_coco"][start:end]) if args.surface_penalty else None
            optim = torch.optim.Adam([delta], lr=.005)
            def forward(initial=initial, selector=selector, delta=delta, beta=beta,
                        fixed=fixed, target=target, weights=weights,
                        original_mesh=original_mesh, original_coco=original_coco):
                pose = initial + torch.einsum("jk,nkc->njc", selector, delta)
                out = model(betas=beta, body_pose=pose.reshape(-1, 63), return_verts=True, **fixed)
                coco = regress_coco17_torch(out.vertices, reg)
                loss = ((coco - target).square().sum(-1) * weights).sum() / weights.sum().clamp_min(1e-8)
                if args.surface_penalty:
                    for part, ids in surface_indices.items():
                        displacement = torch.linalg.vector_norm(out.vertices[:, ids] - original_mesh[:, ids], dim=-1).amax(dim=1)
                        margin = protocol["surface_penalty"]["soft_margins_m"][part]
                        loss = loss + args.surface_penalty * torch.relu(displacement - margin).square().mean()
                    lower = torch.linalg.vector_norm(coco[:, 11:] - original_coco[:, 11:], dim=-1).amax(dim=1)
                    loss = loss + args.surface_penalty * torch.relu(lower - .0008).square().mean()
                return pose, out, coco, loss
            with torch.no_grad():
                _, out, _, loss0 = forward()
                parity = float(np.max(np.abs(out.vertices.cpu().numpy() - src["vertices"][start:end])))
                parity_max = max(parity_max, parity)
                if parity > 2e-6:
                    raise ValueError(f"replay_mismatch:{parity}")
            for _ in range(100):
                optim.zero_grad()
                _, _, _, loss = forward()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite_loss")
                loss.backward()
                if not torch.isfinite(delta.grad).all():
                    raise ValueError("nonfinite_gradient")
                optim.step()
                with torch.no_grad():
                    delta.clamp_(-.15, .15)
            with torch.no_grad():
                pose, out, coco, loss1 = forward()
                poses.append(pose.reshape(-1, 63).cpu().numpy())
                cocos.append(coco.cpu().numpy())
                vertices.append(out.vertices.cpu().numpy())
                histories.append({"start": start, "end_exclusive": end, "before": float(loss0), "after": float(loss1)})
            print(name, end, "/", n, flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        pose, coco, mesh = np.concatenate(poses), np.concatenate(cocos), np.concatenate(vertices)
        displacement = np.linalg.norm(coco - src["predicted_coco"], axis=-1) * 1000
        before = np.linalg.norm(src["predicted_coco"] * 1000 - src["raw_triangulated_points"], axis=-1)
        after = np.linalg.norm(coco * 1000 - src["raw_triangulated_points"], axis=-1)
        summaries[name] = {"elapsed_seconds": elapsed, "replay_max_m": parity_max,
            "lower_body_max_change_mm": float(displacement[:, 11:].max()),
            "lower_body_under_1mm": bool(displacement[:, 11:].max() <= 1),
            "history": histories, "accepted_for_main_fit": False,
            "groups": {g: {"before_mm": stats(before[:, js][src["body_accepted"][:, js]]),
                           "after_mm": stats(after[:, js][src["body_accepted"][:, js]])}
                       for g, js in (("arms", [7, 8, 9, 10]), ("shoulders", [5, 6]), ("legs", [13, 14, 15, 16]))}}
        np.savez_compressed(args.output / (name + ".npz"), pair_id=src["pair_id"], body_pose=pose,
                            predicted_coco=coco, vertices=mesh, faces=model.faces,
                            betas=src["betas"], global_orient=src["global_orient"], transl=src["transl"],
                            left_hand_pose=src["left_hand_pose"], right_hand_pose=src["right_hand_pose"])
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
