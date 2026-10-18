#!/usr/bin/env python3
"""Strict no-contact (Stage C) vs foot-only (Stage D) SMPL window comparison.

Clean-male mainline only. This runner never imports the historical SMPL-X
benchmark route (``benchmark_smplx_r3_integration``,
``benchmark_smplx_structured_contact``, ``diagnose_smplx_loss_gradients``).

Both routes invoke the current clean male entry
``G20260924_smpl_vposer_shared_beta_v1/fit_vposer_shared_beta.py`` on the same
full-clip inputs:

- male SMPL, 6890 vertices; model-side supervision is
  ``J_SMPL = J_regressor_coco @ vertices`` only;
- one shared beta vector per route, fitted with the identical deterministic
  procedure from the identical initialization (zero-beta orientation init);
- hand contact forced off (``--hand-contact-weight 0``);
- foot-only is a refinement that starts from the completed Stage C result of
  the same run with beta frozen, using only the current-run
  ``foot_contact_weight`` labels;
- ground transform comes from the current ``scene_transforms.npz``
  (``rotation_ground_from_left`` / ``translation_ground_from_left_mm``);
- the foot term is pseudo-Huber of model-side COCO ankle 15/16 ``z_G`` with
  delta = 25 mm;
- Stage 1 uses ``support`` only; ``swing`` / ``ambiguous`` / ``invalid`` get
  weight 0; Stage 2 uses ``stage2_contact_candidate`` only (no force-all);
  transition/warming_up weights are 0;
- COCO index 14 (right knee) never enters the contact loss or pass evidence.

This task only audits and repairs the runner; it does not run fitting here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "left": ROOT / "research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/c3_predictions/left/pmpose/raw_predictions.json",
    "right": ROOT / "research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/c3_predictions/right/pmpose/raw_predictions.json",
    "triangulation": ROOT / "research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1/full448_formal/triangulation.npz",
    "scene_transforms": ROOT / "research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1/scene_stage_ground_v3_pair_replay/scene_transforms.npz",
    "stage_jsonl": ROOT / "research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1/scene_stage_ground_v3_pair_replay/stage1_stage2.jsonl",
    "contact_labels": ROOT / "research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1/contact_labels_stage_audit_v2/contact_labels.npz",
    "model": ROOT / "models/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl",
    "regressor": ROOT / "models/smpl/J_regressor_coco.npy",
    "vposer_dir": ROOT / "models/VPoser02_05/V02_05",
    "walker_topology": ROOT / "research_records/engineering_validation/G20260918_offline_walker_frame_evidence_v1/coarse_complete_model_v2_camera_rail/coarse_walker_model.json",
    "fit_entry": ROOT / "research_records/engineering_validation/G20260924_smpl_vposer_shared_beta_v1/fit_vposer_shared_beta.py",
}

COCO_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
RIGHT_KNEE = 14
ANKLES = (15, 16)
CONTACT_DELTA_M = 0.025


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Strict Stage-C no-contact vs Stage-D foot-only comparison (clean male SMPL).")
    p.add_argument("--left", type=Path, default=DEFAULTS["left"])
    p.add_argument("--right", type=Path, default=DEFAULTS["right"])
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, default=DEFAULTS["model"])
    p.add_argument("--regressor", type=Path, default=DEFAULTS["regressor"])
    p.add_argument("--vposer-dir", type=Path, default=DEFAULTS["vposer_dir"])
    p.add_argument("--triangulation", type=Path, default=DEFAULTS["triangulation"])
    p.add_argument("--scene-transforms", type=Path, default=DEFAULTS["scene_transforms"])
    p.add_argument("--stage-jsonl", type=Path, default=DEFAULTS["stage_jsonl"])
    p.add_argument("--contact-labels", type=Path, default=DEFAULTS["contact_labels"])
    p.add_argument("--walker-topology", type=Path, default=DEFAULTS["walker_topology"])
    p.add_argument("--fit-entry", type=Path, default=DEFAULTS["fit_entry"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--start", type=int, default=60)
    p.add_argument("--end", type=int, default=90)
    p.add_argument("--base-steps", type=int, default=120)
    p.add_argument("--beta-steps", type=int, default=120)
    p.add_argument("--joint-steps", type=int, default=120)
    p.add_argument("--contact-steps", type=int, default=80)
    p.add_argument("--foot-contact-weight", type=float, default=0.02)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return p


ALLOWED_STAGES = frozenset({
    "warming_up", "transition",
    "stage1_walker_static_human_moving", "stage2_feet_static_walker_moving",
})
STAGE1_FULL = "stage1_walker_static_human_moving"
STAGE2_FULL = "stage2_feet_static_walker_moving"


def read_operational_stages(path: Path) -> list[str]:
    """Read one normalized operational stage per JSONL frame.

    ``record["stage"]`` may be a nested object (prefer ``operational``, fall
    back to ``confirmed``) or a plain string. Anything unparseable or outside
    the four allowed values is recorded as ``transition`` -- never defaulted
    to Stage 1/2.
    """
    stages: list[str] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            stages.append("transition")
            continue
        raw = record.get("stage", None)
        if isinstance(raw, dict):
            value = raw.get("operational", raw.get("confirmed", None))
        else:
            value = raw
        value = str(value).strip() if value is not None else ""
        stages.append(value if value in ALLOWED_STAGES else "transition")
    return stages


def _pseudo_huber(z: "np.ndarray", delta: float) -> "np.ndarray":
    import numpy as np
    return delta * delta * (np.sqrt(1.0 + (z / delta) ** 2) - 1.0)


def strict_foot_weights(labels: "np.ndarray", stages: list[str], start: int, end: int) -> tuple["np.ndarray", dict]:
    """Build window foot weights honoring the strict label contract.

    Stage 1 (full string ``stage1_walker_static_human_moving``) allows only
    ``support``; Stage 2 (full string ``stage2_feet_static_walker_moving``)
    allows only ``stage2_contact_candidate``. Every other label, every
    transition/warming_up frame, and COCO 14 are exactly zero. The label
    file must cover exactly the same frames as the stage list.
    """
    import numpy as np
    foot_w = np.asarray(labels["foot_contact_weight"], dtype=np.float64)[:, :]
    foot_label = np.asarray(labels["foot_label"])
    hand_w = np.asarray(labels["hand_contact_weight"], dtype=np.float64)
    if np.any(hand_w != 0):
        raise ValueError("hand contact must stay off: hand_contact_weight must be all zero")
    if len(stages) != foot_w.shape[0]:
        raise ValueError("stage frame count must equal contact label frame count")
    n = foot_w.shape[0]
    if not (0 <= start <= end < n):
        raise ValueError("invalid window")
    weights = np.zeros((end - start + 1, 2), dtype=np.float64)
    counts = {"support": 0, "stage2_contact_candidate": 0, "zeroed": 0}
    for out_i, t in enumerate(range(start, end + 1)):
        stage = stages[t]
        for side in range(2):
            label = str(foot_label[t, side])
            allowed = (stage == STAGE1_FULL and label == "support") or (
                stage == STAGE2_FULL and label == "stage2_contact_candidate"
            )
            if allowed:
                weights[out_i, side] = float(foot_w[t, side])
                counts[label] += 1
            else:
                counts["zeroed"] += 1
    return weights, counts


def window_metrics(result_npz: Path, raw_npz: Path, tri_npz: Path, scene_npz: Path,
                   foot_weights: "np.ndarray", start: int, end: int,
                   calibration_dir: Path) -> dict:
    """Compare fit outputs against full-sequence references.

    ``result_npz``/``raw_npz`` are window-local (fit with ``--start/--end``),
    so they are indexed ``0..fit_frames-1``; ``tri_npz``/``scene_npz`` still
    cover the full sequence and are indexed by ``start..end``.
    """
    import numpy as np
    from pose_app.fisheye_camera import fisheye_project_numpy, load_stereo_fisheye
    res = np.load(result_npz, allow_pickle=True)
    raw_obs = np.load(raw_npz, allow_pickle=True)
    tri = np.load(tri_npz, allow_pickle=True)
    scene = np.load(scene_npz, allow_pickle=True)
    cal = load_stereo_fisheye(calibration_dir)
    pred = np.asarray(res["predicted_coco"], dtype=np.float64)
    left2d = np.asarray(raw_obs["left_2d"], dtype=np.float64)
    right2d = np.asarray(raw_obs["right_2d"], dtype=np.float64)
    n = end - start + 1
    if pred.shape[0] != n or left2d.shape[0] != n:
        raise RuntimeError("fit output frame count does not match the requested window")
    accepted = np.asarray(tri["accepted"], dtype=bool)[start:end + 1]
    pred_w, acc_w = pred, accepted
    e2l = np.full((n, 17), np.nan)
    e2r = np.full((n, 17), np.nan)
    for t in range(n):
        pl = fisheye_project_numpy(pred_w[t], cal.K0, cal.D0)
        jr = (cal.R_cam0_to_cam1 @ (pred_w[t] * 1000.0).T + cal.T_cam0_to_cam1_mm.reshape(3, 1)).T / 1000.0
        pr = fisheye_project_numpy(jr, cal.K1, cal.D1)
        e2l[t] = np.linalg.norm(pl - left2d[t, :, :2], axis=-1)
        e2r[t] = np.linalg.norm(pr - right2d[t, :, :2], axis=-1)
    finite_l = np.isfinite(e2l) & acc_w
    finite_r = np.isfinite(e2r) & acc_w
    tri_raw = np.asarray(tri["raw"], dtype=np.float64)[start:end + 1] / 1000.0
    with np.errstate(invalid="ignore"):
        e3 = np.linalg.norm(pred_w - tri_raw, axis=-1) * 1000.0
    m3 = acc_w & np.isfinite(e3)
    vel = np.diff(pred_w, axis=0)
    accel = np.diff(vel, axis=0)
    coco_accel = np.linalg.norm(accel, axis=-1)
    root = pred_w[:, [11, 12]].mean(axis=1)
    root_accel = np.linalg.norm(np.diff(root, n=2, axis=0), axis=-1) if n >= 3 else np.array([])
    rmat = np.asarray(scene["rotation_ground_from_left"], dtype=np.float64)[start:end + 1]
    tvec = np.asarray(scene["translation_ground_from_left_mm"], dtype=np.float64)[start:end + 1] / 1000.0
    jg = np.einsum("nij,nvj->nvi", rmat, pred_w) + tvec[:, None, :]
    zg = jg[:, list(ANKLES), 2]
    proxy = np.sqrt(_pseudo_huber(zg, CONTACT_DELTA_M))
    proxy_rms = float(np.sqrt(np.mean((foot_weights > 0) * proxy ** 2))) if np.any(foot_weights > 0) else 0.0
    neg_depth = 0
    for t in range(n):
        left_pts = (np.linalg.inv(rmat[t]) @ (pred_w[t] - tvec[t]).T).T
        right_mm = (cal.R_cam0_to_cam1 @ (left_pts * 1000.0).T + cal.T_cam0_to_cam1_mm.reshape(3, 1)).T
        neg_depth += int(np.sum(left_pts[:, 2] <= 0) + np.sum(right_mm[:, 2] <= 0))
    joint_p95 = {}
    for j in range(17):
        vals = e3[:, j][m3[:, j]]
        joint_p95[COCO_NAMES[j]] = float(np.percentile(vals, 95)) if vals.size else None
    both2d = np.r_[e2l[finite_l], e2r[finite_r]] if (finite_l.any() or finite_r.any()) else np.array([])
    return {
        "window": [start, end],
        "reproj_left_p95_px": float(np.percentile(e2l[finite_l], 95)) if finite_l.any() else None,
        "reproj_right_p95_px": float(np.percentile(e2r[finite_r], 95)) if finite_r.any() else None,
        "reproj_p95_px": float(np.percentile(both2d, 95)) if both2d.size else None,
        "tri_3d_median_mm": float(np.median(e3[m3])) if m3.any() else None,
        "tri_3d_p95_mm": float(np.percentile(e3[m3], 95)) if m3.any() else None,
        "joint_p95_mm": joint_p95,
        "right_knee_p95_mm": joint_p95["right_knee"],
        "right_knee_in_contact_or_pass": False,
        "root_accel_p95_mm": float(np.percentile(np.linalg.norm(root_accel, axis=-1) * 1000.0, 95)) if root_accel.size else None,
        "coco_accel_p95_mm": float(np.percentile(coco_accel, 95)) if coco_accel.size else None,
        "negative_depth_count": neg_depth,
        "foot_contact_proxy_rms": proxy_rms,
        "valid_contact_frames": int(np.sum((foot_weights > 0).any(axis=1))),
    }


def run_fit(python: str, fit_entry: Path, args: argparse.Namespace, out_dir: Path,
            foot_weight: float, stage: str) -> tuple[Path, Path]:
    cmd = [python, str(fit_entry), "--left", str(args.left), "--right", str(args.right),
           "--calibration-dir", str(args.calibration_dir), "--model", str(args.model),
           "--regressor", str(args.regressor), "--vposer-dir", str(args.vposer_dir),
           "--output-dir", str(out_dir), "--start", str(args.start), "--end", str(args.end),
           "--base-steps", str(args.base_steps),
           "--beta-steps", str(args.beta_steps), "--joint-steps", str(args.joint_steps),
           "--contact-steps", str(args.contact_steps),
           "--contact-labels", str(args.contact_labels),
           "--scene-transforms", str(args.scene_transforms),
           "--foot-contact-weight", str(foot_weight), "--hand-contact-weight", "0.0",
           "--device", args.device]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / f"stdout_{stage}.log").write_text(proc.stdout, encoding="utf-8")
    (out_dir / f"stderr_{stage}.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"fit entry failed for {stage}: {proc.stderr[-2000:]}")
    return out_dir / "result.npz", out_dir / "metrics.json"


def main() -> int:
    args = build_parser().parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refuse to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for key in ("left", "right", "triangulation", "scene_transforms", "stage_jsonl",
                "contact_labels", "model", "regressor", "vposer_dir", "fit_entry"):
        if not Path(getattr(args, key)).exists():
            raise FileNotFoundError(f"missing required input {key}: {getattr(args, key)}")
    if not args.calibration_dir.exists():
        raise FileNotFoundError(f"missing calibration dir: {args.calibration_dir}")
    walker = json.loads(Path(args.walker_topology).read_text(encoding="utf-8-sig"))
    if "coarse_walker_model" not in json.dumps(walker) and not isinstance(walker, dict):
        raise ValueError("walker topology file is not the fixed static topology")
    import numpy as np
    labels = np.load(args.contact_labels, allow_pickle=True)
    stages = read_operational_stages(args.stage_jsonl)
    foot_weights, label_counts = strict_foot_weights(labels, stages, args.start, args.end)
    (out / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    stage_c_dir = out / "stage_c_no_contact"
    stage_d_dir = out / "stage_d_foot_only"
    stage_c_dir.mkdir()
    stage_d_dir.mkdir()
    fit_frames = args.end - args.start + 1
    run_fit(sys.executable, args.fit_entry, args, stage_c_dir, 0.0, "stage_c")
    fit_c = json.loads((stage_c_dir / "metrics.json").read_text(encoding="utf-8"))
    if fit_c.get("window") != [args.start, args.end] or fit_c.get("frames") != fit_frames:
        raise RuntimeError("Stage C fit did not cover the requested window")
    if fit_c.get("stage_d_contact") is not None:
        raise RuntimeError("Stage C no-contact run must not produce a Stage D result")
    res_c = np.load(stage_c_dir / "result_stage_c_joint.npz", allow_pickle=True)
    beta_c = np.asarray(res_c["betas"][0], dtype=np.float64)
    metrics_c = window_metrics(stage_c_dir / "result_stage_c_joint.npz",
                               stage_c_dir / "raw_observations.npz",
                               args.triangulation, args.scene_transforms,
                               foot_weights, args.start, args.end, args.calibration_dir)
    metrics_c["beta"] = beta_c.tolist()
    (stage_c_dir / "result.npz").write_bytes((stage_c_dir / "result_stage_c_joint.npz").read_bytes())
    (stage_c_dir / "metrics.json").write_text(
        json.dumps({"route": "stage_c_no_contact", "foot_contact_weight": 0.0,
                    "hand_contact_weight": 0.0,
                    "window": [args.start, args.end], "fit_frames": fit_frames,
                    "initialization": "identical_zero_beta_orientation_init",
                    "metrics": metrics_c}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Foot run: the fit process itself must complete A->B->C, save Stage C,
    # freeze beta, then continue to Stage D in the same process.
    run_fit(sys.executable, args.fit_entry, args, stage_d_dir,
            args.foot_contact_weight, "stage_d")
    fit_d = json.loads((stage_d_dir / "metrics.json").read_text(encoding="utf-8"))
    if fit_d.get("window") != [args.start, args.end] or fit_d.get("frames") != fit_frames:
        raise RuntimeError("Stage D fit did not cover the requested window")
    if not fit_d.get("stage_c_to_d_same_process"):
        raise RuntimeError("foot run did not complete Stage C->D in the same process")
    if not fit_d.get("beta_frozen_during_stage_d"):
        raise RuntimeError("beta was not really frozen during Stage D")
    if bool(fit_d.get("full_sequence_fit")):
        raise RuntimeError("fit must be window-only, not full-sequence")
    res_d = np.load(stage_d_dir / "result_stage_d_contact.npz", allow_pickle=True)
    beta_d = np.asarray(res_d["betas"][0], dtype=np.float64)
    metrics_d = window_metrics(stage_d_dir / "result_stage_d_contact.npz",
                               stage_d_dir / "raw_observations.npz",
                               args.triangulation, args.scene_transforms,
                               foot_weights, args.start, args.end, args.calibration_dir)
    metrics_d["beta"] = beta_d.tolist()
    (stage_d_dir / "result.npz").write_bytes((stage_d_dir / "result_stage_d_contact.npz").read_bytes())
    (stage_d_dir / "metrics.json").write_text(
        json.dumps({"route": "stage_d_foot_only",
                    "foot_contact_weight": args.foot_contact_weight,
                    "hand_contact_weight": 0.0,
                    "window": [args.start, args.end], "fit_frames": fit_frames,
                    "stage_c_to_d_same_process": True,
                    "beta_frozen_during_stage_d": True,
                    "initialization": "identical_zero_beta_orientation_init",
                    "metrics": metrics_d}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    comparison = {
        "schema_version": "strict_foot_only_window_v1",
        "window": [args.start, args.end],
        "fit_frames": fit_frames,
        "stage_c_to_d_same_process": True,
        "beta_frozen_during_stage_d": True,
        "engineering_comparison_only": True,
        "inputs": {k: str(Path(getattr(args, k)).resolve()) for k in
                   ("left", "right", "triangulation", "scene_transforms", "stage_jsonl",
                    "contact_labels", "model", "regressor", "vposer_dir", "walker_topology",
                    "fit_entry", "calibration_dir")},
        "initialization_shared": True,
        "hand_contact_weight": 0.0,
        "foot_contact_weight_stage_d": args.foot_contact_weight,
        "contact_contract": {
            "model_side_point": "J_SMPL = J_regressor_coco @ vertices (male SMPL 6890)",
            "foot_term": "pseudo_huber(z_G of COCO ankles 15/16; delta=25mm)",
            "stage1_labels": ["support"],
            "stage2_labels": ["stage2_contact_candidate"],
            "transition_weight": 0,
            "right_knee_coco14_in_contact_or_pass": False,
        },
        "label_counts": label_counts,
        "stage_c_no_contact": metrics_c,
        "stage_d_foot_only": metrics_d,
        "interpretation": ("engineering_comparison_only: differences here do not establish "
                           "physical touch-down, load bearing, support, or true 3-D accuracy."),
    }
    (out / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source_audit = {
        "old_benchmark_imports": False,
        "old_contact_target_input": False,
        "old_motion_json_default": False,
        "forbidden_modules": ["realtime_app.tools.benchmark_smplx_r3_integration",
                              "realtime_app.tools.benchmark_smplx_structured_contact",
                              "realtime_app.tools.diagnose_smplx_loss_gradients"],
        "forbidden_inputs": ["interaction_distance_records.jsonl", "old motion.json",
                             "old prefit JSONL", "old HTML", "old dynamic walker poses"],
        "fit_entry": str(args.fit_entry.resolve()),
        "model_topology": "male SMPL 6890 vertices",
        "observation": "J_SMPL = J_regressor_coco @ vertices",
    }
    (out / "source_audit.json").write_text(json.dumps(source_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "comparison_complete", "window": [args.start, args.end],
                      "fit_frames": fit_frames,
                      "stage_c_to_d_same_process": True,
                      "beta_frozen_during_stage_d": True,
                      "outputs": ["stage_c_no_contact/metrics.json", "stage_c_no_contact/result.npz",
                                  "stage_d_foot_only/metrics.json", "stage_d_foot_only/result.npz",
                                  "comparison.json", "source_audit.json", "command.txt"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
