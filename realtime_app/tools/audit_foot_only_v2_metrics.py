#!/usr/bin/env python3
"""Read-only metric audit of a saved foot-only window result.

Recomputes negative depth, root/COCO acceleration P95 and the weighted foot
proxy RMS with corrected coordinate handling. Never modifies the fitted
result and never reruns A/B/C/D. The ankle z_G proxy targets zero height in
the fit loss while COCO ankles are not the foot sole, so this audit is a
metric correction only, not contact validation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import numpy as np

from pose_app.fisheye_camera import load_stereo_fisheye


def corrected_metrics(pred: np.ndarray, R01: np.ndarray, T01_mm: np.ndarray,
                      Rgc: np.ndarray, Tgc_mm: np.ndarray,
                      foot_weights: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    R01 = np.asarray(R01, dtype=np.float64)
    T01_m = np.asarray(T01_mm, dtype=np.float64).reshape(3) / 1000.0
    Rgc = np.asarray(Rgc, dtype=np.float64)
    Tgc_m = np.asarray(Tgc_mm, dtype=np.float64) / 1000.0
    foot_weights = np.asarray(foot_weights, dtype=np.float64)
    n = pred.shape[0]
    if pred.shape != (n, 17, 3) or Rgc.shape != (n, 3, 3) or Tgc_m.shape != (n, 3) or foot_weights.shape != (n, 2):
        raise ValueError("window input shape mismatch")
    if not np.isfinite(pred).all() or not np.isfinite(Rgc).all() or not np.isfinite(Tgc_m).all():
        raise ValueError("non-finite model or scene input")
    if not np.isfinite(foot_weights).all() or np.any(foot_weights < 0):
        raise ValueError("invalid foot weights")
    right = np.einsum("ij,nvj->nvi", R01, pred) + T01_m
    negative_left = int(np.count_nonzero(pred[:, :, 2] <= 0))
    negative_right = int(np.count_nonzero(right[:, :, 2] <= 0))
    root = pred[:, [11, 12], :].mean(axis=1)
    root_acc = np.linalg.norm(np.diff(root, n=2, axis=0), axis=-1) * 1000.0
    coco_acc = np.linalg.norm(np.diff(pred, n=2, axis=0), axis=-1) * 1000.0
    ground = np.einsum("nij,nvj->nvi", Rgc, pred) + Tgc_m[:, None, :]
    ankle_z = ground[:, [15, 16], 2]
    active = foot_weights > 0
    den = float(foot_weights.sum())
    weighted_rms_mm = float(np.sqrt((foot_weights * ankle_z ** 2).sum() / den) * 1000.0) if den > 0 else None
    return {
        "negative_left_count": negative_left,
        "negative_right_count": negative_right,
        "negative_depth_count": negative_left + negative_right,
        "root_accel_p95_mm_per_frame2": float(np.percentile(root_acc, 95)) if root_acc.size else None,
        "coco_accel_p95_mm_per_frame2": float(np.percentile(coco_acc, 95)) if coco_acc.size else None,
        "foot_proxy_weighted_rms_mm": weighted_rms_mm,
        "active_foot_observations": int(active.sum()),
        "active_contact_frames": int(active.any(axis=1).sum()),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only metric audit of a saved foot-only window result.")
    p.add_argument("--result-dir", type=Path, required=True)
    p.add_argument("--scene-transforms", type=Path, required=True)
    p.add_argument("--contact-labels", type=Path, required=True)
    p.add_argument("--calibration-dir", type=Path, required=True)
    p.add_argument("--start", type=int, default=60)
    p.add_argument("--end", type=int, default=90)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> int:
    args = build_parser().parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refuse to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for key in ("result_dir", "scene_transforms", "contact_labels", "calibration_dir"):
        if not Path(getattr(args, key)).exists():
            raise FileNotFoundError(f"missing required input {key}: {getattr(args, key)}")

    result_dir = Path(args.result_dir)
    comparison_path = result_dir / "comparison.json"
    if not comparison_path.is_file():
        raise FileNotFoundError(f"missing comparison.json under {result_dir}")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("window") != [args.start, args.end]:
        raise ValueError("requested window does not match the saved comparison window")
    if comparison.get("fit_frames") != args.end - args.start + 1:
        raise ValueError("requested frame count does not match the saved fit frames")

    cal = load_stereo_fisheye(args.calibration_dir)
    scene = np.load(args.scene_transforms, allow_pickle=True)
    labels = np.load(args.contact_labels, allow_pickle=True)
    sl = slice(args.start, args.end + 1)
    Rgc = np.asarray(scene["rotation_ground_from_left"], dtype=np.float64)[sl]
    Tgc_mm = np.asarray(scene["translation_ground_from_left_mm"], dtype=np.float64)[sl]
    foot_w = np.asarray(labels["foot_contact_weight"], dtype=np.float64)[sl]
    ankle_proxy = np.asarray(labels["ankle_proxy_height_m"], dtype=np.float64)

    routes: dict[str, dict] = {}
    for route in ("stage_c_no_contact", "stage_d_foot_only"):
        result_path = result_dir / route / "result.npz"
        if not result_path.is_file():
            raise FileNotFoundError(f"missing {result_path}")
        pred = np.asarray(np.load(result_path, allow_pickle=True)["predicted_coco"], dtype=np.float64)
        n = args.end - args.start + 1
        if pred.shape != (n, 17, 3):
            raise ValueError(f"{route}: predicted_coco shape {pred.shape} != ({n},17,3)")
        corrected = corrected_metrics(pred, cal.R_cam0_to_cam1, cal.T_cam0_to_cam1_mm,
                                      Rgc, Tgc_mm, foot_w)
        saved = comparison.get(route, {})
        old = {
            "negative_depth_count": saved.get("negative_depth_count"),
            "root_accel_p95_mm": saved.get("root_accel_p95_mm"),
            "coco_accel_p95_mm": saved.get("coco_accel_p95_mm"),
            "foot_contact_proxy_rms": saved.get("foot_contact_proxy_rms"),
        }
        routes[route] = {
            "result_file": str(result_path.resolve()),
            "corrected": corrected,
            "saved_old_metrics": old,
            "differences": {
                "negative_depth_count": (corrected["negative_depth_count"] - old["negative_depth_count"]
                                         if old["negative_depth_count"] is not None else None),
                "root_accel_p95": (corrected["root_accel_p95_mm_per_frame2"] - old["root_accel_p95_mm"]
                                   if old["root_accel_p95_mm"] is not None else None),
                "coco_accel_p95": (corrected["coco_accel_p95_mm_per_frame2"] - old["coco_accel_p95_mm"]
                                   if old["coco_accel_p95_mm"] is not None else None),
            },
        }

    audit = {
        "schema_version": "foot_only_v2_metric_audit_v1",
        "status": "metric_correction_only_not_contact_validation",
        "window": [args.start, args.end],
        "fit_frames": args.end - args.start + 1,
        "routes": routes,
        "left_ankle_proxy_height_m": float(ankle_proxy[0]),
        "right_ankle_proxy_height_m": float(ankle_proxy[1]),
        "fit_loss_target_height": "z_G = 0 for model-side COCO ankles 15/16",
        "interpretation": ("COCO ankles are not the foot sole; the zero-height proxy is an "
                           "engineering regularizer, not physical sole contact. This audit "
                           "corrects metric computation only and does not validate contact."),
        "source_audit": {
            "inputs": {k: str(Path(getattr(args, k)).resolve())
                       for k in ("result_dir", "scene_transforms", "contact_labels", "calibration_dir")},
            "files_actually_read": [
                str((result_dir / "comparison.json").resolve()),
                str((result_dir / "stage_c_no_contact" / "result.npz").resolve()),
                str((result_dir / "stage_d_foot_only" / "result.npz").resolve()),
                str(Path(args.scene_transforms).resolve()),
                str(Path(args.contact_labels).resolve()),
            ],
            "fit_rerun": False,
            "result_modified": False,
        },
    }
    (out / "metric_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps({route: data["corrected"] for route, data in routes.items()},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
