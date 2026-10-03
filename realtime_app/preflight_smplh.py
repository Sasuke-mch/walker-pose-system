"""Validate explicit offline-fit resources; missing assets are an error."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import sys

from walker_tools.body.fit_smplh_wilor_sequence import build_parser
from pose_app.smplh_fitting.stages import validate_optimization_options


def fit_arguments(path: Path):
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ValueError("args JSON must be an array of CLI argument strings")
    args = build_parser().parse_args(values)
    validate_optimization_options(args)
    return args, values


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--args-json", type=Path, required=True)
    parser.add_argument("--load-models", action="store_true",
                        help="also load licensed assets and test VPoser decoding")
    options = parser.parse_args(argv)
    args, values = fit_arguments(options.args_json)
    missing = []
    for package in ("numpy", "cv2", "scipy", "torch", "smplx", "human_body_prior", "omegaconf", "chumpy"):
        if importlib.util.find_spec(package) is None:
            missing.append(f"package:{package}")
    for name in ("left_raw", "right_raw", "wilor_left", "wilor_right", "regressor",
                 "smplh_model", "mano_left", "mano_right"):
        path = getattr(args, name)
        if not path.is_file():
            missing.append(f"{name}:{path}")
    for name in ("cam0_fisheye.json", "cam1_fisheye.json", "stereo_fisheye.json"):
        path = args.calibration_dir / name
        if not path.is_file():
            missing.append(f"calibration:{path}")
    if args.vposer_dir is None or len(list(args.vposer_dir.glob("*.yaml"))) != 1 or not list((args.vposer_dir / "snapshots").glob("*.ckpt")):
        missing.append(f"VPoser:{args.vposer_dir}")
    if (args.mano_pose_init or args.mano_pose_weight > 0) and not args.canonical_mano_right.is_file():
        missing.append(f"canonical_mano_right:{args.canonical_mano_right}")
    for name in ("wrist_reference", "contact_labels", "scene_transforms", "contact_vertex_sets",
                 "walker_topology", "global_hand_handle_pose"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            missing.append(f"{name}:{path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        missing.append(f"output_not_empty:{args.output_dir}")
    report = dict(status="blocked" if missing else "paths_ready", missing=missing,
                  model_load_verified=False, device=args.device,
                  source="explicit_args_json", physical_accuracy_verified=False)
    if not missing and options.load_models:
        # The fitting module applies the same legacy-pickle compatibility patch.
        from pose_app.smplh_fitting import pipeline
        import torch
        from pose_app.fisheye_camera import load_stereo_fisheye
        from pose_app.smpl_coco_observation import load_coco17_regressor
        from pose_app.wilor_mano_prior import audit_assets
        load_stereo_fisheye(args.calibration_dir)
        load_coco17_regressor(args.regressor)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if args.mano_pose_init or args.mano_pose_weight > 0:
            assets = []
            for path in (args.smplh_model, args.mano_left, args.mano_right, args.canonical_mano_right):
                with path.open("rb") as stream:
                    assets.append(pickle.load(stream, encoding="latin1"))
            audit_assets(*assets)
        model, _, _ = pipeline.load_vposer_explicit(args.vposer_dir, args.device)
        with torch.no_grad():
            pose = model.decode(torch.zeros(1, 32, device=args.device))["pose_body"]
        if pose.numel() != 63 or not bool(torch.isfinite(pose).all()):
            raise RuntimeError("invalid VPoser output")
        report.update(status="model_load_ready", model_load_verified=True)
    report["fit_command"] = [sys.executable, "-B", "realtime_app/tools/fit_smplh_wilor_sequence.py", *values]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
