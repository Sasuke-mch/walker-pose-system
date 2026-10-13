"""Audit saved fits, compare frozen archives, or run a seeded five-step regression.

Run from the repository root. Resource failures exit 2; mismatches exit 1.
Reports are always written to a new directory; existing goldens are read-only.
"""
from pathlib import Path
import argparse
import json
import platform
import sys
import traceback

from pose_app.mainline_regression import (
    audit_contract, compare_archives, load_archive, observation_snapshot,
    read_json, replay_parameters, ResourceUnavailableError,
)


def fit_arguments(path):
    from walker_tools.body.fit_smplh_wilor_sequence import build_parser

    values = read_json(path)
    if not isinstance(values, list):
        raise ValueError("command JSON must be an argv list")
    if "--left-raw" not in values:
        raise ValueError("command has no --left-raw")
    # Accept both historical fit_command.json and an options-only argv list.
    values = values[next(i for i, value in enumerate(values) if value.startswith("--")):]
    if "--fit-policy" not in values:
        values.extend(["--fit-policy", "legacy"])
    if any(value.startswith("--") and values.count(value) != 1 for value in values):
        raise ValueError("duplicate options in command")
    try:
        return build_parser().parse_args(values)
    except SystemExit as error:
        raise ValueError("invalid saved fit command") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "compare", "short"))
    parser.add_argument("--command", type=Path, help="saved fitter argv JSON; required by audit/short")
    parser.add_argument("--fit", type=Path, help="directory containing result.npz")
    parser.add_argument("--reference", type=Path, help="read-only short-budget baseline fit directory")
    parser.add_argument("--observation-reference", type=Path, help="frozen observations.npz including rejection reasons")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--route", type=Path, default=Path("realtime_app/configs/current_mainline.json"))
    cli = parser.parse_args(argv)
    if cli.output.exists():
        parser.error("output already exists; use a new directory")
    if cli.chunk_size < 1:
        parser.error("chunk-size must be positive")
    if cli.mode in ("audit", "short") and not cli.command:
        parser.error("--command is required")
    if cli.mode in ("audit", "compare") and not cli.fit:
        parser.error("--fit is required")
    if cli.mode in ("short", "compare") and not cli.reference:
        parser.error("--reference is required; this command never records a golden automatically")
    cli.output.mkdir(parents=True)
    report = {"mode": cli.mode, "evidence": "implementation_consistency_only",
              "environment": {"python": sys.version, "platform": platform.platform()},
              "tolerances": {"archive_parity": "exact; shape/dtype/NaN locations included",
                             "replay": "2e-6 m for geometry; 2e-6 rad for decoded axis-angle; faces exact"},
              "scope": "offline soft-weight body candidate; not strict live stereo certification",
              "inputs": {key: str(value) if value is not None else None
                         for key, value in vars(cli).items()}, "checks": {}}
    try:
        args = fit_arguments(cli.command) if cli.command else None
        if args is not None:
            import torch
            if args.device.startswith("cuda") and not torch.cuda.is_available():
                raise ResourceUnavailableError("saved command requires CUDA, but CUDA is unavailable")
            report["environment"].update(torch=torch.__version__, cuda=torch.version.cuda,
                                         device=args.device)
            frozen = read_json(cli.route)
            budget_names = {"base_steps", "beta_steps", "joint_steps", "hand_steps", "contact_steps",
                            "contact_refine_steps"}
            report["checks"]["frozen_options"] = {
                key: {"passed": getattr(args, key) == value}
                for key, value in frozen["frozen_scalar_options"].items()
                if cli.mode != "short" or key not in budget_names
            }
        fit = cli.fit
        if cli.mode == "short":
            import numpy as np
            import torch
            from pose_app.smplh_fitting.pipeline import run_fit

            np.random.seed(0)
            torch.manual_seed(0)
            args.output_dir = cli.output / "fit"
            for name in ("base_steps", "beta_steps", "joint_steps", "hand_steps", "contact_steps"):
                setattr(args, name, 1)
            args.contact_refine_steps = 0
            # Archive exactly what was consumed, including all resolved asset paths.
            report["resolved_fit_options"] = {key: str(value) if isinstance(value, Path) else value
                                               for key, value in vars(args).items()}
            run_fit(args)
            fit = args.output_dir
        result = load_archive(fit / "result.npz")
        initial = load_archive(fit / "cold_initialization.npz")
        if args is not None:
            import numpy as np
            snapshot = observation_snapshot(args)
            np.savez_compressed(cli.output / "observations.npz", **snapshot)
            summary = read_json(fit / "fit_summary.json")
            report["checks"]["contract"] = audit_contract(result, initial, summary, snapshot, args)
            report["checks"]["parameter_replay"] = replay_parameters(result, args, cli.chunk_size)
            report["frames"] = len(result["pair_id"])
            report["checks"]["contract"]["frozen_frame_range"] = {
                "passed": np.array_equal(result["pair_id"], np.arange(frozen["frames"], dtype=np.int64))}
        if cli.reference:
            report["checks"]["result_parity"] = compare_archives(
                load_archive(cli.reference / "result.npz"), result)
            report["checks"]["initialization_parity"] = compare_archives(
                load_archive(cli.reference / "cold_initialization.npz"), initial)
        if args is not None:
            previous_snapshot = cli.observation_reference
            if previous_snapshot is None and cli.reference:
                previous_snapshot = cli.reference / "observations.npz"
            if previous_snapshot is not None:
                report["checks"]["observation_parity"] = compare_archives(
                    load_archive(previous_snapshot), snapshot)
            else:
                report["observation_baseline"] = "not_requested; audit is not a historical rejection-reason parity claim"
        else:
            report["checks"]["candidate_finiteness"] = {
                key: {"passed": bool(__import__("numpy").isfinite(result[key]).all())}
                for key in ("vertices", "betas", "body_pose", "vposer_latent")
            }
        passed = all(check["passed"] for group in report["checks"].values() for check in group.values())
        report["status"] = "passed" if passed else "mismatch"
        code = 0 if passed else 1
    except (FileNotFoundError, ModuleNotFoundError, ResourceUnavailableError) as error:
        report.update(status="unavailable", error=str(error), traceback=traceback.format_exc())
        code = 2
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        code = 1
    (cli.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(cli.output / "report.json")}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
