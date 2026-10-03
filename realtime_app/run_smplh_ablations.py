"""Full-sequence one-factor ablations using the existing SMPL-H solver.

Each child runs in a fresh seeded process. Resume retains failures and completed
outputs; it never changes a case or replaces its result. Launch from repo root.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def modified_options(values, changes, output):
    values = list(values[values.index("--left-raw"):])
    changes = {**changes, "output_dir": str(output)}
    for name, value in changes.items():
        flag = "--" + name.replace("_", "-")
        if flag in values:
            index = values.index(flag)
            end = index + 1
            if end < len(values) and not values[end].startswith("--"):
                end += 1
            del values[index:end]
        if value is True:
            values.append(flag)
        elif value is not None and value is not False:
            values.extend([flag, str(value)])
    return values


def validate_plan(plan):
    """Reject malformed protocols before running expensive model code."""
    cases = plan["cases"]
    ids = [row["id"] for row in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs")
    budgets = {"steps", "base_steps", "beta_steps", "joint_steps", "hand_steps",
               "contact_steps", "contact_refine_steps"}
    for row in cases:
        if Path(row["id"]).name != row["id"] or row["id"] in (".", ".."):
            raise ValueError("case ID must be one directory name")
        if row["detector"] not in plan["source_commands"]:
            raise ValueError("case detector has no source command")
        if budgets & set(row["changes"]):
            raise ValueError("ablation protocol must retain the same optimization budgets")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("realtime_app/configs/smplh_ablation_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case", help="internal single-case execution")
    cli = parser.parse_args()
    plan = read(cli.plan)
    validate_plan(plan)
    if cli.case and cli.case not in {row["id"] for row in plan["cases"]}:
        parser.error("unknown case ID")
    if cli.case:
        import numpy as np
        import torch
        from walker_tools.body.fit_smplh_wilor_sequence import build_parser
        from pose_app.smplh_fitting.pipeline import run_fit

        case = next(row for row in plan["cases"] if row["id"] == cli.case)
        directory = cli.output / case["id"]
        if directory.exists():
            raise FileExistsError(directory)
        directory.mkdir(parents=True)
        values = modified_options(read(plan["source_commands"][case["detector"]]),
                                  case["changes"], directory / "fit")
        args = build_parser().parse_args(values)
        write(directory / "fit_args.json", values)
        write(directory / "resolved_options.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        started = time.perf_counter()
        try:
            np.random.seed(plan["seed"])
            torch.manual_seed(plan["seed"])
            def observe(stage, step, arrays):
                np.savez_compressed(directory / f"checkpoint_{stage}_{step}.npz", **arrays)
            run_fit(args, observer=observe)
            write(directory / "execution.json", {"status": "completed", "seconds": time.perf_counter() - started,
                  "seed": plan["seed"], "case": case, "torch": torch.__version__, "cuda": torch.version.cuda,
                  "cuda_device": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                  "includes": "input audits, fitting, checkpoint serialization, output serialization; excludes Python imports and evaluation"})
            return 0
        except Exception as error:
            write(directory / "FAILURE.json", {"status": "failed", "seconds": time.perf_counter() - started,
                  "case": case, "error": str(error), "traceback": traceback.format_exc()})
            return 1
    if cli.output.exists() and not cli.resume:
        parser.error("output exists; use --resume to retain completed runs and failures")
    cli.output.mkdir(parents=True, exist_ok=True)
    protocol = cli.output / "protocol.json"
    if protocol.exists() and read(protocol) != plan:
        raise ValueError("resume plan differs from recorded protocol")
    if not protocol.exists():
        write(protocol, plan)
    for case in plan["cases"]:
        directory = cli.output / case["id"]
        if directory.exists():
            if not (directory / "execution.json").exists() and not (directory / "FAILURE.json").exists():
                write(directory / "FAILURE.json", {"status": "interrupted", "case": case})
            print(json.dumps({"retained": case["id"]}), flush=True)
            continue
        print(json.dumps({"starting": case["id"]}), flush=True)
        with (cli.output / f"{case['id']}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-B", __file__, "--plan", str(cli.plan),
                    "--output", str(cli.output), "--case", case["id"]], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode and not (directory / "FAILURE.json").exists():
            directory.mkdir(exist_ok=True)
            write(directory / "FAILURE.json", {"status": "process_failed", "returncode": result.returncode, "case": case})
        print(json.dumps({"finished": case["id"], "exit": result.returncode}), flush=True)
    failures = [row["id"] for row in plan["cases"] if (cli.output / row["id"] / "FAILURE.json").exists()]
    write(cli.output / "execution_summary.json", {"completed": len(plan["cases"]) - len(failures), "failed": failures})
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
