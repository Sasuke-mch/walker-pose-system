"""Isolated, reproducible 360-frame 2D observation ablation of reference25."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_app.smplh_fitting.pipeline import run_fit
from walker_tools.body.fit_smplh_wilor_sequence import build_parser


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--reprojection-weight", type=float, default=.002)
    cli.add_argument("--vposer-weight", type=float, default=.02)
    cli.add_argument("--beta-weight", type=float, default=.001)
    cli.add_argument("--root-weight", type=float, default=.01)
    cli.add_argument("--joint-steps", type=int, default=300)
    cli.add_argument("--hand-steps", type=int, default=100)
    cli.add_argument("--source-args", type=Path,
                     default=Path("research_records/engineering_validation/G20261005_reference_weight_fit_v1/runs/reference25/fit_args.json"))
    opts = cli.parse_args()
    source = opts.source_args
    argv = json.loads(source.read_text(encoding="utf-8"))
    argv[argv.index("--output-dir") + 1] = str(opts.output)
    argv[argv.index("--joint-steps") + 1] = str(opts.joint_steps)
    for flag in ("--hand-steps", "--contact-steps"):
        argv[argv.index(flag) + 1] = str(opts.hand_steps)
    argv += ["--body-observation-mode", "reprojection_only", "--body-3d-weight", "0",
             "--body-reprojection-weight", str(opts.reprojection_weight),
             "--body-reprojection-scale-px", "25", "--vposer-prior-weight", str(opts.vposer_weight),
             "--beta-prior-weight", str(opts.beta_weight), "--root-anchor-weight", str(opts.root_weight)]
    args = build_parser().parse_args(argv)
    if opts.output.exists():
        raise RuntimeError("refuse existing experiment output")
    import numpy as np
    import torch
    np.random.seed(0)
    torch.manual_seed(0)
    code = run_fit(args)
    (opts.output / "fit_args.json").write_text(json.dumps(argv, indent=2), encoding="utf-8")
    (opts.output / "resolved_options.json").write_text(
        json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
