#!/usr/bin/env python3
"""Fit PCA/VPoser SMPL-H from body stereo and optional native WiLoR pose.

This is an engineering candidate run.  WiLoR pixels are model-derived MANO
projections, so they are masked and reported as an auxiliary 2-D term rather
than independent ground truth.
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
from pathlib import Path

from pose_app.smplh_hand_observation import read_wilor as read_wilor

ROOT = _tool_project_root


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left-raw", type=Path, required=True)
    ap.add_argument("--right-raw", type=Path, required=True)
    ap.add_argument("--wilor-left", type=Path, required=True,
                    help="WiLoR JSONL from left camera; both left/right records are read")
    ap.add_argument("--wilor-right", type=Path, required=True,
                    help="WiLoR JSONL from right camera; both left/right records are read")
    ap.add_argument("--calibration-dir", type=Path, required=True)
    ap.add_argument("--regressor", type=Path, required=True)
    ap.add_argument("--smplh-model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--base-steps", type=int, default=None,
                    help="Stage A body/VPoser steps; defaults to steps//6")
    ap.add_argument("--beta-steps", type=int, default=None,
                    help="Stage B shared-beta steps; defaults to steps//6")
    ap.add_argument("--joint-steps", type=int, default=None,
                    help="Stage C VPoser/body refinement steps; defaults to steps//6")
    ap.add_argument("--hand-steps", type=int, default=None,
                    help="Stage D1 proximal-weighted hand steps; defaults to steps//6")
    ap.add_argument("--contact-steps", type=int, default=None,
                    help="Stage D2 hand-contact steps; defaults to steps//6")
    ap.add_argument("--contact-refine-steps", type=int, default=None,
                    help="Stage D3 limited contact refinement; defaults to steps//6")
    ap.add_argument("--vposer-dir", type=Path, default=None)
    ap.add_argument("--vposer-prior-weight", type=float, default=0.02)
    ap.add_argument("--root-anchor-weight", type=float, default=0.01,
                    help="Stage C/D3 penalty for drifting from the triangulation-initialized root")
    ap.add_argument("--max-init-body-rms-mm", type=float, default=300.0,
                    help="stop before optimization when rigid initialization is inconsistent")
    ap.add_argument("--hand-pca-comps", type=int, default=12,
                    help="number of real MANO PCA hand-pose coefficients per hand")
    ap.add_argument("--hand-pca-profile", choices=("pca12", "pca24", "full45"), default=None,
                    help="explicit ablation label; must agree with --hand-pca-comps")
    ap.add_argument("--hand-pca-prior-weight", type=float, default=1e-3,
                    help="quadratic prior on normalized MANO PCA coefficients")
    ap.add_argument("--hand-temporal-weight", type=float, default=2e-2,
                    help="second-order temporal prior on MANO PCA coefficients")
    ap.add_argument("--mano-pose-weight", type=float, default=0.0,
                    help="native WiLoR local rotation soft prior (SO3 chordal); enabled only in D1/D2/D3")
    ap.add_argument("--mano-pose-init", action="store_true",
                    help="initialize hand PCA at the start of D1 from accepted native MANO poses")
    ap.add_argument("--shared-hand-pose", action="store_true",
                    help="one PCA vector per anatomical hand shared by all frames; local articulation only")
    ap.add_argument("--hand-2d-weight", type=float, default=1e-7,
                    help="WiLoR model-derived 2D auxiliary weight; set 0 for native-MANO-only hand information")
    ap.add_argument("--body-temporal-weight", type=float, default=0.0,
                    help="model-COCO relative-to-pelvis second-order prior; 0 preserves the audited baseline")
    ap.add_argument("--body-reprojection-weight", type=float, default=0.0,
                    help="dual-fisheye body 2-D reprojection term; 0 preserves the audited baseline")
    ap.add_argument("--body-reprojection-scale-px", type=float, default=100.0,
                    help="pixel scale used to make the body reprojection Huber term dimensionless")
    ap.add_argument("--wrist-reference", type=Path, default=None)
    ap.add_argument("--wrist-reference-weight", type=float, default=0.0,
                    help="independent manual wrist mean squared metre residual in body stages A/B/C")
    ap.add_argument("--allow-diagnostic-wrist-reference", action="store_true")
    ap.add_argument("--mano-left", type=Path, default=ROOT / "third_party/WiLoR/mano_data/models/MANO_LEFT.pkl")
    ap.add_argument("--mano-right", type=Path, default=ROOT / "third_party/WiLoR/mano_data/models/MANO_RIGHT.pkl")
    ap.add_argument("--canonical-mano-right", type=Path,
                    default=ROOT / "third_party/WiLoR/mano_data/MANO_RIGHT.pkl",
                    help="canonical WiLoR right MANO asset used by native-pose audit")
    ap.add_argument("--bone-weight", type=float, default=0.0,
                    help="optional beta-zero bone-length prior; default off because it biases shared beta")
    ap.add_argument("--contact-labels", type=Path, default=None)
    ap.add_argument("--scene-transforms", type=Path, default=None)
    ap.add_argument("--contact-vertex-sets", type=Path, default=None)
    ap.add_argument("--walker-topology", type=Path, default=None,
                    help="validated walker topology; used to check handle semantics")
    ap.add_argument("--surface-hand-contact-weight", type=float, default=0.0)
    ap.add_argument("--surface-foot-contact-weight", type=float, default=0.0)
    ap.add_argument("--global-hand-handle-pose", type=Path, default=None)
    ap.add_argument("--global-hand-handle-weight", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    from pose_app.smplh_fitting.pipeline import run_fit
    return run_fit(args)


if __name__ == "__main__":
    raise SystemExit(main())
