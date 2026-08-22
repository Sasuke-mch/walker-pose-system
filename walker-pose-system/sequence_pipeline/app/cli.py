from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .menu import interactive_request
from .orchestrator import parse_models, run_pipeline
from .preflight import run_preflight
from .utils import parse_key_value


def _pipeline_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-model pose estimation for image sequences.")
    p.add_argument("--config", default=None, help="Path to configs/local.json")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("list", help="List configured models")

    c = sub.add_parser("check", help="Check paths, weights, Docker daemon and images")
    c.add_argument("--models", default="all")
    c.add_argument("--weight", action="append", default=[])
    c.add_argument("--no-docker", action="store_true")
    c.add_argument("--input-dir", default=None)

    r = sub.add_parser("run", help="Run selected models")
    r.add_argument("--input-dir", required=True)
    r.add_argument("--output-dir", required=True)
    r.add_argument("--models", default="all")
    r.add_argument("--weight", action="append", default=[], help="MODEL=HOST_WEIGHT_PATH; may be repeated")
    r.add_argument("--run-name", default=None)
    r.add_argument("--device", default=None)
    r.add_argument("--imgsz", type=int, default=None)
    r.add_argument("--det-conf", type=float, default=None)
    r.add_argument("--pose-conf", type=float, default=None)
    r.add_argument("--det-iou", type=float, default=None)
    r.add_argument("--save-vis", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--skip-existing", action="store_true")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--continue-on-error", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    root = _pipeline_root()
    args = parser().parse_args(argv)
    local_path = Path(args.config).resolve() if args.config else None
    config = load_config(root, local_path)

    if args.command is None:
        req = interactive_request(config)
        selected = parse_models(config, req["models"])
        return run_pipeline(
            config=config,
            input_dir=req["input_dir"].resolve(),
            output_dir=req["output_dir"].resolve(),
            selected=selected,
            weight_overrides={},
            options={"save_vis": req["save_vis"]},
        )

    if args.command == "list":
        print(f"Project root: {config.project_root}\n")
        for key in config.model_order:
            model = config.model(key)
            print(f"{key:16s} {model.get('display_name', key):22s} image={model.get('docker_image')}")
        return 0

    overrides = parse_key_value(getattr(args, "weight", []))
    if args.command == "check":
        selected = parse_models(config, args.models)
        report = run_preflight(config, selected, overrides, check_docker=not args.no_docker, input_dir=Path(args.input_dir).resolve() if args.input_dir else None)
        report.print()
        return 2 if report.errors else 0

    if args.command == "run":
        selected = parse_models(config, args.models)
        options = vars(args).copy()
        return run_pipeline(
            config=config,
            input_dir=Path(args.input_dir).resolve(),
            output_dir=Path(args.output_dir).resolve(),
            selected=selected,
            weight_overrides=overrides,
            options=options,
        )
    return 0
