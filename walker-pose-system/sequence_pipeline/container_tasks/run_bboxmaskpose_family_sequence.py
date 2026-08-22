#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        required=True,
        choices=["pmpose", "bboxmaskpose"],
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--det-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--vis-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--pose-variant",
        default="PMPose-b",
    )
    parser.add_argument(
        "--bmp-config",
        default="bmp_v2",
    )
    parser.add_argument(
        "--bmp-iters",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--mask-mode",
        default="bbox",
    )
    parser.add_argument(
        "--save-vis",
        default="false",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.20,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    benchmark_script = Path(
        "/workspace/BBoxMaskPose/tools/"
        "benchmark_from_yolo26_bboxes.py"
    )

    if not benchmark_script.is_file():
        raise FileNotFoundError(benchmark_script)

    command = [
        sys.executable,
        str(benchmark_script),
        "--det-json",
        args.det_json,
        "--method",
        args.method,
        "--output-csv",
        args.output_csv,
        "--summary-csv",
        args.summary_csv,
        "--save-pred-json",
        args.pred_json,
        "--device",
        args.device,
        "--pose-variant",
        args.pose_variant,
        "--warmup",
        "0",
        "--repeat",
        "1",
    ]

    if args.method == "bboxmaskpose":
        command.extend(
            [
                "--bmp-config",
                args.bmp_config,
                "--bmp-iters",
                str(args.bmp_iters),
                "--mask-mode",
                args.mask_mode,
            ]
        )

    print("Running inference:")
    print(" ".join(command))

    subprocess.run(command, check=True)

    pred_path = Path(args.pred_json)

    if not pred_path.is_file():
        raise FileNotFoundError(
            f"Prediction JSON was not created: {pred_path}"
        )

    if not parse_bool(args.save_vis):
        print("Visualization disabled.")
        return

    visualizer = Path(
        "/workspace/pipeline/container_tasks/"
        "visualize_bboxmaskpose_family.py"
    )

    vis_command = [
        sys.executable,
        str(visualizer),
        "--input-dir",
        args.input_dir,
        "--pred-json",
        args.pred_json,
        "--vis-dir",
        args.vis_dir,
        "--score-thr",
        str(args.score_thr),
    ]

    print("Running visualization:")
    print(" ".join(vis_command))

    subprocess.run(vis_command, check=True)


if __name__ == "__main__":
    main()
