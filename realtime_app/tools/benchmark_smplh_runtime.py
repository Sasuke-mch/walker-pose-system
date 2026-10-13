"""Time the existing CUDA fit and saved-parameter decode, without changing its method."""
from __future__ import annotations

import argparse
import contextlib
import json
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from pose_app.smplh_fitting.pipeline import run_fit
from pose_app.smplh_hand_observation import hand21
from pose_app.smpl_coco_observation import load_coco17_regressor, regress_coco17_torch
from pose_app.smplx_fitting import load_vposer_explicit
from walker_tools.body.fit_smplh_wilor_sequence import build_parser


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def gpu_state():
    command = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,"
               "utilization.gpu,temperature.gpu,power.draw", "--format=csv"]
    return subprocess.check_output(command, text=True, encoding="utf-8").strip()


def latency_stats(wall_ms, cuda_ms, batch):
    return {
        "samples": len(wall_ms), "batch": batch,
        "wall_median_ms": float(np.median(wall_ms)),
        "wall_p95_ms": float(np.percentile(wall_ms, 95)),
        "cuda_median_ms": float(np.median(cuda_ms)),
        "amortized_wall_ms_per_frame": float(np.median(wall_ms) / batch),
        "batch_throughput_frames_per_second": float(1000 * batch / np.median(wall_ms)),
    }


def benchmark_decoder(args, result_path, warmup, samples):
    import smplx
    from smplx.utils import Struct

    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device="cuda")

    with args.smplh_model.open("rb") as stream:
        data = pickle.load(stream, encoding="latin1")
    hands = {}
    for side, path in (("l", args.mano_left), ("r", args.mano_right)):
        with path.open("rb") as stream:
            mano = pickle.load(stream, encoding="latin1")
        data["hands_components" + side] = np.asarray(mano["hands_components"], np.float32)
        data["hands_mean" + side] = np.asarray(mano["hands_mean"], np.float32)
        basis = tensor(mano["hands_components"][:args.hand_pca_comps])
        hands[side] = (basis, tensor(mano["hands_mean"]),
                       torch.linalg.vector_norm(basis, dim=1).clamp_min(1e-6))
    model = smplx.SMPLH(str(args.smplh_model), data_struct=Struct(**data), gender="male",
                        use_pca=False, flat_hand_mean=True, batch_size=1).to("cuda")
    vposer, _, _ = load_vposer_explicit(args.vposer_dir, "cuda")
    reg = load_coco17_regressor(args.regressor)
    with np.load(result_path, allow_pickle=False) as archive:
        saved = {key: tensor(archive[key]) for key in
                 ("betas", "global_orient", "transl", "vposer_latent",
                  "left_hand_pca", "right_hand_pca")}
        n = len(archive["pair_id"])
    results = []
    for batch, copy_to_cpu in ((1, False), (n, False), (n, True)):
        # All parameters already reside on CUDA, as in the fitter's final decode.
        # Shared local hand coefficients are decoded once then expanded.
        def decode():
            pose = vposer.decode(saved["vposer_latent"][:batch])["pose_body"].reshape(batch, 63)
            poses = {}
            for name, side in (("left", "l"), ("right", "r")):
                basis, mean, scale = hands[side]
                coeff = saved[name + "_hand_pca"][:1 if args.shared_hand_pose else batch]
                poses[name] = (mean[None] + (coeff / scale[None]) @ basis).expand(batch, -1)
            out = model(betas=saved["betas"].expand(batch, -1),
                        global_orient=saved["global_orient"][:batch], body_pose=pose,
                        left_hand_pose=poses["left"], right_hand_pose=poses["right"],
                        transl=saved["transl"][:batch], return_verts=True)
            values = (out.vertices, regress_coco17_torch(out.vertices, reg),
                      hand21(out.joints, out.vertices, "left"),
                      hand21(out.joints, out.vertices, "right"))
            return tuple(value.cpu() for value in values) if copy_to_cpu else values

        wall_ms, cuda_ms = [], []
        with torch.no_grad():
            for _ in range(warmup):
                decode()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(samples):
                torch.cuda.synchronize()
                started = time.perf_counter()
                start_event.record()
                output = decode()
                end_event.record()
                torch.cuda.synchronize()
                wall_ms.append((time.perf_counter() - started) * 1000)
                cuda_ms.append(start_event.elapsed_time(end_event))
                del output
        row = latency_stats(wall_ms, cuda_ms, batch)
        row.update(copy_mesh_coco_hands_to_cpu=copy_to_cpu,
                   peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                   wall_samples_ms=wall_ms, cuda_samples_ms=cuda_ms)
        results.append(row)
        print("decode", batch, "CPU copy", copy_to_cpu, "median ms", row["wall_median_ms"], flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-args", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    cli = parser.parse_args()
    if cli.output.exists():
        raise FileExistsError(cli.output)
    if cli.warmup < 1 or cli.samples < 2:
        raise ValueError("positive warmup and at least two samples required")
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA GPU unavailable")
    argv = json.loads(cli.source_args.read_text(encoding="utf-8"))
    argv[argv.index("--output-dir") + 1] = str(cli.output / "fit")
    args = build_parser().parse_args(argv)
    if args.device != "cuda":
        raise ValueError("benchmark requires the CUDA fit configuration")
    cli.output.mkdir(parents=True)
    write(cli.output / "fit_args.json", argv)
    before = gpu_state()
    properties = torch.cuda.get_device_properties(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # Context is initialized before solver timing. Asset loading, input geometry,
    # initialization, optimization, diagnostics and output writes remain included.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    events, stage_rows = {}, []
    active_stage = {}

    def observe(event, info):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if event == "stage_start":
            active_stage.update(info, started=now)
        elif event == "stage_end":
            elapsed = now - active_stage["started"]
            stage_rows.append({**info, "execution": active_stage["execution"],
                               "seconds": elapsed,
                               "mean_ms_per_optimization_step": elapsed * 1000 / info["steps"]
                               if info["steps"] else None})
        else:
            events[event] = now

    print("Starting full current CUDA fit; source", cli.source_args, flush=True)
    try:
        with (cli.output / "fit.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                run_fit(args, runtime_observer=observe)
        peak = torch.cuda.max_memory_allocated() / 2**20
        summary = json.loads((args.output_dir / "fit_summary.json").read_text(encoding="utf-8"))
        total = events["fit_end"] - events["fit_start"]
        report = {
            "status": "fit_completed_decoder_pending",
            "hardware": {"gpu": properties.name, "gpu_total_mib": properties.total_memory / 2**20,
                         "cpu": platform.processor(), "torch": torch.__version__,
                         "cuda_runtime": torch.version.cuda, "python": sys.version,
                         "torch_cpu_threads": torch.get_num_threads(), "nvidia_smi_before": before},
            "source_arguments": str(cli.source_args.resolve()), "seed": 0,
            "frames": summary["frames"], "optimization_steps": summary["steps"],
            "fit_policy": summary["fit_policy"], "warmup": cli.warmup, "samples": cli.samples,
            "full_fit": {"measurements": 1, "solver_wall_seconds": total,
                         "preparation_seconds": events["optimization_start"] - events["fit_start"],
                         "optimization_seconds": events["optimization_end"] - events["optimization_start"],
                         "final_decode_seconds": events["final_decode_end"] - events["optimization_end"],
                         "transfer_and_output_seconds": events["fit_end"] - events["final_decode_end"],
                         "effective_frames_per_second": summary["frames"] / total,
                         "amortized_ms_per_frame": total * 1000 / summary["frames"],
                         "peak_torch_allocated_mib": peak,
                         "peak_torch_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                         "stages": stage_rows},
            "scope": "One full offline sequence fit, then warmed float32/no_grad decode; CUDA synchronized at boundaries. Solver timing excludes interpreter/module import and CUDA context startup; includes inputs/assets/initialization/560-step solve/logs/NPZ writes. Decoder excludes observation fitting, detectors, scene replay, visualization and disk serialization. Batch throughput is not causal realtime latency.",
        }
        write(cli.output / "runtime.json", report)
        print("Full fit wall seconds", total, flush=True)
        report["decoder"] = benchmark_decoder(args, args.output_dir / "result.npz", cli.warmup, cli.samples)
        report["hardware"]["nvidia_smi_after"] = gpu_state()
        report["status"] = "completed"
        write(cli.output / "runtime.json", report)
        print("Runtime measurements saved", cli.output / "runtime.json", flush=True)
    except Exception as error:
        write(cli.output / "failure.json", {"error": repr(error), "outputs_preserved": True})
        raise


if __name__ == "__main__":
    main()
