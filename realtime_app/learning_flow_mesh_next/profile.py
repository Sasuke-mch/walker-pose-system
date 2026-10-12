"""End-to-end typed-pixel G0/J/G1/decode/decision timing, explicitly bounded scope."""
import argparse
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .body import SurfaceDecoder
from .information_risk import RiskPolicy, decide_information
from .joint_flow import JointFlow
from .pipeline import TwoStageInference
from .replay import fixture_normalization
from .tangent import TangentScale
from .tests.test_aggregate import rig
from .tests.test_firewall import observation


class CountedSurface:
    def __init__(self, surface):
        self.surface, self.calls, self.states = surface, 0, 0

    def __call__(self, rotations, translation, beta):
        self.calls += 1
        self.states += len(beta)
        return self.surface(rotations, translation, beta)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def run(*, repeats=5, warmup=1, device="cpu", matched_controls=False):
    from realtime_app.learning_flow_mesh.body import BodyDecoder
    from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor

    from .geometry_observation import NoiseModel, project

    if repeats < 2 or warmup < 0:
        raise ValueError("at least two timing repeats and nonnegative warmup required")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    g0, g1 = JointFlow().to(device).eval(), JointFlow().to(device).eval()
    config = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    surface = CountedSurface(SurfaceDecoder(BodyDecoder(**config["assets"]).to(device),
                                            torch.tensor(load_coco17_regressor(Path("models/smpl/J_regressor_coco.npy")),
                                                         device=device, dtype=torch.float32)))
    cameras, noise = rig(), NoiseModel.proxy()
    normal = fixture_normalization(g0.topology, device=device)
    scale = TangentScale.fixture(g0.topology)
    pipeline = TwoStageInference(g0, g1, normal, scale, surface, cameras, noise)
    with torch.no_grad():
        points = surface(torch.eye(3, device=device).expand(1, 52, 3, 3),
                         torch.tensor([[0., 0., 3.]], device=device), torch.zeros(1, 10, device=device))["coco17"][0]
    raw = torch.stack([project(points, c)[0] for c in cameras]).detach()
    records = []
    surface.calls = surface.states = 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for repeat in range(repeats+warmup):
        sync(device)
        start = time.perf_counter()
        fixture = observation()
        obs = replace(fixture, **{name: getattr(fixture, name).to(device) for name in
                                  ("confidence", "present", "quality", "geometry", "views", "origins", "directions")},
                      xy=raw.clone()).validate()
        sync(device)
        preprocessing = (time.perf_counter()-start)*1000
        before_calls, before_states = surface.calls, surface.states
        with torch.no_grad():
            generated, _, information, _, resources = pipeline(obs, jacobian_step=1e-3)
            sync(device)
            decode_at = time.perf_counter()
            legal = generated.numerical
            surface(generated.rotations[legal], generated.translation[legal], generated.beta[legal])
            sync(device)
            decode_ms = (time.perf_counter()-decode_at)*1000
            aggregate_at = time.perf_counter()
            decision, _ = decide_information(generated, information, scale,
                                              RiskPolicy(lambda_g=.1), observation=obs,
                                              cameras=cameras, noise=noise, surface=surface)
            sync(device)
            aggregate_ms = (time.perf_counter()-aggregate_at)*1000
        full = (time.perf_counter()-start)*1000
        if repeat >= warmup:
            records.append({"preprocessing_ms": preprocessing, **resources,
                            "decode_ms": decode_ms, "aggregate_ms": aggregate_ms, "full_ms": full,
                            "all_decoder_batch_calls": surface.calls-before_calls,
                            "all_decoder_state_evaluations": surface.states-before_states,
                            "selected_id": decision.selected_id, "fallback": decision.diagnostic.get("fallback", False)})
    summary = {key: {"p50": float(np.quantile([r[key] for r in records], .5)),
                     "p95": float(np.quantile([r[key] for r in records], .95))}
               for key in ("preprocessing_ms", "G0_ms", "J_ms", "G1_ms", "decode_ms", "aggregate_ms", "full_ms")}
    report = {"scope": "random-float32 typed-pixel two-generator real-surface engineering timing",
            "device": device, "hardware": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
            "torch": torch.__version__, "repeats": repeats, "warmup": warmup,
            "fixed_budget": {"G0_K": 32, "G1_K": 32, "Euler_steps": 8, "TTA": False},
            "jacobian": "central difference 1e-3 for float32 timing; float64 accuracy audit separate",
            "summary": summary, "records": records,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "capture_and_2D_detector_included": False, "timing_gate_33_3ms": summary["full_ms"]["p95"] <= 33.3,
            "full_system_30fps_claim_supported": False,
            "training_executed": False, "checkpoint_loaded": False,
            "boundary": "small timing sample and recorded-pixel boundary; not trained efficacy or production latency certification"}
    if matched_controls:
        from .aggregate import decide
        from .candidate_store import infer

        def control(steps):
            sync(device)
            start_control = time.perf_counter()
            with torch.no_grad():
                candidates, _ = infer(g0, normal, obs, k=64, steps=steps)
                sync(device)
                generation_ms = (time.perf_counter()-start_control)*1000
                valid = candidates.numerical
                surface(candidates.rotations[valid], candidates.translation[valid], candidates.beta[valid])
                decide(candidates, "A5", scale, observation=obs, cameras=cameras,
                       noise=noise, surface=surface, lambda_g=.1)
            sync(device)
            return {"full_ms": (time.perf_counter()-start_control)*1000,
                    "generation_ms": generation_ms, "K": 64, "Euler_steps": steps,
                    "network_calls": steps, "candidate_state_evaluations": steps*64}
        equal_samples = [control(8) for _ in range(repeats)]
        target = summary["full_ms"]["p50"]
        pilot = equal_samples[-1]
        fixed_cost = pilot["full_ms"]-pilot["generation_ms"]
        steps = max(8, min(512, round(8*max(0, target-fixed_cost)/max(1e-3, pilot["generation_ms"]))))
        calibration = []
        # Runtime-only pilot, no labels/quality tuning; bounded two adjustments.
        for _ in range(2):
            sample = control(steps)
            calibration.append(sample)
            if abs(sample["full_ms"]-target)/target <= .1:
                break
            fixed_cost = sample["full_ms"]-sample["generation_ms"]
            steps = max(8, min(512, round(steps*max(0, target-fixed_cost)/max(1e-3, sample["generation_ms"]))))
        walltime = [control(steps) for _ in range(repeats)]
        median = float(np.median([value["full_ms"] for value in walltime]))
        report["matched_controls"] = {"equal_total_samples_G0_K64_Euler8": equal_samples,
                                      "walltime_G0_fixed_K64_NFE_calibrated_without_labels": walltime,
                                      "runtime_only_calibration": calibration,
                                      "target_p50_ms": target, "walltime_control_p50_ms": median,
                                      "within_10_percent_p50": abs(median-target)/target <= .1,
                                      "comparison_scope": "same recorded-pixel input, explicit decode and A5 decision; time profile only, no efficacy"}
        report["peak_cuda_allocated_bytes_with_controls"] = torch.cuda.max_memory_allocated() if device == "cuda" else None
    return report


def run_representation_controls(*, repeats=8, warmup=2, device="cuda"):
    """Same fixed-input single-model cost comparison; never tunes model budget."""
    from realtime_app.learning_flow_mesh.body import BodyDecoder
    from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor

    from .aggregate import decide
    from .candidate_store import infer
    from .geometry_observation import project
    from .so3_flow import SO3JointFlow

    if repeats < 2 or warmup < 0:
        raise ValueError("at least two recorded repetitions required")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    ambient = JointFlow().to(device).eval()
    normal = fixture_normalization(ambient.topology, device=device)
    manifold = SO3JointFlow(normal).to(device).eval()
    common = {key: value for key, value in ambient.state_dict().items() if not key.startswith("pose_out.")}
    manifold.load_state_dict(common, strict=False)
    models = {"ambient": ambient, "SO3": manifold}
    config = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    surface = CountedSurface(SurfaceDecoder(BodyDecoder(**config["assets"]).to(device),
                                            torch.tensor(load_coco17_regressor(Path("models/smpl/J_regressor_coco.npy")),
                                                         device=device, dtype=torch.float32)))
    cameras, scale = rig(), TangentScale.fixture(ambient.topology)
    with torch.no_grad():
        points = surface(torch.eye(3, device=device).expand(1, 52, 3, 3),
                         torch.tensor([[0., 0., 3.]], device=device), torch.zeros(1, 10, device=device))["coco17"][0]
        raw = torch.stack([project(points, camera)[0] for camera in cameras]).detach()
    fixture = observation()
    obs = replace(fixture, xy=raw, **{name: getattr(fixture, name).to(device) for name in
                                    ("confidence", "present", "quality", "geometry", "views", "origins", "directions")}).validate()
    modes = [("ambient", False), ("SO3", False), ("ambient", True), ("SO3", True)]

    def control(mode, repetition):
        representation, regression = mode
        model = models[representation]
        k, steps = (1, 1) if regression else (32, 8)
        surface.calls = surface.states = 0
        sync(device)
        baseline = torch.cuda.memory_allocated() if device == "cuda" else None
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.no_grad():
            candidates, _ = infer(model, normal, obs, state=None, k=k, steps=steps, regression=regression)
            sync(device)
            generated_at = time.perf_counter()
            legal = candidates.numerical
            decoded_finite = False
            if legal.any():
                decoded = surface(candidates.rotations[legal], candidates.translation[legal], candidates.beta[legal])
                decoded_finite = bool(torch.isfinite(decoded["vertices"]).all())
            sync(device)
            decoded_at = time.perf_counter()
            decision = decide(candidates, "A4", scale)
            sync(device)
            finished = time.perf_counter()
        peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
        return {"mode": representation+("_regression" if regression else "_F0"), "repeat": repetition,
                "generation_outer_ms": (generated_at-start)*1000,
                "generation_internal_ms": candidates.resources["total_ms"],
                "decode_ms": (decoded_at-generated_at)*1000, "decision_ms": (finished-decoded_at)*1000,
                "full_ms": (finished-start)*1000, "network_calls": candidates.resources["network_calls"],
                "candidate_state_evaluations": candidates.resources["candidate_state_evaluations"],
                "decoder_batch_calls": surface.calls, "decoder_states": surface.states,
                "numeric_candidates": int(legal.sum()), "action_candidates": int(candidates.actions.sum()),
                "decoded_vertices_finite": decoded_finite, "rejection_reasons": candidates.reasons,
                "selected_id": decision.selected_id, "solver": candidates.resources["solver"],
                "cuda_baseline_bytes": baseline, "cuda_peak_bytes": peak,
                "cuda_peak_increment_bytes": peak-baseline if peak is not None else None}

    for _ in range(warmup):
        for mode in modes:
            control(mode, -1)
    records = []
    for repetition in range(repeats):
        order = modes[repetition % len(modes):]+modes[:repetition % len(modes)]
        records.extend(control(mode, repetition) for mode in order)
    summary = {}
    for name in ("ambient_F0", "SO3_F0", "ambient_regression", "SO3_regression"):
        rows = [row for row in records if row["mode"] == name]
        metrics = ("generation_outer_ms", "generation_internal_ms", "decode_ms", "decision_ms", "full_ms")
        summary[name] = {key: {"p50": float(np.quantile([row[key] for row in rows], .5)),
                               "p95": float(np.quantile([row[key] for row in rows], .95))} for key in metrics}
        summary[name].update(numeric_candidates=[row["numeric_candidates"] for row in rows],
                             decoded_vertices_finite=all(row["decoded_vertices_finite"] for row in rows),
                             cuda_peak_increment_max_bytes=max(row["cuda_peak_increment_bytes"] for row in rows) if device == "cuda" else None)
    ratio = summary["SO3_F0"]["full_ms"]["p50"]/summary["ambient_F0"]["full_ms"]["p50"]
    counts_valid = all(row["network_calls"] == (1 if "regression" in row["mode"] else 8)
                       and row["candidate_state_evaluations"] == (1 if "regression" in row["mode"] else 256)
                       and row["decoder_states"] == row["numeric_candidates"]
                       and row["decoder_batch_calls"] == 1 and row["decoded_vertices_finite"]
                       and row["selected_id"] is not None for row in records)
    legal_equal = summary["SO3_F0"]["numeric_candidates"] == summary["ambient_F0"]["numeric_candidates"]
    return {"scope": "fixed analytic typed-pixel single-model generate/publish/SMPLH/A4 cost; random weights",
            "device": device, "hardware": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
            "torch": torch.__version__, "cpu_threads": 1, "seed": 7, "repeats": repeats, "warmup": warmup,
            "input": "actual licensed identity SMPLH beta0/Tz3 projected with fixture rig; no detector/capture",
            "asset_paths": config["assets"], "normalization_source": normal.source,
            "budgets": {"F0": {"K": 32, "steps": 8}, "regression": {"K": 1, "calls": 1}, "decision": "A4"},
            "parameters": {name: sum(parameter.numel() for parameter in model.parameters()) for name, model in models.items()},
            "common_initialization_copied_except_pose_head": True, "state_reset_per_call": True,
            "summary": summary, "records": records, "budget_and_decoder_counts_valid": counts_valid,
            "SO3_over_ambient_full_p50": ratio, "cost_promotion_gate": counts_valid and legal_equal and ratio <= 1.25,
            "memory_scope": "per-call allocator peak/increment; both models and surface resident; process/driver usage excluded",
            "training": False, "checkpoint_loaded": False, "full_system_30fps_claim_supported": False,
            "boundary": "small fixed random-model timing sample; clocks/power/concurrent load uncontrolled; no accuracy or calibration conclusion"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--matched-controls", action="store_true")
    parser.add_argument("--representation-controls", action="store_true")
    args = parser.parse_args()
    if args.representation_controls:
        if args.matched_controls:
            parser.error("representation and two-stage matched controls are separate experiments")
        report = run_representation_controls(repeats=args.repeats, warmup=args.warmup, device=args.device)
    else:
        report = run(repeats=args.repeats, warmup=args.warmup, device=args.device, matched_controls=args.matched_controls)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text+"\n")
    print(text)


if __name__ == "__main__":
    main()
