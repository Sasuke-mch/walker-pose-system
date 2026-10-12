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


def check_endpoint_direction(surface):
    """One real float64 endpoint/FD comparison, separate from runtime timing."""
    from realtime_app.learning_flow_mesh.rotations import matrix_to_six

    from .candidate_store import from_parameters
    from .geometry_information import endpoint_cost_vjp, local_information
    from .geometry_observation import NoiseModel, observation_errors, project
    from .input_binding import observation_snapshot
    from .schema import Topology
    from .so3_flow import rotation_exp

    generator = torch.Generator().manual_seed(7)
    rotations = rotation_exp(torch.randn(1, 52, 3, generator=generator, dtype=torch.float64)*.03)
    translation = torch.tensor([[.04, -.02, 3.]], dtype=torch.float64)
    beta = torch.linspace(-.05, .05, 10, dtype=torch.float64)[None]
    raw = torch.cat((matrix_to_six(rotations).flatten(1), translation, beta), -1)
    cameras, noise = rig(), NoiseModel.proxy()
    with torch.no_grad():
        reference = surface(torch.eye(3, dtype=torch.float64).expand(1, 52, 3, 3),
                            torch.tensor([[0., 0., 3.]], dtype=torch.float64),
                            torch.zeros(1, 10, dtype=torch.float64))["coco17"]
        xy = torch.stack([project(reference, camera)[0][0] for camera in cameras])
    obs = replace(observation(), xy=xy).validate()
    store = from_parameters(raw, Topology(), ids=["real_endpoint:0"], seeds=[7], current_observed=True,
                            provenance={"observation_binding": observation_snapshot(obs)}, resources={})
    values = torch.full((169,), .1, dtype=torch.float64)
    values[156:159] = torch.tensor([.2, .4, .3], dtype=torch.float64)
    values[-10:] = .08
    scale = TangentScale(values, "analytic_fixture_nonuniform; not training statistics")
    vjp = endpoint_cost_vjp(store, obs, cameras, noise, scale, surface)
    info = local_information(store, obs, cameras, noise, scale, surface, step=1e-4, chunk=16)
    report = {"scope": "one real SMPLH52 endpoint; CPU float64; independent local FD Jacobian",
              "seed": 7, "rotation_source": "Exp(0.03*Gaussian3) per node", "translation": translation[0].tolist(),
              "beta": beta[0].tolist(), "scale": values.tolist(), "admitted_count": int(obs.admitted().sum()),
              "vjp_available": vjp.available.tolist(), "vjp_reasons": vjp.reasons,
              "vjp_resources": vjp.resources, "FD_resources": info.resources, "FD_reasons": info.reasons,
              "boundary": "FD validates its probes; VJP validates only the endpoint. FD cost excluded from runtime modes"}
    if not vjp.available.all() or info.jacobian is None:
        return {**report, "direction_gate": False}
    with torch.no_grad():
        points = surface(rotations, translation, beta)["coco17"]
        _, _, uv, _ = observation_errors(points, obs, cameras, noise)
        residual = (uv[0]-obs.xy)[obs.admitted()]
        precision = noise.precision.to(residual)[obs.admitted()]
        square = torch.einsum("ji,jil,jl->j", residual, precision, residual)
        quadratic = bool((square <= 9).all())
        if not quadratic:
            return {**report, "quadratic_region": False, "direction_gate": False}
        reference_gradient = info.jacobian.T @ (precision @ residual[..., None]).flatten()/obs.admitted().sum()
        actual = vjp.gradient[0]
        relative = float((actual-reference_gradient).norm()/reference_gradient.norm())
        blocks = {name: {"finite": bool(torch.isfinite(actual[part]).all()),
                         "norm": float(actual[part].norm()),
                         "max_abs_difference": float((actual-reference_gradient)[part].abs().max())}
                  for name, part in (("pose", slice(0, 156)), ("translation", slice(156, 159)), ("beta", slice(159, 169)))}
    return {**report, "quadratic_region": quadratic, "VJP_gradient": actual.tolist(),
            "FD_JtWr_over_admitted_gradient": reference_gradient.tolist(), "relative_error": relative,
            "blocks": blocks, "direction_gate": relative <= 1e-4 and all(b["finite"] for b in blocks.values())}


def run_representation_controls(*, repeats=8, warmup=2, device="cuda", compare_rotation_kernels=False,
                                endpoint_feedback_controls=False):
    """Same fixed-input single-model cost comparison; never tunes model budget."""
    from realtime_app.learning_flow_mesh.body import BodyDecoder
    from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor

    from .aggregate import decide
    from .candidate_store import infer
    from .geometry_information import endpoint_cost_vjp
    from .geometry_observation import NoiseModel, project
    from .so3_flow import SO3JointFlow

    if repeats < 2 or warmup < 0:
        raise ValueError("at least two recorded repetitions required")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    ambient = JointFlow().to(device).eval()
    normal = fixture_normalization(ambient.topology, device=device)
    manifold = SO3JointFlow(normal).to(device).eval()
    common = {key: value for key, value in ambient.state_dict().items()
              if key in manifold.state_dict() and not key.startswith("pose_out.")}
    manifold.load_state_dict(common, strict=False)
    models = {"ambient": ambient, "SO3": manifold}
    if compare_rotation_kernels:
        matrix_model = SO3JointFlow(normal, rotation_kernel="matrix_exp").to(device).eval()
        matrix_model.load_state_dict(manifold.state_dict())
        models["SO3_matrix_exp"] = matrix_model
    config = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    surface = CountedSurface(SurfaceDecoder(BodyDecoder(**config["assets"]).to(device),
                                            torch.tensor(load_coco17_regressor(Path("models/smpl/J_regressor_coco.npy")),
                                                         device=device, dtype=torch.float32)))
    direction_check = None
    if endpoint_feedback_controls:
        direction_check = check_endpoint_direction(surface.surface.to(device="cpu", dtype=torch.float64))
        surface.surface.to(device=device, dtype=torch.float32)
        if not direction_check["direction_gate"]:
            return {"scope": "endpoint VJP correctness failed; timing not executed",
                    "direction_check": direction_check, "cost_promotion_gate": False,
                    "training": False, "checkpoint_loaded": False}
        models["SO3_endpoint_VJP"] = manifold  # Same instance/weights; overhead only.
    cameras, scale = rig(), TangentScale.fixture(ambient.topology)
    with torch.no_grad():
        points = surface(torch.eye(3, device=device).expand(1, 52, 3, 3),
                         torch.tensor([[0., 0., 3.]], device=device), torch.zeros(1, 10, device=device))["coco17"][0]
        raw = torch.stack([project(points, camera)[0] for camera in cameras]).detach()
    fixture = observation()
    obs = replace(fixture, xy=raw, **{name: getattr(fixture, name).to(device) for name in
                                    ("confidence", "present", "quality", "geometry", "views", "origins", "directions")}).validate()
    modes = [("ambient", False), ("SO3", False), ("ambient", True), ("SO3", True)]
    if compare_rotation_kernels:
        modes = [(representation, regression) for regression in (False, True) for representation in models]
    if endpoint_feedback_controls:
        modes = [("SO3", False), ("SO3_endpoint_VJP", False)]
    noise = NoiseModel.proxy()

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
        feedback = None
        with torch.no_grad():
            candidates, _ = infer(model, normal, obs, state=None, k=k, steps=steps, regression=regression)
            sync(device)
            generated_at = time.perf_counter()
            if representation == "SO3_endpoint_VJP":
                feedback = endpoint_cost_vjp(candidates, obs, cameras, noise, scale, surface)
                sync(device)
            feedback_at = time.perf_counter()
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
                "endpoint_vjp_ms": (feedback_at-generated_at)*1000 if feedback is not None else 0.,
                "decode_ms": (decoded_at-feedback_at)*1000, "decision_ms": (finished-decoded_at)*1000,
                "full_ms": (finished-start)*1000, "network_calls": candidates.resources["network_calls"],
                "candidate_state_evaluations": candidates.resources["candidate_state_evaluations"],
                "decoder_batch_calls": surface.calls, "decoder_states": surface.states,
                "numeric_candidates": int(legal.sum()), "action_candidates": int(candidates.actions.sum()),
                "decoded_vertices_finite": decoded_finite, "rejection_reasons": candidates.reasons,
                "endpoint_vjp_available": feedback.available.tolist() if feedback is not None else None,
                "endpoint_vjp_reasons": feedback.reasons if feedback is not None else None,
                "endpoint_vjp_resources": feedback.resources if feedback is not None else None,
                "reverse_calls": feedback.resources["reverse_calls"] if feedback is not None else 0,
                "selected_id": decision.selected_id, "solver": candidates.resources["solver"],
                "rotation_kernel": getattr(model, "rotation_kernel", "ambient_additive"),
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
    for name in (representation+("_regression" if regression else "_F0") for representation, regression in modes):
        rows = [row for row in records if row["mode"] == name]
        metrics = ("generation_outer_ms", "generation_internal_ms", "endpoint_vjp_ms", "decode_ms", "decision_ms", "full_ms")
        summary[name] = {key: {"p50": float(np.quantile([row[key] for row in rows], .5)),
                               "p95": float(np.quantile([row[key] for row in rows], .95))} for key in metrics}
        summary[name].update(numeric_candidates=[row["numeric_candidates"] for row in rows],
                             decoded_vertices_finite=all(row["decoded_vertices_finite"] for row in rows),
                             cuda_peak_increment_max_bytes=max(row["cuda_peak_increment_bytes"] for row in rows) if device == "cuda" else None)
    compared = "SO3_endpoint_VJP_F0" if endpoint_feedback_controls else "SO3_F0"
    baseline_mode = "SO3_F0" if endpoint_feedback_controls else "ambient_F0"
    ratio = summary[compared]["full_ms"]["p50"]/summary[baseline_mode]["full_ms"]["p50"]
    counts_valid = all(row["network_calls"] == (1 if "regression" in row["mode"] else 8)
                       and row["candidate_state_evaluations"] == (1 if "regression" in row["mode"] else 256)
                       and row["decoder_states"] == row["numeric_candidates"]*(2 if row["endpoint_vjp_available"] is not None else 1)
                       and row["decoder_batch_calls"] == (2 if row["endpoint_vjp_available"] is not None else 1)
                       and row["reverse_calls"] == (1 if row["endpoint_vjp_available"] is not None else 0)
                       and (row["endpoint_vjp_available"] is None or all(row["endpoint_vjp_available"]))
                       and row["decoded_vertices_finite"]
                       and row["selected_id"] is not None for row in records)
    legal_equal = summary[compared]["numeric_candidates"] == summary[baseline_mode]["numeric_candidates"]
    return {"scope": ("fixed analytic typed-pixel SO3 generate/publish/endpoint VJP/final SMPLH/A4 overhead; no correction"
                      if endpoint_feedback_controls else
                      "fixed analytic typed-pixel single-model generate/publish/SMPLH/A4 cost; random weights"),
            "device": device, "hardware": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
            "torch": torch.__version__, "cpu_threads": 1, "seed": 7, "repeats": repeats, "warmup": warmup,
            "input": "actual licensed identity SMPLH beta0/Tz3 projected with fixture rig; no detector/capture",
            "asset_paths": config["assets"], "normalization_source": normal.source,
            "budgets": {"F0": {"K": 32, "steps": 8}, "regression": {"K": 1, "calls": 1}, "decision": "A4"},
            "parameters": {name: sum(parameter.numel() for parameter in model.parameters()) for name, model in models.items()},
            "common_initialization_copied_except_pose_head": True, "state_reset_per_call": True,
            "rotation_kernels": {name: getattr(model, "rotation_kernel", "ambient_additive") for name, model in models.items()},
            "SO3_kernel_control_weights_identical": compare_rotation_kernels,
            "summary": summary, "records": records, "budget_and_decoder_counts_valid": counts_valid,
            ("endpoint_VJP_over_SO3_full_p50" if endpoint_feedback_controls else "SO3_over_ambient_full_p50"): ratio,
            "cost_promotion_gate": counts_valid and legal_equal and ratio <= 1.25,
            "comparison_modes": {"numerator": compared, "denominator": baseline_mode},
            "endpoint_direction_check": direction_check,
            "endpoint_feedback_applied": False,
            "endpoint_controls_share_same_model_instance": endpoint_feedback_controls,
            "noise_source": noise.source, "noise_calibrated": noise.calibrated,
            "tangent_scale_source": scale.source,
            "memory_scope": "per-call allocator peak/increment; all compared models and surface resident; process/driver usage excluded",
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
    parser.add_argument("--compare-rotation-kernels", action="store_true")
    parser.add_argument("--endpoint-feedback-controls", action="store_true",
                        help="one real direction check then fixed K32/8 SO3 endpoint VJP overhead; no correction applied")
    args = parser.parse_args()
    if args.compare_rotation_kernels and not args.representation_controls:
        parser.error("rotation kernel comparison requires representation controls")
    if args.endpoint_feedback_controls and (args.representation_controls or args.matched_controls or args.compare_rotation_kernels):
        parser.error("endpoint feedback and other controls are separate experiments")
    if args.representation_controls or args.endpoint_feedback_controls:
        if args.matched_controls:
            parser.error("representation and two-stage matched controls are separate experiments")
        report = run_representation_controls(repeats=args.repeats, warmup=args.warmup, device=args.device,
                                             compare_rotation_kernels=args.compare_rotation_kernels,
                                             endpoint_feedback_controls=args.endpoint_feedback_controls)
    else:
        report = run(repeats=args.repeats, warmup=args.warmup, device=args.device, matched_controls=args.matched_controls)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text+"\n")
    print(text)


if __name__ == "__main__":
    main()
