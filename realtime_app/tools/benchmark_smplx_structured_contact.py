#!/usr/bin/env python3
"""Clean SMPL-X window redesign with low-dimensional temporal parameters and contact ablation."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from tools.diagnose_smplx_loss_gradients import parser  # noqa: E402
from tools.benchmark_smplx_r3_integration import (  # noqa: E402
    CHECKPOINTS, DEV_WINDOW, LEARNING_RATE, FitEnv, FitRouteConfig,
    build_per_frame_rows, materialize_checkpoint_result, run_fit, write_jsonl,
)

KNOT_STRIDE = 5


@dataclass
class StructuredRun:
    config: FitRouteConfig
    snapshots: list[dict]
    states: dict[int, dict[str, np.ndarray]]
    final_metrics: dict
    per_frame_rows: list[dict]


def knot_indices(frame_count: int, stride: int = KNOT_STRIDE) -> list[int]:
    indices = list(range(0, frame_count, stride))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices


def interpolate_knots(torch, values, frame_count: int):
    # (K,D) -> (N,D), differentiable piecewise-linear latent trajectory.
    source = values.transpose(0, 1).unsqueeze(0)
    return torch.nn.functional.interpolate(
        source, size=frame_count, mode="linear", align_corners=True
    ).squeeze(0).transpose(0, 1)


def interpolate_knots_cubic(torch, values, frame_count: int):
    """Uniform Catmull-Rom interpolation with replicated endpoint controls."""
    if values.shape[0] < 2:
        return values.expand(frame_count, -1)
    positions = torch.linspace(0.0, float(values.shape[0] - 1), frame_count,
                               dtype=values.dtype, device=values.device)
    segment = torch.floor(positions).to(torch.long).clamp(max=values.shape[0] - 2)
    t = (positions - segment.to(values.dtype))[:, None]
    i0 = (segment - 1).clamp(min=0)
    i1 = segment
    i2 = (segment + 1).clamp(max=values.shape[0] - 1)
    i3 = (segment + 2).clamp(max=values.shape[0] - 1)
    p0, p1, p2, p3 = values[i0], values[i1], values[i2], values[i3]
    return 0.5 * ((2.0 * p1) + (-p0 + p2) * t
                  + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t.square()
                  + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t.pow(3))


def expand_knots(torch, values, frame_count: int, mode: str):
    if mode == "linear":
        return interpolate_knots(torch, values, frame_count)
    if mode == "cubic":
        return interpolate_knots_cubic(torch, values, frame_count)
    raise ValueError(f"unsupported structured interpolation: {mode}")


def contact_proxy_mm(metrics: dict) -> dict:
    terms = metrics["loss_terms"]
    return {
        "foot_contact_rms_mm": float(np.sqrt(max(terms["foot_contact"], 0.0)) * 1000.0),
        "hand_contact_rms_mm": float(np.sqrt(max(terms["hand_contact"], 0.0)) * 1000.0),
    }


def enforce_coco_pelvis_delta(env: FitEnv, root, translation, latent):
    """Apply a differentiable linear translation correction to preserve pelvis motion."""
    torch = env.torch
    _, predicted_coco, _, _ = env.forward(root, translation, latent)
    predicted_pelvis = 0.5 * (predicted_coco[:, 11] + predicted_coco[:, 12])
    target_pelvis = 0.5 * (env.prefit_all_target_m[:, 11] + env.prefit_all_target_m[:, 12])
    correction_delta = (target_pelvis[-1] - target_pelvis[0]) - (
        predicted_pelvis[-1] - predicted_pelvis[0]
    )
    alpha = torch.linspace(0.0, 1.0, env.window_size, dtype=translation.dtype,
                           device=translation.device)[:, None]
    return translation + alpha * correction_delta[None, :]


def run_structured(env: FitEnv, config: FitRouteConfig, iterations: int = 400, base_params=None) -> StructuredRun:
    torch = env.torch
    knots = knot_indices(env.window_size, config.structured_knot_stride)
    if base_params is None:
        base_root, base_translation, base_latent = env.make_params(set())
        root_knots = torch.nn.Parameter(base_root.detach()[knots].clone())
        translation_base_knots = base_translation.detach()[knots].clone()
        latent_knots = torch.nn.Parameter(base_latent.detach()[knots].clone())
    elif isinstance(base_params, dict):
        root_knots = torch.nn.Parameter(torch.as_tensor(base_params["root"], dtype=torch.float32, device=env.device).clone())
        translation_base_knots = torch.as_tensor(base_params["translation"], dtype=torch.float32, device=env.device).clone()
        latent_knots = torch.nn.Parameter(torch.as_tensor(base_params["latent"], dtype=torch.float32, device=env.device).clone())
    else:
        base_root, base_translation, base_latent = base_params
        root_knots = torch.nn.Parameter(base_root.detach()[knots].clone())
        translation_base_knots = base_translation.detach()[knots].clone()
        latent_knots = torch.nn.Parameter(base_latent.detach()[knots].clone())
    if config.preserve_translation_endpoints:
        translation_initial = translation_base_knots[1:-1]
    elif config.preserve_translation_net_delta:
        translation_initial = translation_base_knots[:-1]
    else:
        translation_initial = translation_base_knots
    translation_knots = torch.nn.Parameter(translation_initial.clone())
    latent_knots = torch.nn.Parameter(base_latent.detach()[knots].clone())
    parameters = (root_knots, translation_knots, latent_knots)
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)
    masks = env.masks_with_holdout(set())
    snapshots = []
    states = {}

    def expanded():
        if config.preserve_translation_endpoints:
            translation_values = torch.cat(
                (translation_base_knots[:1], translation_knots, translation_base_knots[-1:]), dim=0
            )
        elif config.preserve_translation_net_delta:
            frozen_delta = translation_base_knots[-1] - translation_base_knots[0]
            translation_values = torch.cat(
                (translation_knots, (translation_knots[:1] + frozen_delta)), dim=0
            )
        else:
            translation_values = translation_knots
        root_full = expand_knots(torch, root_knots, env.window_size, config.structured_interpolation)
        translation_full = expand_knots(torch, translation_values, env.window_size, config.structured_interpolation)
        latent_full = expand_knots(torch, latent_knots, env.window_size, config.structured_interpolation)
        if config.preserve_coco_pelvis_net_delta:
            translation_full = enforce_coco_pelvis_delta(
                env, root_full, translation_full, latent_full
            )
        return root_full, translation_full, latent_full

    for step in range(iterations + 1):
        if step in CHECKPOINTS:
            full = expanded()
            terms = env.loss_terms(*full, masks, config)
            snap = env.snapshot(*full, masks, config, set())
            snap.update({"iteration": step, "loss_terms": env.terms_to_float(terms),
                         "loss_total": float(env.total_loss(terms, config).detach().cpu()),
                         "knot_count": len(knots), "knot_stride": config.structured_knot_stride})
            snapshots.append(snap)
            states[step] = {name: value.detach().cpu().numpy().copy() for name, value in zip(
                ("root", "translation", "latent"), parameters
            )}
        if step == iterations:
            break
        optimizer.zero_grad()
        full = expanded()
        terms = env.loss_terms(*full, masks, config)
        env.total_loss(terms, config).backward()
        optimizer.step()
    full = expanded()
    final = snapshots[-1]
    rows = build_per_frame_rows(env, config, *full, set(), "structured_development", 0, False)
    return StructuredRun(config, snapshots, states, final, rows)


def materialize_structured(env: FitEnv, source: StructuredRun, iteration: int) -> tuple[dict, list[dict]]:
    torch = env.torch
    state = source.states[iteration]
    values = tuple(torch.nn.Parameter(torch.tensor(state[name], dtype=torch.float32, device=env.device))
                   for name in ("root", "translation", "latent"))
    root_values, translation_values, latent_values = values
    if source.config.preserve_translation_endpoints:
        _, base_translation, _ = env.make_params(set())
        knots = knot_indices(env.window_size, source.config.structured_knot_stride)
        base_knots = base_translation.detach()[knots]
        translation_values = torch.cat((base_knots[:1], translation_values, base_knots[-1:]), dim=0)
    elif source.config.preserve_translation_net_delta:
        _, base_translation, _ = env.make_params(set())
        knots = knot_indices(env.window_size, source.config.structured_knot_stride)
        base_knots = base_translation.detach()[knots]
        frozen_delta = base_knots[-1] - base_knots[0]
        translation_values = torch.cat(
            (translation_values, (translation_values[:1] + frozen_delta)), dim=0
        )
    full = tuple(expand_knots(torch, value, env.window_size, source.config.structured_interpolation) for value in (
        root_values, translation_values, latent_values
    ))
    if source.config.preserve_coco_pelvis_net_delta:
        full = (full[0], enforce_coco_pelvis_delta(env, *full), full[2])
    masks = env.masks_with_holdout(set())
    terms = env.loss_terms(*full, masks, source.config)
    metrics = env.snapshot(*full, masks, source.config, set())
    metrics.update({"iteration": iteration, "loss_terms": env.terms_to_float(terms),
                    "loss_total": float(env.total_loss(terms, source.config).detach().cpu()),
                    "knot_count": len(knot_indices(env.window_size, source.config.structured_knot_stride)),
                    "knot_stride": source.config.structured_knot_stride})
    rows = build_per_frame_rows(env, source.config, *full, set(), "structured_selected", 0, False)
    return metrics, rows


def eligible(metric: dict, baseline: dict) -> bool:
    return bool(metric["negative_depth_count"] == 0 and metric["available_joint_count"] >= 372
                and metric["reproj_left_strict_p95_px"] <= baseline["reproj_left_strict_p95_px"] + 5
                and metric["reproj_right_strict_p95_px"] <= baseline["reproj_right_strict_p95_px"] + 5
                and 57.02 <= metric["pelvis_net_displacement_fitted_mm"] <= 69.69)


def main() -> int:
    args = parser().parse_args()
    out = args.output_dir.resolve()
    if any(out.rglob("metrics.json")):
        raise FileExistsError(f"refuse to overwrite experiment: {out}")
    out.mkdir(parents=True, exist_ok=True)
    env = FitEnv(args, *DEV_WINDOW, device=args.device)
    base_config = FitRouteConfig("B0-per-frame-gated", "prefit", True, None)
    baseline = run_fit(env, base_config, set(), CHECKPOINTS, "structured_reference", iterations=400)
    common = dict(init="prefit", data3d=True, r3soft_weight=None,
                  temporal_point_space="coco", anatomical_temporal_weight=0.0)
    no_contact = run_structured(env, FitRouteConfig(
        "S0-structured-no-contact", foot_contact_weight=0.0, hand_contact_weight=0.0, **common
    ))
    with_contact = run_structured(env, FitRouteConfig("S1-structured-contact", **common))
    repeat_config = FitRouteConfig("B0-repeat", "prefit", True, None)
    repeat = run_fit(env, repeat_config, set(), CHECKPOINTS, "structured_repeat", iterations=400)
    summary = {"knot_stride": KNOT_STRIDE, "routes": {},
               "route_order": ["B0-per-frame-gated", "S0-structured-no-contact", "S1-structured-contact", "B0-repeat"]}

    for name, run, structured in (
        ("B0-per-frame-gated", baseline, False),
        ("S0-structured-no-contact", no_contact, True),
        ("S1-structured-contact", with_contact, True),
        ("B0-repeat", repeat, False),
    ):
        snapshots = run.snapshots
        candidates = [m for m in snapshots if eligible(m, baseline.final_metrics)]
        selected_iteration = min(candidates, key=lambda m: (m["accel_p95_mm_per_frame2"], m["iteration"]))["iteration"] if candidates else None
        if structured and selected_iteration is not None:
            selected_metrics, rows = materialize_structured(env, run, selected_iteration)
        elif structured:
            selected_metrics, rows = run.final_metrics, run.per_frame_rows
        else:
            selected_run = materialize_checkpoint_result(env, run, selected_iteration) if selected_iteration is not None else run
            selected_metrics, rows = selected_run.final_metrics, selected_run.per_frame_rows
        route_dir = out / "routes" / name
        route_dir.mkdir(parents=True)
        write_jsonl(route_dir / "per_frame_metrics.jsonl", rows)
        details = {"endpoint_400": run.final_metrics, "selected_iteration": selected_iteration,
                   "selected_metrics": selected_metrics, "contact_proxy_endpoint": contact_proxy_mm(run.final_metrics),
                   "contact_proxy_selected": contact_proxy_mm(selected_metrics), "checkpoint_metrics": snapshots}
        (route_dir / "metrics.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["routes"][name] = details
    keys = ("loss_total", "accel_p95_mm_per_frame2", "pelvis_accel_p95_mm_per_frame2",
            "pelvis_net_displacement_fitted_mm", "reproj_left_strict_p95_px", "reproj_right_strict_p95_px")
    summary["baseline_repeat_exact"] = all(
        baseline.final_metrics[key] == repeat.final_metrics[key] for key in keys
    )
    (out / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
