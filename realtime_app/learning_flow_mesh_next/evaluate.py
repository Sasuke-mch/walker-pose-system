"""Offline reference metrics and hierarchical statistics; oracle never deploys."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def energy_score(samples, target):
    """Equal-weight multivariate empirical ES (V-statistic, fixed finite K)."""
    if samples.ndim != 2 or target.shape != samples.shape[1:] or not len(samples):
        raise ValueError("expected nonempty K,D samples and D reference")
    if not torch.isfinite(samples).all() or not torch.isfinite(target).all():
        raise ValueError("raw distribution score undefined with nonfinite samples; report failure")
    return (samples-target).norm(dim=-1).mean() - .5*torch.cdist(samples, samples).mean()


def distribution(samples, target):
    if not len(samples) or not torch.isfinite(samples).all() or not torch.isfinite(target).all():
        return {"count": len(samples), "energy_score": None, "coverage90": None,
                "width90": None, "status": "undefined_empty_or_nonfinite_pool"}
    lower, upper = torch.quantile(samples, samples.new_tensor([.05, .95]), dim=0)
    return {"count": len(samples), "energy_score": float(energy_score(samples, target)),
            "coverage90": float(((target >= lower) & (target <= upper)).float().mean()),
            "width90": float((upper-lower).mean()),
            "scope": "marginal_coordinate_intervals; multivariate_ES; fixed_K_V_statistic"}


def candidate_metrics(points, reference, numerical, actions, *, selected_index=None, root_index=0):
    if points.ndim != 3 or reference.shape != points.shape[1:] or points.shape[-1] != 3:
        raise ValueError("K,J,3 absolute own-T points and J,3 reference required")
    if numerical.dtype != torch.bool or actions.dtype != torch.bool or (actions & ~numerical).any():
        raise ValueError("explicit legal/action masks required")
    if not torch.isfinite(reference).all() or not torch.isfinite(points[numerical]).all():
        raise ValueError("nonfinite applicable reference/declared legal sample")
    errors = (points-reference).norm(dim=-1).mean(-1)
    absolute = errors[numerical]
    relative = points - points[:, root_index:root_index+1]
    ref_relative = reference-reference[root_index]
    root_error = (relative-ref_relative).norm(dim=-1).mean(-1)
    selected = float(errors[selected_index]) if selected_index is not None and actions[selected_index] else None
    oracle = float(absolute.min()) if len(absolute) else None
    return {"absolute_mpjpe_m_selected": selected,
            "root_relative_mpjpe_m_selected": float(root_error[selected_index]) if selected is not None else None,
            "fixed_budget_oracle_min_m": oracle,
            "selected_minus_oracle_m": selected-oracle if selected is not None and oracle is not None else None,
            "raw": distribution(points.flatten(1), reference.flatten()),
            "numerically_legal": distribution(points[numerical].flatten(1), reference.flatten()),
            "action_filtered": distribution(points[actions].flatten(1), reference.flatten()),
            "legal_retention": float(numerical.float().mean()), "action_retention": float(actions.float().mean()),
            "illegal_count": int((~numerical).sum()), "selected_unavailable": selected is None}


def surface_error(vertices, reference):
    if vertices.shape != reference.shape or vertices.shape[-1] != 3:
        raise ValueError("matching surface topology required for PVE")
    if not torch.isfinite(vertices).all() or not torch.isfinite(reference).all():
        raise ValueError("nonfinite applicable surface")
    return float((vertices-reference).norm(dim=-1).mean())


def neutral_shape_metrics(surface, candidate_beta, reference_beta, *, basis, reference_basis):
    if basis != reference_basis or not basis:
        raise ValueError("shape coefficients comparable only in the same audited basis")
    from .schema import Topology
    if candidate_beta.shape != (10,) or reference_beta.shape != (10,):
        raise ValueError("beta10 required")
    beta = torch.stack((candidate_beta, reference_beta))
    neutral = torch.eye(3, device=beta.device, dtype=beta.dtype).expand(2, Topology().nodes, 3, 3)
    vertices = surface(neutral, beta.new_zeros(2, 3), beta)["vertices"]
    # Coordinate extent is a declared neutral-mesh diagnostic, not anthropometry.
    dimensions = vertices.amax(1)-vertices.amin(1)
    return {"neutral_surface_error_m": surface_error(vertices[0], vertices[1]),
            "beta_coefficient_rmse_same_basis": float((candidate_beta-reference_beta).square().mean().sqrt()),
            "neutral_coordinate_extent_difference_m": (dimensions[0]-dimensions[1]).cpu().tolist(),
            "measurement_scope": "axis extents are not calibrated clinical body measurements"}


def paired_bootstrap(records, *, iterations=2000, seed=7):
    """Equal subject weight; seeds averaged within matched sequence blocks first.

    Each input is a precomputed contiguous sequence-block metric, not a frame.
    Bootstrap subjects, then sequences, then blocks within each selected subject.
    """
    if not records or iterations < 1:
        raise ValueError("nonempty paired subject/sequence/block records required")
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        if not all(r.get(k) is not None for k in ("subject", "sequence", "block", "training_seed", "baseline", "proposed")):
            raise ValueError("missing paired block provenance")
        if not np.isfinite([r["baseline"], r["proposed"]]).all():
            raise ValueError("failures cannot silently drop from statistical records")
        groups[r["subject"]][r["sequence"]][r["block"]].append((r["training_seed"], r["proposed"]-r["baseline"]))
    data = {}
    all_seed_sets = []
    for subject, sequences in groups.items():
        data[subject] = {}
        for sequence, blocks in sequences.items():
            values = []
            for seeds_and_values in blocks.values():
                seeds = [s for s, _ in seeds_and_values]
                if len(set(seeds)) != len(seeds):
                    raise ValueError("duplicate training seed within block")
                all_seed_sets.append(set(seeds))
                values.append(np.mean([v for _, v in seeds_and_values]))
            data[subject][sequence] = np.asarray(values)
    if any(s != all_seed_sets[0] for s in all_seed_sets):
        raise ValueError("matched training seed coverage required across every block")
    subjects = sorted(data)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iterations):
        subject_values = []
        for subject in rng.choice(subjects, len(subjects), replace=True):
            sequences = sorted(data[subject])
            sequence_values = []
            for sequence in rng.choice(sequences, len(sequences), replace=True):
                values = data[subject][sequence]
                sequence_values.append(rng.choice(values, len(values), replace=True).mean())
            subject_values.append(np.mean(sequence_values))
        draws.append(np.mean(subject_values))
    estimate = np.mean([np.mean([v.mean() for v in seq.values()]) for seq in data.values()])
    return {"proposed_minus_baseline": float(estimate), "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "subjects": len(subjects), "training_seeds": sorted(all_seed_sets[0]),
            "iterations": iterations, "scope": "subject paired; hierarchical sequence blocks; seeds not independent subjects"}


def online_metrics(timestamps, current_observed, published, *, reference_modes=None, selected_modes=None):
    times = np.asarray(timestamps, dtype=float)
    observed, available = np.asarray(current_observed, bool), np.asarray(published, bool)
    if times.shape != observed.shape or available.shape != times.shape or np.any(np.diff(times) <= 0):
        raise ValueError("matching ordered physical time/observed/published arrays required")
    if np.any(available & ~observed):
        raise ValueError("publication without a current observation")
    durations, recoveries = [], []
    missing_start = None
    recovery_start = None
    for t, obs, pub in zip(times, observed, available):
        if not obs and missing_start is None:
            missing_start = t
        if obs and missing_start is not None:
            durations.append(t-missing_start)
            missing_start, recovery_start = None, t
        if pub and recovery_start is not None:
            recoveries.append(t-recovery_start)
            recovery_start = None
    if missing_start is not None:
        durations.append(times[-1]-missing_start)
    result = {"missing_duration_bins_seconds": [0, .1, .5, 2, None],
              "missing_episode_counts": np.histogram(durations, [0, .1, .5, 2, np.inf])[0].tolist(),
              "engineering_publication_recovery_lag_s": recoveries,
              "unrecovered_at_end": recovery_start is not None or missing_start is not None,
              "true_motion_mode_metric": "unavailable_without_reference_mode_labels"}
    if reference_modes is not None or selected_modes is not None:
        if reference_modes is None or selected_modes is None:
            raise ValueError("both independent true-motion mode labels and assessed modes required")
        a, b = np.asarray(reference_modes), np.asarray(selected_modes)
        if a.shape != times.shape or b.shape != times.shape:
            raise ValueError("mode labels must align with physical frames")
        result["true_motion_mode_metric"] = {"accuracy_published": float((a[available] == b[available]).mean()) if available.any() else None,
                                              "reference_switches": int(np.count_nonzero(a[1:] != a[:-1]))}
    return result


def fault_tree(report):
    if report.get("oracle_bad"):
        return "generator_or_data"
    if report.get("oracle_good_selection_bad"):
        return "aggregator"
    if report.get("point_good_coverage_bad"):
        return "distribution_collapse_or_filter_bias"
    if report.get("distribution_good_time_bad"):
        return "time_decision"
    if report.get("efficacy_passed_runtime_bad"):
        return "cache_or_sampling_cost"
    return "insufficient_evidence"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, required=True,
                        help="offline NPZ: candidate_points, reference_points, numerical, actions; never inference input")
    parser.add_argument("--selected-index", type=int)
    args = parser.parse_args()
    with np.load(args.points, allow_pickle=False) as data:
        report = candidate_metrics(*(torch.from_numpy(data[k].copy()) for k in
                                     ("candidate_points", "reference_points", "numerical", "actions")),
                                   selected_index=args.selected_index)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
