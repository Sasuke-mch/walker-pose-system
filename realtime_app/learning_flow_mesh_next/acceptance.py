"""Pre-test validation locks and evidence-aware preregistered decision gates."""
from dataclasses import asdict

import torch

from .information_risk import RiskPolicy


def lock_policy(validation_risks, *, split, lambda0, lambda_g, iqr_fraction, source):
    if split != "validation" or iqr_fraction not in (0, .01, .05) or lambda_g not in (0, .1, 1, 10):
        raise ValueError("policy selection must use the preregistered validation grid")
    values = torch.as_tensor(validation_risks, dtype=torch.float64)
    if not len(values) or not torch.isfinite(values).all() or not source:
        raise ValueError("finite validation risks and calibration/validation source required")
    iqr = torch.quantile(values, .75)-torch.quantile(values, .25)
    policy = RiskPolicy(lambda0=lambda0, lambda_g=lambda_g, delta=float(iqr)*iqr_fraction,
                        source=source, ambiguity_calibrated=False).validate()
    return {"policy": asdict(policy), "validation_risk_iqr": float(iqr), "iqr_fraction": iqr_fraction,
            "locked_before_test": True, "test_used": False}


def preregistered_gates(metrics):
    requirements = {"ambiguity_energy_relative_improvement": (lambda x: x >= .05),
                    "energy_paired_ci_supports_improvement": bool,
                    "strong_geometry_relative_degradation": (lambda x: x <= .02),
                    "aggregation_relative_improvement": (lambda x: x >= .05),
                    "retention_difference_absolute": (lambda x: x <= .01),
                    "illegal_rate_increase": (lambda x: x <= .005),
                    "coverage90": (lambda x: .87 <= x <= .93),
                    "full_pipeline_p95_ms": (lambda x: x <= 33.3)}
    result = {}
    for key, check in requirements.items():
        value = metrics.get(key)
        result[key] = "not_evaluated" if value is None else ("pass" if check(value) else "fail")
    if result["retention_difference_absolute"] == "fail":
        result["aggregation_comparison"] = "coverage_matching_required_before_claim"
    result["accuracy_claim"] = "blocked" if not metrics.get("independent_3d_truth") else "requires_all_relevant_gates"
    result["shape_claim"] = "blocked" if not metrics.get("independent_scan_or_bounded_dimensions") else "requires_shape_metrics"
    result["mechanism_revisions_remaining"] = max(0, 2-int(metrics.get("justified_performance_revisions", 0)))
    result["direction"] = "retain_FM; no mechanism efficacy claimed without preregistered evidence"
    return result
