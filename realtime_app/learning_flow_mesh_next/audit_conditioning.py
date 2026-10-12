"""Scalar counterexamples for curvature-only information and reused likelihood."""
import argparse
import json
from fractions import Fraction
from pathlib import Path


def audit():
    # y = x + noise, unit precision, fixed anchor x=0: H=1 for both y signs.
    cases = [{"measurement": y, "anchor": 0, "curvature": 1,
              "gradient": -y, "gauss_newton_step": y} for y in (-1, 1)]
    assert cases[0]["curvature"] == cases[1]["curvature"]
    assert cases[0]["gauss_newton_step"] == -cases[1]["gauss_newton_step"]
    # Prior N(0,1), y=1 with unit Gaussian noise. Multiply by likelihood once/twice.
    once_variance, twice_variance = Fraction(1, 2), Fraction(1, 3)
    once_mean, twice_mean = once_variance, 2*twice_variance
    assert twice_variance < once_variance and twice_mean > once_mean
    return {
        "date": "2026-10-12", "kind": "exact_scalar_analytic_counterexamples",
        "curvature_only": {"model": "y=x+eta, W=1, anchor=0", "cases": cases,
                           "conclusion": "curvature alone has no residual direction; current context also contains observations, so this is not a full G1 indistinguishability claim"},
        "likelihood_reuse": {
            "prior": "N(0,1)", "measurement": 1, "noise_variance": 1,
            "once": {"mean": float(once_mean), "variance": float(once_variance)},
            "twice": {"mean": float(twice_mean), "variance": float(twice_variance)},
            "variance_relative_change": float(twice_variance/once_variance-1),
            "conclusion": "ideal observation-conditional posterior multiplied by the same likelihood changes its target; this is not evidence that present A6 decision risk reweights the generator"},
        "boundary": "no neural model, data replay, training, posterior calibration or measured runtime improvement",
        "next": "audit residual-aware operator conditioning on current nonlinear endpoint with legality and total-cost controls before any new generator"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
