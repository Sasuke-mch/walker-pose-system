"""Compatibility of a local decision chart with a global SO(3) source path."""
import argparse
import json
import math
from pathlib import Path

import torch

from realtime_app.learning_flow_mesh.rotations import validate_rotation_matrices

from .tangent import shortest_log, skew


def run():
    torch.set_num_threads(1)
    dtype = torch.float64
    margin = 1e-3  # Matches the existing shortest_log default.
    vectors = torch.tensor([[0., 0., .4], [0., 0., math.pi-margin/2]], dtype=dtype)
    rotations = torch.matrix_exp(skew(vectors))
    validate_rotation_matrices(rotations)
    regular = shortest_log(rotations[:1])
    assert torch.allclose(regular, vectors[:1], atol=1e-13, rtol=0)
    try:
        shortest_log(rotations)
        raise AssertionError("mixed near-pi tensor should be rejected")
    except ValueError as error:
        rejected = str(error)
    assert rejected == "near_pi_rotation_chart"

    # A uniform point of S3, with antipodes identified, induces uniform SO3.
    generator = torch.Generator().manual_seed(7)
    count = 200000
    quaternion = torch.randn(count, 4, generator=generator, dtype=dtype)
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    angle = 2*torch.acos(quaternion[:, 0].abs().clamp(max=1))
    p = (margin+math.sin(margin))/math.pi
    tail_count = int((angle > math.pi-margin).sum())
    empirical = tail_count/count
    se = math.sqrt(p*(1-p)/count)
    assert abs(empirical-p) < 5*se
    populations = [{"independent_rotation_pairs": n,
                    "probability_any_rejected": -math.expm1(n*math.log1p(-p))}
                   for n in (22, 52, 16*52, 32*52)]

    zero = torch.zeros(3, dtype=dtype)
    forward = shortest_log(torch.matrix_exp(skew(zero)))
    derivative = torch.autograd.functional.jacobian(
        lambda v: shortest_log(torch.matrix_exp(skew(v))), zero)
    finite = bool(torch.isfinite(derivative).all())
    encoded_derivative = [[float(v) if torch.isfinite(v) else "nonfinite_NaN"
                           if torch.isnan(v) else "nonfinite_Infinity" for v in row]
                          for row in derivative]
    time_controls = []
    for name, fixed_log in (("identity", forward.detach()), ("regular", regular[0].detach())):
        omega = skew(fixed_log)
        point, velocity = torch.func.jvp(lambda t, omega=omega: torch.matrix_exp(t*omega),
                                         (zero.new_tensor(.4),), (zero.new_tensor(1.),))
        validate_rotation_matrices(point)
        discrepancy = float((velocity-point@omega).abs().max())
        assert torch.isfinite(velocity).all() and discrepancy < 1e-13
        time_controls.append({"endpoint": name, "time": .4, "velocity_finite": True,
                              "closed_form_max_error": discrepancy})
    return {"device": "cpu", "dtype": "float64", "training": False,
            "local_helper": "tangent.shortest_log", "pi_margin_rad": margin,
            "regular_log": regular.tolist(), "mixed_legal_rotations_rejection": rejected,
            "Haar_angle_cdf": "(theta-sin(theta))/pi for theta in[0,pi]",
            "tail_probability": p, "independent_tensor_populations": populations,
            "MonteCarlo": {"source": "normalized iid Gaussian4 quaternion", "seed": 7,
                           "draws": count, "tail_count": tail_count,
                           "fraction": empirical, "binomial_standard_error": se,
                           "within5_standard_errors": True},
            "identity": {"forward": forward.tolist(), "backward_finite": finite,
                         "backward_J": encoded_derivative,
                         "expected_J": torch.eye(3, dtype=dtype).tolist()},
            "frozen_endpoints_time_JVP": time_controls,
            "decision": "local decision chart is not a drop-in global/differentiable SO3 path",
            "limits": ["not learned JointFlow or dependent pairwiseA6 rejection frequencies",
                       "endpoint gradients are not required for frozen source/target time-only JVP",
                       "existing decision fallback retained; no model or probability calibration claim"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("tail_probability", "independent_tensor_populations",
                                                  "MonteCarlo", "identity", "frozen_endpoints_time_JVP")}))


if __name__ == "__main__":
    main()
