"""Nonlinear transfer obstructions using the actual project fisheye operator."""
import argparse
import json
import math
from pathlib import Path

import torch

from realtime_app.learning_flow_mesh.rotations import matrix_to_six, six_to_matrix

from .geometry_observation import Camera, project


def audit():
    camera = Camera("analytic-camera-left", "LEFT",
                    torch.tensor([[70., 0., 128.], [0., 70., 128.], [0., 0., 1.]],
                                 dtype=torch.float64),
                    torch.zeros(4, dtype=torch.float64), torch.eye(3, dtype=torch.float64),
                    torch.zeros(3, dtype=torch.float64), (256, 256))
    calls = 0

    def pixel(x):
        nonlocal calls
        calls += 1
        point = torch.stack((x, x*0, x*0+1))[None]
        return project(point, camera)

    target, target_valid = pixel(torch.tensor(0., dtype=torch.float64))
    x = torch.tensor(4., dtype=torch.float64, requires_grad=True)
    uv, valid = pixel(x)
    residual = uv[0, 0]-target[0, 0]
    derivative = torch.autograd.grad(uv[0, 0], x)[0]
    step = -residual.detach()/derivative
    initial_cost = float(residual.detach().square()/2)
    attempts = []
    for alpha in (1., .5, .25, .125, .0625):
        proposal = x.detach()+alpha*step
        projected, legal = pixel(proposal)
        cost = float((projected[0, 0]-target[0, 0]).square()/2)
        accepted = bool(legal.all()) and cost < initial_cost
        attempts.append({"alpha": alpha, "x": float(proposal), "cost": cost,
                         "legal": bool(legal.all()), "accepted": accepted})
        if accepted:
            break
    assert bool(target_valid.all()) and bool(valid.all())
    assert abs(float(residual.detach()+derivative*step)) < 1e-10
    assert attempts[0]["cost"] > initial_cost
    assert attempts[-1]["accepted"]
    rotations = []
    for angle in (-math.pi/2, math.pi/2):
        c, s = math.cos(angle), math.sin(angle)
        rotations.append(torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]],
                                      dtype=torch.float64))
    packed = matrix_to_six(torch.stack(rotations))
    _, individual_valid = six_to_matrix(packed)
    mean = packed.mean(0)
    _, mean_valid = six_to_matrix(mean)
    assert individual_valid.all() and not mean_valid.all()
    return {
        "date": "2026-10-12", "kind": "actual_operator_analytic_fixture",
        "fisheye": {"operator": "geometry_observation.project", "camera": "analytic K70/c128, D0, R=I, T=0, image256x256",
                    "point": "[x,0,1] metres", "initial_x": 4., "target_x": 0.,
                    "initial_cost": initial_cost, "jacobian": float(derivative),
                    "full_step": float(step), "linearized_residual": float(residual.detach()+derivative*step),
                    "attempts": attempts, "projection_calls": calls,
                    "interpretation": "local solve is not nonlinear decrease certificate; bounded actual-operator checks incur extra cost"},
        "rotation_mean": {"individual_angles_deg": [-90, 90], "individual_valid": individual_valid.tolist(),
                          "mean6d": mean.tolist(), "mean_valid": bool(mean_valid.all()),
                          "interpretation": "conditional Euclidean mean may lie outside legal decoded rotations; endpoint predictor is not a guaranteed legal mean human"},
        "training": False, "mainline_changed": False, "efficacy_verified": False,
        "decision": "do not directly transplant linear HQS or unprotected endpoint guidance; audit candidate legality and full nonlinear cost before generator replacement"
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
