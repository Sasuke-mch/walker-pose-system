"""Counterexample to anchor-only derivative availability as neighborhood validity."""
import argparse
import json
import math
from pathlib import Path

import torch

from realtime_app.learning_flow_mesh.rotations import matrix_to_six

from .candidate_store import from_parameters
from .geometry_information import local_information
from .geometry_observation import Camera, NoiseModel, project
from .schema import Topology
from .tangent import TangentScale
from .tests.test_firewall import observation
from .tests.test_information import rigid_surface


def run():
    torch.set_num_threads(1)
    dtype = torch.float64
    obs = observation()
    K = torch.tensor([[100., 0., 128.], [0., 100., 128.], [0., 0., 1.]], dtype=dtype)
    cameras = tuple(Camera(identity, role, K, torch.zeros(4, dtype=dtype),
                           torch.eye(3, dtype=dtype), torch.tensor([-i*.2, 0., 0.], dtype=dtype),
                           (256, 256))
                    for i, (identity, role) in enumerate(zip(obs.camera_ids, ("LEFT", "RIGHT"))))
    topology = Topology(52)
    scale = TangentScale.fixture(topology)
    output = {"device": "cpu", "dtype": "float64", "surface": "analytic rigid_surface, not SMPLH",
              "camera": {"f": 100, "c": 128, "D": [0]*4, "baseline": .2, "image_size": [256, 256]},
              "step": 1e-4, "cases": [], "training": False}
    for label, x in (("interior", 0.), ("near_right_edge", math.tan((256-1e-5-128)/100))):
        position = torch.tensor([x, 0., 1.], dtype=dtype)
        raw = torch.cat((matrix_to_six(torch.eye(3, dtype=dtype)).repeat(52), position,
                         torch.zeros(10, dtype=dtype)))[None]
        store = from_parameters(raw, topology, ids=[label], seeds=[0], current_observed=True,
                                provenance={}, resources={})
        info = local_information(store, obs, cameras, NoiseModel.proxy(), scale, rigid_surface)

        def pixels(delta, position=position):
            points = (position + delta*scale.values[156:159])[None].expand(17, 3)
            return torch.stack([project(points, c)[0] for c in cameras]).flatten()

        derivative = torch.autograd.functional.jacobian(pixels, torch.zeros(3, dtype=dtype))
        shift = torch.tensor([1e-4, 0., 0.], dtype=dtype)*scale.values[156:159]
        uv, valid = project(position[None], cameras[0])
        shifted_uv, shifted_valid = project((position+shift)[None], cameras[0])
        output["cases"].append({"name": label, "position": position.tolist(),
                                "anchor_uv": uv.tolist(), "anchor_valid": bool(valid.all()),
                                "plus_x_uv": shifted_uv.tolist(), "plus_x_valid": bool(shifted_valid.all()),
                                "AD_finite": bool(torch.isfinite(derivative).all()),
                                "FD_available": info.matrix is not None, "FD_reasons": info.reasons})
    interior, boundary = output["cases"]
    assert interior["FD_available"] and interior["AD_finite"] and interior["plus_x_valid"]
    assert boundary["anchor_valid"] and boundary["AD_finite"] and not boundary["plus_x_valid"]
    assert not boundary["FD_available"]
    assert any("anchor_or_perturbation_projection_invalid" in r for r in boundary["FD_reasons"])
    output["decision"] = "anchor-only AD is not a drop-in replacement; close this shortcut"
    output["boundary"] = "constructed counterexample, not population frequency or physical accuracy"
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
