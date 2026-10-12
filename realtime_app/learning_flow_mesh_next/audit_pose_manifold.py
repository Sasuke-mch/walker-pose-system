"""Rigid internal bones versus pose-dependent surface landmark distances."""
import argparse
import json
import math
from pathlib import Path

import torch

from .body import SurfaceDecoder
from .tangent import skew


def mean_bone_identity():
    cases = {}
    for name, samples in (
        ("one_edge_ambiguous", [[[0., 0., 1.], [1., 0., 0.]],
                                [[0., 0., -1.], [1., 0., 0.]]]),
        ("both_edges_ambiguous", [[[0., 0., 1.], [1., 0., 0.]],
                                 [[0., 0., -1.], [-1., 0., 0.]]]),
    ):
        edges = torch.tensor(samples, dtype=torch.float64)
        mean = edges.mean(0)
        variance = (edges-mean).square().sum(-1).mean(0)
        square = edges.square().sum(-1).mean(0)
        assert torch.allclose(mean.square().sum(-1), square-variance, atol=1e-14)
        cases[name] = {"mean_edge_lengths": mean.norm(dim=-1).tolist(),
                       "trace_edge_covariance": variance.tolist()}
    assert cases["one_edge_ambiguous"]["mean_edge_lengths"] == [0., 1.]
    assert cases["both_edges_ambiguous"]["mean_edge_lengths"] == [0., 0.]
    return cases


def run():
    from realtime_app.learning_flow_mesh.body import BodyDecoder
    from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor

    torch.set_num_threads(1)
    assets = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text())["assets"]
    decoder = BodyDecoder(**assets).double().eval()
    regressor_path = Path("models/smpl/J_regressor_coco.npy")
    regressor = torch.tensor(load_coco17_regressor(regressor_path), dtype=torch.float64)
    surface = SurfaceDecoder(decoder, regressor)
    delta = torch.zeros(2, 52, 3, dtype=torch.float64)
    for node, axis, degrees in ((18, 2, 70), (19, 2, -70), (4, 0, 60), (5, 0, 45)):
        delta[1, node, axis] = math.radians(degrees)
    rotations = torch.matrix_exp(skew(delta))
    with torch.no_grad():
        result = surface(rotations, torch.tensor([[0., 0., 3.]]*2, dtype=torch.float64),
                         torch.zeros(2, 10, dtype=torch.float64))
    parents = decoder.parents[1:]
    internal = (result["joints"][:, 1:] - result["joints"][:, parents]).norm(dim=-1)
    pairs = ((5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16))
    coco = result["coco17"]
    distances = torch.stack([(coco[:, a]-coco[:, b]).norm(dim=-1) for a, b in pairs], 1)
    internal_change = float((internal[1]-internal[0]).abs().max())
    surface_change = float((distances[1]-distances[0]).abs().max())
    finite = all(bool(torch.isfinite(result[key]).all()) for key in ("vertices", "joints", "coco17"))
    report = {"device": "cpu", "dtype": "float64", "assets": decoder.asset_paths,
              "regressor": str(regressor_path), "beta": [0.]*10, "T": [0., 0., 3.],
              "posed_nodes": [{"node": n, "axis": a, "degrees": d}
                              for n, a, d in ((18, 2, 70), (19, 2, -70), (4, 0, 60), (5, 0, 45))],
              "decoder_states": 2, "finite": finite,
              "internal_edges": [{"node": i+1, "parent": int(p), "lengths_m": internal[:, i].tolist()}
                                 for i, p in enumerate(parents)],
              "surface_edges": [{"coco_pair": list(pair), "lengths_m": distances[:, i].tolist(),
                                 "change_m": float(distances[1, i]-distances[0, i])}
                                for i, pair in enumerate(pairs)],
              "internal_max_change_m": internal_change, "surface_max_change_m": surface_change,
              "analytic_mean_bone": mean_bone_identity(), "training": False,
              "boundary": "two artificial legal poses, not real accuracy, anatomical validity or efficacy"}
    report["gate"] = finite and internal_change < 1e-7 and surface_change > 1e-4
    report["decision"] = "do not impose rigid-skeleton limb lengths on the COCO17 surface operator"
    assert report["gate"], report
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in
                      ("gate", "internal_max_change_m", "surface_max_change_m", "analytic_mean_bone")}))


if __name__ == "__main__":
    main()
