"""Reproducible no-training audits; no checkpoint loading or downloads."""
import argparse
import json
import platform
from pathlib import Path

import torch

from .body import SurfaceDecoder
from .candidate_store import infer
from .data_firewall import Normalization
from .joint_flow import JointFlow
from .tests.test_firewall import observation


def real_body_check():
    # Licensed statistical body assets are not neural network checkpoints.
    from smplx.lbs import lbs

    from realtime_app.learning_flow_mesh.body import BodyDecoder
    from realtime_app.learning_flow_mesh.rotations import matrix_to_six
    from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor

    from .candidate_store import from_parameters
    from .schema import Topology

    cfg = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    decoder = BodyDecoder(**cfg["assets"])
    regressor = torch.tensor(load_coco17_regressor(Path("models/smpl/J_regressor_coco.npy")), dtype=torch.float32)
    surface = SurfaceDecoder(decoder, regressor)
    # Asymmetric candidates expose beta/T mixing. Nonidentity local rotations.
    gen = torch.Generator().manual_seed(27)
    axis = torch.randn(3, 52, 3, generator=gen) * .08
    skew = torch.zeros(3, 52, 3, 3)
    skew[..., 0, 1], skew[..., 0, 2] = -axis[..., 2], axis[..., 1]
    skew[..., 1, 0], skew[..., 1, 2] = axis[..., 2], -axis[..., 0]
    skew[..., 2, 0], skew[..., 2, 1] = -axis[..., 1], axis[..., 0]
    rotations = torch.matrix_exp(skew)
    beta = torch.randn(3, 10, generator=gen) * .2
    translation = torch.tensor([[.1, .2, 3.], [-.1, .1, 2.8], [0., -.1, 3.2]])
    raw = torch.cat((matrix_to_six(rotations).flatten(1), translation, beta), -1)
    store = from_parameters(raw, Topology(), ids=["a", "b", "c"], seeds=[7, 17, 27],
                            current_observed=True, provenance={"source": "analytic_fixture"}, resources={})
    decoded = surface(store.rotations, store.translation, store.beta)
    layer = decoder.layer
    # Independent direct LBS invocation bypasses SurfaceDecoder and BodyDecoder.forward.
    vertices, _ = lbs(beta, store.rotations, layer.v_template, layer.shapedirs,
                      layer.posedirs, layer.J_regressor, layer.parents, layer.lbs_weights, pose2rot=False)
    rebuilt = vertices + translation[:, None]
    error = float((decoded["vertices"] - rebuilt).abs().max())
    assert error < 1e-5
    for i in range(3):
        selected = store.select(i)
        assert torch.equal(selected["beta"], beta[i])
        assert torch.equal(selected["translation"], translation[i])
    return {"status": "passed", "vertices": list(decoded["vertices"].shape),
            "faces": list(decoded["faces"].shape), "direct_lbs_max_error_m": error,
            "operator": "fixed17x6890_surface_regressor", "assets": decoder.asset_paths,
            "boundary": "random parameter internal decoding consistency, not accuracy"}


def run_s1(*, real_body=False):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = JointFlow().eval()
    normal = Normalization.fixture(model.topology)
    with torch.no_grad():
        store, state = infer(model, normal, observation(), k=32, steps=8)
    result = {"stage": "S1", "scope": "random_initialized_engineering_only", "torch": torch.__version__,
              "platform": platform.platform(), "parameters": sum(p.numel() for p in model.parameters()),
              "joint_dimension": 325, "tangent_dimension": 169,
              "regression_parameters": sum(p.numel() for p in model.parameters()),
              "candidates": len(store.raw), "numerical": int(store.numerical.sum()),
              "resources": store.resources, "temporal_state_keys": list(state.to_dict()),
              "training_executed": False, "checkpoint_loaded": False,
              "normalization": normal.source}
    if real_body:
        result["real_body"] = real_body_check()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-body", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_s1(real_body=args.real_body)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
