"""Reproducible no-training audits; no checkpoint loading or downloads."""
import argparse
import json
import platform
from dataclasses import replace
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


def run_s2(cache_path=None):
    from realtime_app.learning_flow_mesh.body import BodyDecoder
    from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor

    from .aggregate import decide
    from .geometry_observation import NoiseModel, project
    from .tangent import TangentScale
    from .tests.test_aggregate import rig, store_fixture

    torch.set_num_threads(1)
    cfg = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    decoder = BodyDecoder(**cfg["assets"]).double()
    surface = SurfaceDecoder(decoder, torch.tensor(load_coco17_regressor(
        Path("models/smpl/J_regressor_coco.npy")), dtype=torch.float64))
    store = store_fixture()
    cameras = rig()
    points = surface(store.rotations, store.translation, store.beta)["coco17"]
    obs = replace(observation(), xy=torch.stack([project(points[1], c)[0] for c in cameras]))
    if cache_path:
        store.save(cache_path)
    results = {}
    for method in ("A0", "A1", "A2", "A3", "A4", "A5"):
        result = decide(store, method, TangentScale.fixture(store.topology),
                        observation=obs, cameras=cameras, noise=NoiseModel.proxy(), surface=surface, lambda_g=.1)
        results[method] = {"selected_id": result.selected_id, "reasons": result.reasons,
                           "diagnostic_keys": list(result.diagnostic),
                           "costs": [float(v) if torch.isfinite(v) else None for v in result.costs],
                           "nonfinite_cost_count": int((~torch.isfinite(result.costs)).sum())}
    return {"stage": "S2", "scope": "fixed analytic candidates real licensed surface decoder",
            "results": results, "noise": "proxy-only", "cache": str(cache_path) if cache_path else None,
            "distribution_weights": "raw_equal_weight", "training_executed": False,
            "checkpoint_loaded": False, "boundary": "interface decisions only, no performance comparison",
            "A3_adaptation": "author joint top-k rule with admitted robust-whitened raw-fisheye costs, own T, no artificial zero root; engineering weights only"}


def run_s3(cache_dir=None):
    from realtime_app.learning_flow_mesh.body import BodyDecoder
    from realtime_app.learning_flow_mesh.rotations import matrix_to_six
    from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor

    from .geometry_observation import NoiseModel, project
    from .pipeline import TwoStageInference
    from .schema import Topology
    from .tangent import TangentScale
    from .tests.test_aggregate import rig

    torch.set_num_threads(1)
    torch.manual_seed(7)
    g0 = JointFlow().double().eval()
    torch.manual_seed(17)
    g1 = JointFlow().double().eval()
    cfg = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    decoder = BodyDecoder(**cfg["assets"]).double()
    surface = SurfaceDecoder(decoder, torch.tensor(load_coco17_regressor(
        Path("models/smpl/J_regressor_coco.npy")), dtype=torch.float64))
    cameras = rig()
    # Artificial constants only: not fitted training statistics or an accuracy target.
    mean = torch.cat((matrix_to_six(torch.eye(3, dtype=torch.float64)).repeat(52),
                      torch.tensor([0., 0., 3.], dtype=torch.float64), torch.zeros(10, dtype=torch.float64)))
    normal = Normalization(mean, torch.ones(325, dtype=torch.float64)*.03, Topology(),
                           "analytic_fixture_identity_R_Tz3_scale0.03_not_train_statistics")
    scale = TangentScale.fixture(g0.topology)
    identity = torch.eye(3, dtype=torch.float64).expand(1, 52, 3, 3)
    points = surface(identity, mean[312:315][None], mean[-10:][None])["coco17"][0]
    obs = replace(observation(), xy=torch.stack([project(points, c)[0] for c in cameras]))
    pipeline = TwoStageInference(g0, g1, normal, scale, surface, cameras, NoiseModel.proxy())
    with torch.no_grad():
        generated, anchor_candidates, information, state, resources = pipeline(obs)
    if cache_dir:
        generated.save(cache_dir / "G1.npz")
        anchor_candidates.save(cache_dir / "G0.npz")
    assert not any(p.requires_grad for p in g0.parameters())
    return {"stage": "S3", "scope": "random two-generator real surface numerical interface",
            "G0_parameters": sum(p.numel() for p in g0.parameters()),
            "G1_parameters": sum(p.numel() for p in g1.parameters()),
            "G0_numerical": int(anchor_candidates.numerical.sum()), "G1_numerical": int(generated.numerical.sum()),
            "anchor_id": information.anchor_id, "information_available": information.matrix is not None,
            "J_shape": list(information.jacobian.shape) if information.jacobian is not None else None,
            "noise": information.noise_source, "calibrated": information.calibrated,
            "diagnostics": information.diagnostics, "resources": resources,
            "state_keys": list(state.to_dict()), "cache_dir": str(cache_dir) if cache_dir else None,
            "training_executed": False, "checkpoint_loaded": False,
            "boundary": "local numerical information and interface only; neither posterior nor G1 efficacy"}


def run_s4(cache_dir):
    from realtime_app.learning_flow_mesh.body import BodyDecoder
    from realtime_app.pose_app.smpl_coco_observation import load_coco17_regressor

    from .aggregate import decide
    from .candidate_store import CandidateStore
    from .geometry_information import local_information
    from .geometry_observation import NoiseModel, project
    from .information_risk import DecisionState, RiskPolicy, decide_information
    from .tangent import TangentScale
    from .tests.test_aggregate import rig

    if cache_dir is None:
        raise ValueError("S4 requires a previously fixed S3 candidate cache directory")
    torch.set_num_threads(1)
    g0, store = CandidateStore.load(cache_dir / "G0.npz"), CandidateStore.load(cache_dir / "G1.npz")
    cfg = json.loads(Path("realtime_app/learning_flow_mesh/config.json").read_text(encoding="utf-8"))
    surface = SurfaceDecoder(BodyDecoder(**cfg["assets"]).double(), torch.tensor(
        load_coco17_regressor(Path("models/smpl/J_regressor_coco.npy")), dtype=torch.float64))
    cameras, noise, scale = rig(), NoiseModel.proxy(), TangentScale.fixture(store.topology)
    points = surface(torch.eye(3, dtype=torch.float64).expand(1, 52, 3, 3),
                     torch.tensor([[0., 0., 3.]], dtype=torch.float64), torch.zeros(1, 10, dtype=torch.float64))["coco17"][0]
    obs = replace(observation(), xy=torch.stack([project(points, c)[0] for c in cameras]))
    information = local_information(g0, obs, cameras, noise, scale, surface)
    policies = RiskPolicy(lambda0=1, lambda_g=.1, delta=0, source="analytic_fixture_not_validation_locked")
    kwargs = {"observation": obs, "cameras": cameras, "noise": noise, "surface": surface}
    a4 = decide(store, "A4", scale)
    a5 = decide(store, "A5", scale, lambda_g=.1, **kwargs)
    a6, _ = decide_information(store, information, scale, policies, **kwargs)
    previous = DecisionState(obs.identity, obs.sequence, -.1, store.rotations[0],
                             store.translation[0], store.beta[0], "analytic_previous")
    a7, _ = decide_information(store, information, scale, policies, method="A7", previous=previous, **kwargs)
    assert a6.selected_id == a7.selected_id  # delta0 is the no-time-effect control.
    order = torch.arange(len(store.raw)-1, -1, -1)
    shuffled, _ = decide_information(store.permuted(order), information, scale, policies, **kwargs)
    assert shuffled.selected_id == a6.selected_id
    return {"stage": "S4", "cache_dir": str(cache_dir), "generator_executed": False,
            "results": {d.method: {"selected_id": d.selected_id, "risk_gap": d.risk_gap,
                        "ambiguity": d.ambiguity, "reasons": d.reasons} for d in (a4, a5, a6, a7)},
            "permutation_same_non_tie_selection": True, "A7_delta0_same_A6": True,
            "risk_policy": vars(policies), "noise": noise.source,
            "training_executed": False, "checkpoint_loaded": False,
            "boundary": "fixed random candidates engineering decision only; no risk/time improvement claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-body", action="store_true")
    parser.add_argument("--stage", choices=("S1", "S2", "S3", "S4"), default="S1")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "S1":
        report = run_s1(real_body=args.real_body)
    elif args.stage == "S2":
        report = run_s2(args.cache)
    elif args.stage == "S3":
        report = run_s3(args.cache_dir)
    else:
        report = run_s4(args.cache_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
