"""Scalar counterexamples for curvature-only information and reused likelihood."""
import argparse
import json
from fractions import Fraction
from pathlib import Path


def noise_contract():
    import torch

    from .geometry_information import local_information
    from .geometry_observation import NoiseModel
    from .information_encoding import information_features
    from .state_encoder import StateEncoder
    from .tangent import TangentScale
    from .tests.test_aggregate import rig, store_fixture
    from .tests.test_firewall import observation
    from .tests.test_information import rigid_surface

    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = StateEncoder().eval()
    obs, store = observation(), store_fixture()
    contexts, matrices, jacobians, encodings = [], [], [], []
    for sigma in (1., 2.):
        with torch.no_grad():
            contexts.append(model(obs)[0])
        info = local_information(store, obs, rig(), NoiseModel.proxy(sigma),
                                 TangentScale.fixture(store.topology), rigid_surface)
        assert info.matrix is not None, info.reasons
        matrices.append(info.matrix)
        jacobians.append(info.jacobian)
        blocks, shape = information_features(info.matrix, 52)
        encodings.append(torch.cat((blocks.flatten(), shape)))
    assert torch.equal(contexts[0], contexts[1])
    assert torch.equal(jacobians[0], jacobians[1])
    assert torch.allclose(matrices[0]/4, matrices[1], atol=1e-12, rtol=1e-12)
    assert not torch.equal(encodings[0], encodings[1])
    return {"operator": "actual local_information on analytic rigid_surface and artificial rig",
            "context_dtype": "float32", "geometry_dtype": "float64", "device": "cpu",
            "sigmas_px": [1, 2], "G0_context_identical": True, "J_identical": True,
            "information_quarter_max_error": float((matrices[0]/4-matrices[1]).abs().max()),
            "G1_feature_max_difference": float((encodings[0]-encodings[1]).abs().max()),
            "scalar_posterior": {"model": "prior N(0,1), y=x+eta, observed y=1",
                                 "variance1": {"mean": .5, "variance": .5},
                                 "variance4": {"mean": .2, "variance": .8}},
            "conclusion": "G0 has no explicit external precision input; G1 receives noise through information; a variable-noise comparison confounds conditioning with generator count",
            "limit": "fixed-noise models may encode noise in weights; this is not proof G1 learns calibrated posterior or that confidence identifies noise"}


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
        "date": "2026-10-12", "kind": "scalar_counterexamples_and_actual_noise_conditioning_audit",
        "noise_input_contract": noise_contract(),
        "curvature_only": {"model": "y=x+eta, W=1, anchor=0", "cases": cases,
                           "conclusion": "curvature alone has no residual direction; current context also contains observations, so this is not a full G1 indistinguishability claim"},
        "likelihood_reuse": {
            "prior": "N(0,1)", "measurement": 1, "noise_variance": 1,
            "once": {"mean": float(once_mean), "variance": float(once_variance)},
            "twice": {"mean": float(twice_mean), "variance": float(twice_variance)},
            "variance_relative_change": float(twice_variance/once_variance-1),
            "conclusion": "ideal observation-conditional posterior multiplied by the same likelihood changes its target; this is not evidence that present A6 decision risk reweights the generator"},
        "boundary": "random untrained encoder only; no generator sampling, data replay, training, posterior calibration or measured runtime improvement",
        "next": "separate fixed-noise architecture ablation from variable-noise conditioning ablation before any generator changes"
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
