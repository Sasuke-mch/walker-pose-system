"""Bounded conditioning and encoder diagnostics without training or extra models."""
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


def encoder_dimension_contract():
    """Narrow fixed-slot input family, not trained information sufficiency."""
    from dataclasses import replace

    import torch

    from .state_encoder import StateEncoder
    from .tests.test_firewall import observation

    torch.set_num_threads(1)
    dtype = torch.float64
    j = torch.arange(17, dtype=dtype)
    left = torch.stack((120+19*j, 100+11*(j % 7)), -1)
    xy = torch.stack((left, left+torch.tensor([-8., 6.], dtype=dtype)))
    score = torch.stack((.4+.02*j, .75-.015*j))
    fixture = observation()
    base = replace(fixture, xy=xy, confidence=score, origins=fixture.origins.double(),
                   directions=fixture.directions.double(), geometry=torch.zeros_like(fixture.geometry)).validate()
    packed = torch.cat((xy.flatten()/1000, score.flatten()))
    control_cases = []
    features = torch.cat((xy.reshape(34, 2)/1000, score.reshape(34, 1)), -1)
    for name in ("full", "sparse", "zero_confidence", "empty"):
        admitted = torch.ones(34, dtype=torch.bool)
        values = features.clone()
        if name == "sparse":
            admitted[:] = False
            admitted[[0, 4, 16, 19, 33]] = True
            values[~admitted] = torch.nan
        elif name == "zero_confidence":
            values[:, 2] = 0
        elif name == "empty":
            admitted[:] = False
            values[:] = torch.nan
        # Each identity routes three values to its own coordinates. Adding1 to
        # score distinguishes admitted score0 from a missing slot's empty block.
        routed = torch.zeros(34, 102, dtype=dtype)
        indices = admitted.nonzero().flatten()
        if len(indices):
            coordinates = 3*indices[:, None]+torch.arange(3)
            routed[indices[:, None], coordinates] = values[admitted]+torch.tensor([0., 0., 1.], dtype=dtype)
            pooled = routed[admitted].mean(0).reshape(34, 3)
        else:
            pooled = torch.zeros(34, 3, dtype=dtype)
        recovered_mask = pooled[:, 2] > 0
        recovered = pooled*recovered_mask.sum()-torch.tensor([0., 0., 1.], dtype=dtype)
        error = float((recovered[admitted]-values[admitted]).abs().max()) if admitted.any() else 0.
        control_cases.append({"case": name, "admitted_count": int(admitted.sum()),
                              "mask_recovered": bool(torch.equal(recovered_mask, admitted)),
                              "admitted_value_max_error": error,
                              "current_observation_available": bool(admitted.any())})
    permutation = torch.arange(34)
    permutation[0], permutation[16] = 16, 0
    swapped = features[permutation]
    permutation_free_collision = bool(torch.allclose(features.mean(0), swapped.mean(0), atol=1e-15, rtol=0))
    routed_identity_distinguishes = bool((features-swapped).abs().max() > 0)
    records = []
    for seed in (7, 17, 27):
        torch.manual_seed(seed)
        model = StateEncoder().double().eval()
        def context(value, model=model):
            current = replace(base, xy=value[:68].reshape(2, 17, 2)*1000,
                              confidence=value[68:].reshape(2, 17))
            return model(current, state=None)[0]
        jacobian = torch.autograd.functional.jacobian(context, packed, vectorize=True)
        singular = torch.linalg.svdvals(jacobian)
        direction = torch.randn(102, dtype=dtype, generator=torch.Generator().manual_seed(19))
        direction = direction/direction.norm()
        with torch.no_grad():
            central = (context(packed+1e-4*direction)-context(packed-1e-4*direction))/(2e-4)
        expected = jacobian@direction
        relative = float((expected-central).norm()/expected.norm())
        finite = bool(torch.isfinite(jacobian).all() and torch.isfinite(singular).all())
        records.append({"seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
                        "jacobian_shape": list(jacobian.shape), "finite": finite,
                        "singular_values": singular.tolist(),
                        "rank_sensitivity": {str(rtol): int((singular > rtol*singular[0]).sum())
                                             for rtol in (1e-8, 1e-10, 1e-12, 1e-14)},
                        "smin_over_smax": float(singular[-1]/singular[0]),
                        "FD_direction_step_standardized": 1e-4, "FD_direction_relative_error": relative,
                        "direction_check": finite and relative <= 1e-4})
    return {"date": "2026-10-12", "scope": "D02 pooling/dimension hypothesis; current-frame fixed102D input family",
            "code": "audit_conditioning.encoder_dimension_contract and corrected state_encoder.StateEncoder",
            "device": "cpu", "dtype": "float64", "xy": xy.tolist(), "confidence": score.tolist(),
            "family": "rawxy/1000(68)+score(34); fixed mask/rig/geometry=False and initial temporal state",
            "control_width": 102, "context_width": 128, "identity_routing_is_model_change": False,
            "control_cases": control_cases, "permutation_free_mean_collision": permutation_free_collision,
            "routed_identity_distinguishes_slot_swap": routed_identity_distinguishes,
            "actual_encoder_records": records,
            "numeric_checks_passed": all(r["mask_recovered"] and r["admitted_value_max_error"] <= 1e-12
                                         for r in control_cases) and permutation_free_collision
                                     and routed_identity_distinguishes and all(r["direction_check"] for r in records),
            "boundary": "Identity routing is an analytic control, not a proposed trained encoder. Local rank at random weights neither certifies global information sufficiency nor proves effect loss; ray/history/calibration inputs and data distribution excluded",
            "training": False, "checkpoint_loaded": False, "architecture_changed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-dimension", action="store_true",
                        help="only bounded fixed-slot encoder dimension/sensitivity checks; no surface/J/noise audit")
    args = parser.parse_args()
    result = encoder_dimension_contract() if args.encoder_dimension else audit()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
