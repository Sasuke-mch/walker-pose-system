from dataclasses import replace

import pytest
import torch

from ..data_firewall import Labels, Normalization, validate_manifest
from ..schema import Observation, Topology


def observation(frame=0, timestamp=0.0):
    return Observation(torch.ones(2, 17, 2), torch.ones(2, 17),
                       torch.ones(2, 17, dtype=torch.bool), torch.ones(2, 17, dtype=torch.bool),
                       torch.ones(2, 17, dtype=torch.bool), torch.ones(2, dtype=torch.bool),
                       torch.zeros(2, 17, 3), torch.tensor([0., 0., 1.]).expand(2, 17, 3),
                       tuple(tuple("" for _ in range(17)) for _ in range(2)),
                       ("PnP_LEFT_fixture", "PnP_RIGHT_fixture"), "person", "sequence",
                       frame, timestamp, "analytic_fixture")


def labels(nodes=52):
    return Labels(torch.eye(3).expand(2, nodes, 3, 3), torch.zeros(2, 3), torch.zeros(2, 10),
                  torch.ones(2, nodes, dtype=torch.bool), torch.ones(2, dtype=torch.bool),
                  torch.ones(2, dtype=torch.bool), Topology(nodes), "analytic_fixture", "engineering_fixture")


def manifest():
    return {"source_path": "pending", "license_status": "missing", "family": "SMPL-H", "gender": "male",
            "beta_basis": "local_asset_basis_pending_audit", "beta_dimensions": 10,
            "rotation_convention": "local_matrix_packed_columns6d", "hand_mean": "flat",
            "dmpl": "excluded_explicitly", "label_scope": "pending", "noise_source": "proxy-only",
            "records": [], "truth_status": "no_independent_3d_or_scan"}


def test_no_gt_or_untyped_fields():
    with pytest.raises(TypeError):
        replace(observation(), gt_root=torch.zeros(3))
    with pytest.raises(TypeError):
        Observation(**{**vars(observation()), "betas": torch.zeros(10)})


def test_missing_nan_preserved_present_nan_rejected():
    obs = observation()
    missing = replace(obs, present=torch.zeros_like(obs.present), xy=torch.full_like(obs.xy, torch.nan))
    missing.validate()
    with pytest.raises(ValueError, match="nonfinite"):
        replace(obs, xy=missing.xy).validate()


@pytest.mark.parametrize("nodes,dim,tangent", [(52, 325, 169), (22, 145, 79)])
def test_topology_and_partial_scope(nodes, dim, tangent):
    target = labels(nodes)
    assert target.packed()[0].shape == (2, dim)
    assert target.topology.tangent_dimensions == tangent
    partial = replace(target, beta_mask=torch.zeros(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="incomplete"):
        partial.packed()
    assert torch.isnan(partial.packed(require_complete=False)[0][:, -10:]).all()
    with pytest.raises(ValueError, match="training data"):
        Normalization.fit(target, source_path="fixture")


def test_split_and_family_firewall():
    m = manifest()
    assert not validate_manifest(m)["training_data_ready"]
    with pytest.raises(ValueError, match="SMPL-X"):
        validate_manifest({**m, "family": "SMPL-X"})
    records = [{"subject": "s", "original_action": "a", "sequence": "q", "rig": "r", "source_path": "x", "split": s}
               for s in ("train", "test")]
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest({**m, "records": records})


@pytest.mark.parametrize("nodes", [22, 52])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fitted_statistics_are_frozen_and_roundtrip_declared_origin(nodes, dtype):
    # Artificial tensors with declared train/reference metadata test the
    # contract only; they are not independently acquired reference data.
    target = labels(nodes)
    target = replace(target, rotations=target.rotations.to(dtype), beta=target.beta.to(dtype),
                     translation=torch.tensor([[0., 0., 3.], [.123456789, .05, 3.123456789]],
                                              dtype=dtype, requires_grad=True),
                     split="train", source_kind="fitted_reference")
    expected, _ = target.packed()
    fitted = Normalization.fit(target, source_path="declared_contract_fixture")
    assert torch.equal(fitted.mean, expected.mean(0))
    assert torch.equal(fitted.scale, expected.std(0, unbiased=False).clamp_min(1e-4))
    assert not fitted.mean.requires_grad and not fitted.scale.requires_grad
    assert fitted.mean.grad_fn is None and fitted.scale.grad_fn is None
    assert target.translation.grad is None
    assert fitted.require_training_statistics() is fitted
    restored = Normalization.from_dict(fitted.to_dict())
    assert restored.split == "train" and restored.source_kind == "fitted_reference"
    assert restored.mean.dtype == restored.scale.dtype == dtype
    assert torch.equal(fitted.mean, restored.mean) and torch.equal(fitted.scale, restored.scale)
    assert restored.require_training_statistics() is restored


def test_fixture_and_legacy_statistics_do_not_claim_train_origin():
    fixture = Normalization.fixture(Topology())
    assert fixture.split == "engineering_fixture" and fixture.source_kind == "analytic_fixture"
    current = fixture.to_dict()
    legacy = {key: current[key] for key in ("mean", "scale", "nodes", "source")}
    restored = Normalization.from_dict(legacy)
    assert restored.split == restored.source_kind == "legacy_unverified"
    assert restored.mean.dtype == torch.float32
    assert torch.equal(restored.normalize(torch.ones(325)), fixture.normalize(torch.ones(325)))
    for value in (fixture, restored):
        with pytest.raises(ValueError, match="declared training statistics"):
            value.require_training_statistics()
    with pytest.raises(ValueError, match="provenance"):
        Normalization.from_dict({**legacy, "split": "train"})


@pytest.mark.parametrize("split,kind", [("validation", "fitted_reference"), ("test", "independent_reference"),
                                        ("train", "analytic_fixture"), ("engineering_fixture", "fitted_reference")])
def test_nontrain_and_conflicting_statistics_provenance_rejected(split, kind):
    fixture = Normalization.fixture(Topology())
    with pytest.raises(ValueError, match="provenance"):
        replace(fixture, split=split, source_kind=kind).validate()


def test_gradient_bearing_or_unsupported_statistics_rejected():
    fixture = Normalization.fixture(Topology())
    with pytest.raises(ValueError, match="must be frozen"):
        replace(fixture, mean=fixture.mean.clone().requires_grad_()).validate()
    with pytest.raises(ValueError, match="dtype"):
        Normalization.from_dict({**fixture.to_dict(), "dtype": "float16"})


def test_so3_codec_and_candidate_provenance_bind_statistics_origin():
    from ..candidate_store import infer
    from ..so3_flow import SO3JointFlow

    torch.set_num_threads(1)
    normal = Normalization.fixture(Topology())
    model = SO3JointFlow(normal).eval()
    with torch.no_grad():
        store, _ = infer(model, normal, observation(), k=1, regression=True)
    assert store.provenance["normalization_split"] == "engineering_fixture"
    assert store.provenance["normalization_source_kind"] == "analytic_fixture"
    relabeled = replace(normal, split="legacy_unverified", source_kind="legacy_unverified")
    assert torch.equal(relabeled.mean, normal.mean) and relabeled.source == normal.source
    with pytest.raises(ValueError, match="normalization mismatch"):
        model.validate_normalization(relabeled)


@pytest.mark.parametrize("geometry", ["ambient", "SO3"])
def test_inference_rejects_unfrozen_statistics_before_encoding(geometry):
    from ..candidate_store import infer
    from ..joint_flow import JointFlow
    from ..so3_flow import SO3JointFlow

    normal = Normalization.fixture(Topology())
    model = JointFlow() if geometry == "ambient" else SO3JointFlow(normal)
    unsafe = replace(normal, mean=normal.mean.clone().requires_grad_())
    with pytest.raises(ValueError, match="must be frozen"):
        infer(model, unsafe, observation(), k=1, steps=1)
