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
