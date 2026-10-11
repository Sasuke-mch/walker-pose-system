from dataclasses import replace

import pytest
import torch

from ..geometry_information import local_information
from ..geometry_observation import NoiseModel
from ..information_risk import RiskPolicy, information_metric
from ..sensitivity import reweight_information
from ..tangent import TangentScale
from .test_aggregate import rig, store_fixture
from .test_firewall import observation
from .test_information import rigid_surface


def test_noise_scale_confounding_and_compensated_metric():
    obs = observation()
    store = store_fixture()
    info = local_information(store, obs, rig(), NoiseModel.proxy(10),
                             TangentScale.fixture(store.topology), rigid_surface)
    matrix = info.matrix.clone()
    high_precision = reweight_information(info, obs, NoiseModel.proxy(5))
    assert torch.allclose(high_precision.matrix, matrix*4, atol=1e-12)
    base_metric = information_metric(info, RiskPolicy())
    assert not torch.allclose(information_metric(high_precision, RiskPolicy()), base_metric)
    assert torch.allclose(information_metric(high_precision, RiskPolicy(lambda0=4)), base_metric, atol=1e-12)
    assert torch.equal(matrix, info.matrix) and high_precision.jacobian is info.jacobian
    assert 'shape_schur_eigenvalues' not in high_precision.diagnostics
    with pytest.raises(ValueError, match='provenance'):
        reweight_information(info, replace(obs, frame=99), NoiseModel.proxy())


def test_anisotropic_noise_reweight_matches_whitened_quadratic():
    obs = observation()
    store = store_fixture()
    info = local_information(store, obs, rig(), NoiseModel.proxy(),
                             TangentScale.fixture(store.topology), rigid_surface)
    block = torch.tensor([[2., .5], [.5, 1.]], dtype=torch.float64)
    noise = NoiseModel(block.expand(2, 17, 2, 2).clone(), 'anisotropic_analytic_fixture')
    reweighted = reweight_information(info, obs, noise)
    # Energy along a physical translation direction, independently using L^T Jv.
    direction = torch.zeros(169, dtype=torch.float64)
    direction[156:159] = torch.tensor([.3, -.2, .1])
    pixels = (info.jacobian @ direction).reshape(-1, 2)
    white = pixels @ torch.linalg.cholesky(block)
    expected = white.square().sum()
    assert torch.allclose(direction @ reweighted.matrix @ direction, expected, atol=1e-12)
