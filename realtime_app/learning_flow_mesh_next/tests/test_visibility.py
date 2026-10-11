import numpy as np
import torch

from ..information_encoding import fixed_probes
from ..visibility import cross_visibility, restricted_information


def test_visibility_matches_independent_least_squares_reconstruction():
    rng = np.random.default_rng(11)
    A = rng.normal(size=(30, 79))
    matrix = torch.from_numpy(A.T @ A)
    report = cross_visibility(matrix, 22)
    parts = [slice(3*i, 3*i+3) for i in range(23)] + [slice(69, 79)]
    captured, total = 0., 0.
    for i, part in enumerate(parts):
        outside = np.ones(79, dtype=bool)
        outside[part] = False
        C = matrix.numpy()[part][:, outside]
        P = fixed_probes(79, 10 if i == 23 else 3).numpy()[outside]
        coef = np.linalg.lstsq(P, C.T, rcond=1e-12)[0]
        captured += np.square(P @ coef).sum()
        total += np.square(C).sum()
    assert abs(report['projectable_cross_energy_fraction']-captured/total) < 1e-12
    assert 0 <= report['projectable_cross_energy_fraction'] <= 1
    assert report['raw_sketch_energy_ratio_not_retention'] > report['projectable_cross_energy_fraction']
    scaled = cross_visibility(matrix*4, 22)
    assert abs(scaled['projectable_cross_energy_fraction']-report['projectable_cross_energy_fraction']) < 1e-12
    assert cross_visibility(torch.eye(79), 22)['projectable_cross_energy_fraction'] is None


def test_restricted_J_row_order_and_nonisotropic_precision():
    J = torch.arange(68*3, dtype=torch.float64).reshape(68, 3)/100
    mask = torch.zeros(2, 17, dtype=torch.bool)
    mask[0, 3] = mask[1, 8] = True
    precision = torch.eye(2, dtype=torch.float64).expand(2, 17, 2, 2).clone()
    precision[0, 3] = torch.tensor([[2., .5], [.5, 1.]])
    precision[1, 8] = torch.tensor([[1., -.2], [-.2, 3.]])
    expected = J[6:8].T @ precision[0, 3] @ J[6:8]
    expected += J[50:52].T @ precision[1, 8] @ J[50:52]
    assert torch.allclose(restricted_information(J, mask, precision), expected, atol=1e-12)
    assert restricted_information(J, mask & False, precision) is None
