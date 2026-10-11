import numpy as np

from ..structure import self_intersection_diagnostic


def test_crossing_coplanar_and_budget_not_false_pass():
    vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.],
                         [.5, .5, -1.], [.5, .5, 1.], [1., .5, 0.],
                         [.2, .2, 0.], [.4, .2, 0.], [.2, .4, 0.]])
    faces = np.arange(9).reshape(3, 3)
    report = self_intersection_diagnostic(vertices, faces)
    assert report["intersection_pair_count_lower_bound"] >= 2
    assert report["complete_for_declared_nonadjacent_test"]
    bounded = self_intersection_diagnostic(vertices, faces, max_pairs=1)
    assert bounded["budget_exhausted"] and not bounded["complete_for_declared_nonadjacent_test"]
