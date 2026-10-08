"""Independent controls for original metric units and alignment invariance."""
import ast
from pathlib import Path

import numpy as np

from realtime_app.learning_flow_mesh.reproduction.check_motionbert_wild import (
    compile_nodes,
)


def original_metrics():
    source = Path("third_party/architecture_reproduction/MotionAGFormer/loss/pose3d.py")
    nodes = [n for n in ast.parse(source.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in ("mpjpe", "p_mpjpe", "acc_error")]
    symbols = {"np": np}
    compile_nodes(source, nodes, symbols)
    return symbols


def test_second_difference_acceleration_has_no_sampling_interval():
    fn = original_metrics()["acc_error"]
    prediction = np.zeros((5, 17, 3))
    prediction[..., 0] = (np.arange(5) ** 2)[:, None]
    np.testing.assert_allclose(fn(prediction, np.zeros_like(prediction)), [2, 2, 2])
    # Every-other-frame samples have fourfold second difference: no dt² division.
    np.testing.assert_allclose(fn(prediction[::2], np.zeros_like(prediction[::2])), [8])


def test_perfect_mpjpe_is_valid_zero_error():
    fn = original_metrics()["mpjpe"]
    truth = np.random.default_rng(4).normal(size=(3, 17, 3))
    assert np.array_equal(fn(truth, truth), np.zeros(3))


def test_procrustes_similarity_can_hide_absolute_scale_and_translation():
    functions = original_metrics()
    truth = np.random.default_rng(8).normal(size=(3, 17, 3))
    prediction = truth * 5 + [40, -20, 10]
    assert np.all(functions["mpjpe"](prediction, truth) > 10)
    np.testing.assert_allclose(functions["p_mpjpe"](prediction, truth), 0, atol=1e-12)
