"""SO3 diagnostics must ignore equivalent axis-angle wraps."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("body_audit", Path(__file__).parents[1] / "tools/audit_grasp_body_pose_prior.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_pose_drift_uses_rotations_not_axis_angle_subtraction():
    first = np.zeros((2, 63))
    second = first.copy()
    second[:, 2] = 2 * np.pi
    assert audit.rotation_degrees(audit.matrices(first), audit.matrices(second)).max() < 1e-10
    second[:, 2] = np.pi / 2
    angles = audit.rotation_degrees(audit.matrices(first), audit.matrices(second))
    assert np.allclose(angles[:, 0], 90.)
    assert np.allclose(angles[:, 1:], 0.)


def test_pose_diagnostic_rejects_invalid_dimensions_and_nan():
    with pytest.raises(ValueError):
        audit.matrices(np.zeros((1, 66)))
    with pytest.raises(ValueError):
        audit.matrices(np.full((1, 63), np.nan))


def test_unrestricted_full_body_entry_fails_before_loading_or_writing(tmp_path):
    script = Path(__file__).parents[1] / "tools/refine_body_with_constructed_grasp.py"
    arguments = []
    for name in ("source-result", "grasp", "walker-model", "scene-transforms",
                 "contact-labels", "contact-vertex-sets", "left-raw", "right-raw"):
        arguments += ["--" + name, str(tmp_path / "not_loaded")]
    output = tmp_path / "must_not_exist"
    result = subprocess.run([sys.executable, str(script), *arguments,
                             "--output-dir", str(output), "--full-body"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "active VPoser body parameterization and prior are required" in result.stderr
    assert not output.exists()

    overridden = subprocess.run([sys.executable, str(script), *arguments,
        "--output-dir", str(output), "--full-body", "--vposer-dir", str(tmp_path),
        "--pose-reference-result", str(tmp_path/'reference.npz'), "--lock-grasp-orientation"],
        capture_output=True, text=True)
    assert overridden.returncode != 0
    assert "forbids overriding decoded wrist rotations" in overridden.stderr
    assert not output.exists()
