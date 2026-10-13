import json

import numpy as np
import pytest

from ..acceptance import lock_policy, preregistered_gates
from ..replay import load_saved_observations


def test_saved_relative_calibration_and_utf8_bom(tmp_path):
    import torch

    from ..geometry_observation import load_rig

    intrinsics = {"K": [[300, 0, 320], [0, 300, 240], [0, 0, 1]], "D": [0, 0, 0, 0]}
    for i in range(2):
        (tmp_path/f"cam{i}.json").write_text(json.dumps(intrinsics), encoding="utf-8-sig")
    stereo = {"model": "opencv_fisheye_stereo", "transform_convention": "X_cam1 = R_cam0_to_cam1 * X_cam0 + T_cam0_to_cam1",
              "cam0_intrinsics": "cam0.json", "cam1_intrinsics": "cam1.json", "image_size": [640, 480],
              "R_cam0_to_cam1": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "T_cam0_to_cam1_mm": [-200, 0, 0]}
    path = tmp_path/"stereo.json"
    path.write_text(json.dumps(stereo), encoding="utf-8-sig")
    cameras = load_rig(path, camera_ids=("left_pnp", "right_pnp"))
    assert cameras[1].T.equal(torch.tensor([-.2, 0., 0.], dtype=torch.float64))


@pytest.mark.parametrize("support", ["stereo", "left_only", "geometry_rejected"])
def test_real_clock_adapter_preserves_2d_support_and_never_reads_xyz_labels(tmp_path, support):
    frames = [{"index": i, "pair_id": 1157+i,
               "left_host_return_timestamp_ns": 1000000000+i*100000000,
               "right_host_return_timestamp_ns": 1020000000+i*100000000} for i in range(2)]
    shape = (2, 2, 17)
    mask = np.ones(shape, bool)
    mask[1] = False
    if support == "left_only":
        mask[0, 1] = False
    strict = np.zeros((2, 17), bool)
    strict[0] = support == "stereo"
    geometry = np.broadcast_to(strict[:, None], shape).copy()
    xy = np.ones((*shape, 2))
    xy[~mask] = np.nan
    confidence = np.ones(shape)
    confidence[~mask] = np.nan
    confidence[0, 0, 0] = 0  # Retain detector score as a feature, not a covariance or new gate.
    strict_reason = "" if support == "stereo" else "source_stereo_rejected"
    reasons = np.array([json.dumps([[""]*17]*2)]*2)
    source = tmp_path / "inputs.npz"
    np.savez(source, frame_ids=np.arange(2), original_pair_ids=np.arange(1157, 1159),
             raw_xy=xy, confidence=confidence, observation_mask=mask,
             geometry_mask=geometry, strict_gate=strict, selected_people=np.array([[0, 0], [-1, -1]]),
             strict_reasons_json=np.array([json.dumps([strict_reason]*17)]*2),
             quality_flags_json=np.full((2, 17), "[]"), point_reasons_json=reasons, ray_reasons_json=reasons,
             ray_origins_m=np.zeros((*shape, 3)), ray_directions_cam0=np.tile([0., 0., 1.], (*shape, 1)),
             strict_xyz_m_diagnostic=np.array([None], dtype=object))
    mapping, protocol = tmp_path / "frames.json", tmp_path / "protocol.json"
    mapping.write_text(json.dumps(frames), encoding="utf-8")
    metadata = {"metadata": {"calibration_camera_mapping": {"camera_registry_resolution": {
        "left": {"expected_instance_id": "PnP-L"}, "right": {"expected_instance_id": "PnP-R"}}},
        "timestamp_type": "host_read_return_perf_counter_ns", "timestamp_warning": "not exposure"}}
    protocol.write_text(json.dumps(metadata), encoding="utf-8")
    observations, provenance = load_saved_observations(source, mapping, protocol)
    assert observations[0].timestamp == pytest.approx(1.01)
    assert observations[1].timestamp-observations[0].timestamp == pytest.approx(.1)
    assert observations[0].view_timestamps == (1., 1.02)
    assert np.array_equal(observations[0].admitted().numpy(), mask[0])
    assert np.array_equal(observations[0].quality.numpy(), mask[0])
    assert np.array_equal(observations[0].geometry.numpy(), geometry[0])
    assert observations[0].confidence[0, 0] == 0
    np.testing.assert_equal(observations[0].xy.numpy(), xy[0])
    assert json.loads(observations[0].rejection_reasons[0][0]) == {
        "point": "", "ray": "", "strict": strict_reason, "quality_flags": []}
    assert not observations[1].views.any() and not observations[1].admitted().any()
    assert provenance["strict_points"] == int(strict.sum())
    assert "strict_xyz_m_diagnostic" not in vars(observations[0])


def test_validation_lock_and_missing_evidence_gates():
    policy = lock_policy([0, 1, 2, 3], split="validation", lambda0=1, lambda_g=.1, iqr_fraction=.05, source="noise_calibration+validation")
    assert policy["policy"]["delta"] == pytest.approx(.075)
    with pytest.raises(ValueError):
        lock_policy([0, 1], split="test", lambda0=1, lambda_g=.1, iqr_fraction=.05, source="test")
    gates = preregistered_gates({"full_pipeline_p95_ms": 368})
    assert gates["full_pipeline_p95_ms"] == "fail" and gates["accuracy_claim"] == "blocked"
