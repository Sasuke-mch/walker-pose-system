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


def test_real_clock_adapter_never_reads_xyz_labels(tmp_path):
    frames = [{"index": i, "pair_id": 1157+i,
               "left_host_return_timestamp_ns": 1000000000+i*100000000,
               "right_host_return_timestamp_ns": 1020000000+i*100000000} for i in range(2)]
    shape = (2, 2, 17)
    mask = np.ones(shape, bool)
    mask[1] = False
    reasons = np.array([json.dumps([[""]*17]*2)]*2)
    source = tmp_path / "inputs.npz"
    np.savez(source, frame_ids=np.arange(2), original_pair_ids=np.arange(1157, 1159),
             raw_xy=np.ones((*shape, 2)), confidence=np.ones(shape), observation_mask=mask,
             geometry_mask=mask, strict_gate=mask[:, 0], selected_people=np.array([[0, 0], [-1, -1]]),
             strict_reasons_json=np.array([json.dumps([""]*17)]*2),
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
    assert not observations[1].admitted().any() and provenance["strict_points"] == 17
    assert "strict_xyz_m_diagnostic" not in vars(observations[0])


def test_validation_lock_and_missing_evidence_gates():
    policy = lock_policy([0, 1, 2, 3], split="validation", lambda0=1, lambda_g=.1, iqr_fraction=.05, source="noise_calibration+validation")
    assert policy["policy"]["delta"] == pytest.approx(.075)
    with pytest.raises(ValueError):
        lock_policy([0, 1], split="test", lambda0=1, lambda_g=.1, iqr_fraction=.05, source="test")
    gates = preregistered_gates({"full_pipeline_p95_ms": 368})
    assert gates["full_pipeline_p95_ms"] == "fail" and gates["accuracy_claim"] == "blocked"
