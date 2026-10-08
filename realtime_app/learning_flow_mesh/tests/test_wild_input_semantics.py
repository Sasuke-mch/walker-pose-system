"""Original wild reader is not a causal crop or a frame/person associator."""
import json

import numpy as np
import pytest

from realtime_app.learning_flow_mesh.reproduction.check_motionbert_wild import (
    original_reader,
)


def test_crop_future_dependency_and_zero_confidence_geometry():
    crop = original_reader()["crop_scale"]
    motion = np.zeros((2, 17, 3))
    motion[..., 0] = np.arange(17)
    motion[..., 1] = np.arange(17)[::-1]
    motion[..., 2] = .5
    before = crop(motion, [1, 1])
    changed = motion.copy()
    changed[1, 0, 0] = 100
    assert not np.array_equal(before[0], crop(changed, [1, 1])[0])
    changed[..., 2] = 0
    assert np.count_nonzero(crop(changed, [1, 1])) == 0


def test_negative_confidence_participates_in_crop_and_nan_poisoning():
    crop = original_reader()["crop_scale"]
    motion = np.zeros((1, 17, 3))
    motion[..., 0] = np.arange(17)
    motion[..., 1] = np.arange(17)
    motion[..., 2] = -.5
    output = crop(motion, [1, 1])
    assert np.count_nonzero(output[..., :2]) > 0
    assert np.all(output[..., 2] == -.5)
    motion[0, 0, 0] = np.nan
    assert np.count_nonzero(~np.isfinite(crop(motion, [1, 1]))) > 0


def test_json_order_people_focus_and_empty_selection(tmp_path):
    read = original_reader()["read_input"]
    first = np.zeros((26, 3))
    first[..., 0] = 40
    first[..., 1] = 60
    first[..., 2] = .8
    second = first.copy()
    second[..., 0] = 80
    records = [{"idx": 7, "image_id": "frame9", "keypoints": first.reshape(-1).tolist()},
               {"idx": 9, "image_id": "frame0", "keypoints": second.reshape(-1).tolist()}]
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    out = read(path, (200, 100), None, None)
    np.testing.assert_allclose(out[:, 0, 0], [-1.2, -.4])
    assert read(path, (200, 100), None, 7).shape == (1, 17, 3)
    with pytest.raises(ValueError):
        read(path, (200, 100), None, 0)
