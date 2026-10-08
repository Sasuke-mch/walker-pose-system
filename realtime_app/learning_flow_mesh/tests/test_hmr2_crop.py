"""R16 author preprocessing against independent affine/color convention."""
import numpy as np

from realtime_app.learning_flow_mesh.reproduction.check_hmr2_crop import (
    independent_crop,
    load_dataset,
    prepare,
)


def test_author_asymmetric_crop_matches_independent_rgb_normalization():
    cls, cfg = load_dataset()
    y, x = np.indices((240, 180))
    image = np.stack((x, y, (x + y) % 255), axis=-1).astype(np.uint8)
    box = np.array([[20., 30., 140., 190.]], dtype=np.float32)
    item, _logs = prepare(image, box, cfg, cls)
    assert np.array_equal(item["box_center"], [80., 110.])
    assert abs(item["box_size"] - 160.) < 2e-5
    expected = independent_crop(image, item["box_center"], item["box_size"], cfg)
    np.testing.assert_allclose(item["img"], expected, atol=1e-5, rtol=0)
    assert np.array_equal(item["img_size"], [180., 240.])


def test_author_outside_crop_returns_normalized_black_without_rejection():
    cls, cfg = load_dataset()
    image = np.full((80, 100, 3), 127, dtype=np.uint8)
    item, _logs = prepare(image, np.array([[500., 500., 560., 580.]], dtype=np.float32), cfg, cls)
    black_rgb = -np.array(cfg.MODEL.IMAGE_MEAN) / np.array(cfg.MODEL.IMAGE_STD)
    np.testing.assert_allclose(item["img"], np.broadcast_to(black_rgb[:, None, None], item["img"].shape), atol=1e-6, rtol=0)
    assert np.isfinite(item["img"]).all()
    # Dataset carries personid even though supplied box contains no image pixels.
    assert item["personid"] == 0


def test_empty_author_boxes_have_no_manufactured_person():
    cls, cfg = load_dataset()
    assert len(cls(cfg, np.zeros((80, 100, 3), dtype=np.uint8), np.zeros((0, 4)))) == 0
