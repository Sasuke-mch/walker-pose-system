"""R16 original HMR2 image dataset to full original forward subpath."""
import argparse
import contextlib
import importlib
import io
import json
import traceback
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from yacs.config import CfgNode

from .check_hmr2_mesh import run as mesh_run
from .run_original import ROOT, namespace

BASE = Path("research_records/research/G20261007_flow_mesh_baseline_v1")


def load_dataset():
    root = ROOT / "HMR2"
    namespace("hmr2", root / "hmr2")
    namespace("hmr2.datasets", root / "hmr2/datasets")
    cls = importlib.import_module("hmr2.datasets.vitdet_dataset").ViTDetDataset
    cfg = CfgNode(yaml.safe_load((root / "hmr2/configs_hydra/experiment/hmr_vit_transformer.yaml").read_text()))
    # Original load_hmr2 lines75..78 applies this to ViT checkpoints; no checkpoint read.
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    return cls, cfg


def prepare(image, boxes, cfg, cls):
    original_image, original_boxes = image.copy(), boxes.copy()
    logs = io.StringIO()
    with contextlib.redirect_stdout(logs):
        dataset = cls(cfg, image, boxes)
        item = dataset[0]
    assert np.array_equal(image, original_image) and np.array_equal(boxes, original_boxes)
    return item, logs.getvalue()


def independent_crop(image, center, size, cfg):
    scale = 256. / size
    transform = np.array([[scale, 0., 128. - center[0] * scale],
                          [0., scale, 128. - center[1] * scale]], dtype=np.float32)
    # This independent affine control is limited to non-antialiased fixtures.
    patch = cv2.warpAffine(image, transform, (256, 256), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rgb = patch[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    mean = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
    std = 255. * np.array(cfg.MODEL.IMAGE_STD)
    for channel in range(3):
        rgb[channel] = (rgb[channel] - mean[channel]) / std[channel]
    return rgb


def run(output):
    protocol = json.loads((BASE / "architecture_r16_protocol.json").read_text(encoding="utf-8"))
    cls, cfg = load_dataset()
    image = cv2.imread(protocol["image"])
    if image is None:
        raise FileNotFoundError(protocol["image"])
    height, width = image.shape[:2]
    boxes = np.array([[0., 0., width, height]], dtype=np.float32)
    item, logs = prepare(image, boxes, cfg, cls)
    expected_size = max(height, width * 256. / 192.)
    assert abs(item["box_size"] - expected_size) < 1e-3
    assert np.array_equal(item["box_center"], [width / 2., height / 2.])
    assert np.array_equal(item["img_size"], [width, height]) and item["personid"] == 0
    # Asymmetric non-antialiased image exposes RGB channel order and affine offset.
    y, x = np.indices((240, 180))
    fixture = np.stack((x % 255, y % 255, (x + y) % 255), axis=-1).astype(np.uint8)
    small, _ = prepare(fixture, np.array([[20., 30., 140., 190.]], dtype=np.float32), cfg, cls)
    independent = independent_crop(fixture, small["box_center"], small["box_size"], cfg)
    affine_error = float(np.abs(independent - small["img"]).max())
    assert affine_error < 1e-5
    invalid = {}
    for name, box in {"outside": [width + 500, height + 500, width + 600, height + 700],
                      "zero_size": [20, 20, 20, 20], "NaN_box": [np.nan, 20, 100, 100]}.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                value, _ = prepare(image, np.array([box], dtype=np.float32), cfg, cls)
                invalid[name] = {"status": "returned", "finite_img": bool(np.isfinite(value["img"]).all()),
                                 "finite_box_size": bool(np.isfinite(value["box_size"])),
                                 "channel_spatial_std": [float(v) for v in value["img"].std(axis=(1, 2))]}
            except Exception as error:  # noqa: BLE001 -- retain original input failure
                invalid[name] = {"status": "exception", "error": f"{type(error).__name__}: {error}"}
            invalid[name]["warnings"] = [str(v.message) for v in caught]
    empty = cls(cfg, image, np.zeros((0, 4), dtype=np.float32))
    assert len(empty) == 0
    arrays = {"original_bgr": image, "box": boxes, "normalized_rgb": item["img"],
              "fixture_rgb_author": small["img"], "fixture_rgb_independent": independent}
    np.savez_compressed(output.with_name(output.stem + "_input.npz"), **arrays)
    forward = mesh_run(output, normalized_image=torch.from_numpy(item["img"])[None])
    return {"status": "R16_original_dataset_forward_male_subpath_passed", "protocol": "architecture_r16_protocol.json",
            "image": protocol["image"], "box_source": protocol["box"], "original_image_shape": list(image.shape),
            "box_size": float(item["box_size"]), "box_center": item["box_center"].tolist(),
            "RGB_shape": list(item["img"].shape), "original_logs": logs,
            "independent_affine_RGB_normalization_max_error": affine_error, "empty_box_dataset_length": len(empty),
            "invalid_boxes": invalid, "forward": forward, "training_executed": False, "checkpoint_loaded": False,
            "network_requests": 0, "whole_goal_complete": False, "boundary": protocol["boundary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(p.exists() for p in (args.output, args.output.with_suffix(".npz"),
                               args.output.with_name(args.output.stem + "_input.npz"))):
        raise FileExistsError("retain previous outputs; choose new version")
    try:
        result = run(args.output)
    except Exception as error:  # noqa: BLE001 -- preserve crop/forward failures
        result = {"status": "R16_original_crop_forward_failed", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc(), "training_executed": False}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False), flush=True)
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
