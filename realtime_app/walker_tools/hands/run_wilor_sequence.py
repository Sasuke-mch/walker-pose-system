#!/usr/bin/env python3
"""Run WiLoR on the project's ordered raw-fisheye image pairs.

This produces an audit artifact only.  It does not triangulate hands or alter
the SMPL/SMPL-H fitting chain.  The saved 2-D points are projected from WiLoR
MANO joints into the original image pixel frame; local 3-D remains tagged as a
model-camera hypothesis.
"""

from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
from walker_tools._compat import PROJECT_ROOT as _tool_project_root
_tool_prepare_imports()

import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def legacy_compatibility() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, value in {
        "bool": np.bool_, "int": np.int64, "float": np.float64,
        "complex": np.complex128, "object": np.object_,
        "unicode": np.str_, "str": np.str_,
    }.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def load_model(repo: Path, checkpoint: Path, device: str):
    legacy_compatibility()
    os.environ.setdefault("PYOPENGL_PLATFORM", "win32")
    sys.path.insert(0, str(repo))
    import torch
    from wilor.configs import get_config
    from wilor.models import WiLoR

    cfg = get_config(str(repo / "pretrained_models" / "model_config.yaml"), update_cachedir=True)
    if "vit" in cfg.MODEL.BACKBONE.TYPE and "BBOX_SHAPE" not in cfg.MODEL:
        cfg.defrost(); cfg.MODEL.BBOX_SHAPE = [192, 256]; cfg.freeze()
    if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
        cfg.defrost(); cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS"); cfg.freeze()
    if "DATA_DIR" in cfg.MANO:
        cfg.defrost()
        mano_root = repo / "mano_data"
        cfg.MANO.DATA_DIR = str(mano_root)
        cfg.MANO.MODEL_PATH = str(mano_root)
        cfg.MANO.MEAN_PARAMS = str(mano_root / "mano_mean_params.npz")
        cfg.freeze()
    model = WiLoR.load_from_checkpoint(
        str(checkpoint), strict=False, cfg=cfg, init_renderer=False,
    ).to(device).eval()
    return model, cfg


def process_image(model, cfg, detector, image_path: Path, device: str, orientation: str,
                  raw_size: tuple[int, int] = (1920, 1080)):
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")
    raw_width, raw_height = raw_size
    source_height, source_width = image.shape[:2]
    expected_source = (raw_width, raw_height) if orientation == "raw" else (raw_height, raw_width)
    if (source_width, source_height) != expected_source:
        raise ValueError(
            f"orientation/image-size contract failed for {image_path.name}: "
            f"orientation={orientation}, source={(source_width, source_height)}, "
            f"expected={expected_source}")
    if orientation == "left_ccw90":
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == "right_cw90":
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation != "raw":
        raise ValueError(f"unsupported orientation: {orientation}")
    if (image.shape[1], image.shape[0]) != (raw_width, raw_height):
        raise ValueError(
            f"inverse orientation did not restore raw size for {image_path.name}: "
            f"got={(image.shape[1], image.shape[0])}, expected={(raw_width, raw_height)}")
    import torch
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to
    detections = detector(image, conf=0.3, verbose=False)[0]
    boxes, right, detector_confidences = [], [], []
    for det in detections:
        data = det.boxes.data.detach().cpu().numpy()
        if data.ndim == 1:
            data = data[None, :]
        for row in data:
            boxes.append(row[:4].tolist())
            right.append(float(row[5]))
            detector_confidences.append(float(row[4]))
    if not boxes:
        return []
    dataset = ViTDetDataset(
        cfg, image, np.asarray(boxes, dtype=np.float32),
        np.asarray(right, dtype=np.float32), rescale_factor=2.0,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    records = []
    batch_offset = 0
    for batch in loader:
        raw_batch = {key: value for key, value in batch.items()}
        batch = recursive_to(batch, torch.device(device))
        with torch.no_grad():
            out = model(batch)
        # Keep native parameters BEFORE full-image camera/handedness mutation.
        # WiLoR flips left crops and uses canonical right MANO for both sides.
        if not model.mano.is_rhand:
            raise RuntimeError("native MANO schema requires canonical right model")
        native_mano = {k: v.detach().cpu().numpy().copy()
                       for k, v in out["pred_mano_params"].items()}
        native_cam = out["pred_cam"].detach().cpu().numpy().copy()
        multiplier = 2.0 * batch["right"] - 1.0
        pred_cam = out["pred_cam"].clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * max(image.shape[:2])
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        image_size = batch["img_size"].float()
        from wilor.utils.renderer import cam_crop_to_full
        cam_t = cam_crop_to_full(pred_cam, box_center, box_size, image_size, focal)
        joints = out["pred_keypoints_3d"].detach().cpu().numpy()
        verts = out["pred_vertices"].detach().cpu().numpy()
        cam_t_np = cam_t.detach().cpu().numpy()
        side = batch["right"].detach().cpu().numpy()
        centers = box_center.detach().cpu().numpy()
        sizes = box_size.detach().cpu().numpy()
        for index in range(len(joints)):
            sign = 2.0 * float(side[index]) - 1.0
            joints_i = joints[index].copy(); verts_i = verts[index].copy()
            joints_i[:, 0] *= sign; verts_i[:, 0] *= sign
            cam_i = cam_t_np[index]
            xyz = joints_i + cam_i[None, :]
            projection_ok = bool(np.isfinite(xyz).all() and (xyz[:, 2] > 0).all())
            pixels = np.full((len(xyz), 2), np.nan)
            if projection_ok:
                pixels = xyz[:, :2] / xyz[:, 2:3]
                pixels[:, 0] = pixels[:, 0] * focal + image.shape[1] / 2.0
                pixels[:, 1] = pixels[:, 1] * focal + image.shape[0] / 2.0
            pixels_in_bounds = bool(
                projection_ok and not ((pixels[:, 0] < 0).any() or (pixels[:, 0] >= image.shape[1]).any()
                     or (pixels[:, 1] < 0).any() or (pixels[:, 1] >= image.shape[0]).any())
            )
            records.append({
                "candidate_index": index,
                "side": "right" if sign > 0 else "left",
                "detector_confidence": detector_confidences[batch_offset + index],
                "detector_bbox_xyxy_raw": boxes[batch_offset + index],
                "mano_parameter_convention": "wilor_canonical_right_rotmat_v1",
                "mano_parameters": {
                    "global_orient": native_mano["global_orient"][index].reshape(1, 3, 3).tolist(),
                    "hand_pose": native_mano["hand_pose"][index].reshape(15, 3, 3).tolist(),
                    "betas": native_mano["betas"][index].reshape(10).tolist(),
                },
                "pred_cam_crop_native": native_cam[index].tolist(),
                "mano_parameter_metadata": {
                    "joint_order": ["index", "middle", "pinky", "ring", "thumb"],
                    "rotation_type": "parent_relative_rotmat",
                    "left_crop_mirrored": bool(sign < 0),
                    "pose_mean_added_by_layer": False,
                    "metric_camera_pose_accepted": False,
                },
                "bbox_xyxy": [
                    float(centers[index][0] - sizes[index] / 2.0),
                    float(centers[index][1] - sizes[index] / 2.0),
                    float(centers[index][0] + sizes[index] / 2.0),
                    float(centers[index][1] + sizes[index] / 2.0),
                ],
                "box_center": centers[index].tolist(),
                "box_size": float(np.asarray(sizes[index]).reshape(-1)[0]),
                "keypoints_2d_raw_fisheye": pixels.tolist(),
                "keypoints_3d_model_local": joints_i.tolist(),
                "vertices_3d_model_local": verts_i.tolist(),
                "camera_translation_model": cam_i.tolist(),
                "coordinate_frame_3d": "model_local_unaccepted",
                "pixel_frame": "raw_fisheye",
                "raw_pixel_bounds_ok": pixels_in_bounds,
                "model_projection_status": "pinhole_hypothesis" if projection_ok else "invalid_model_depth_or_nonfinite",
                "input_image_transform": f"inverse_{orientation}_to_raw_fisheye",
                "orientation_contract": {
                    "source_size_wh": [source_width, source_height],
                    "raw_size_wh": [raw_width, raw_height],
                    "validated": True,
                },
                "focal_length_model_projection_px": float(focal),
            })
        batch_offset += len(joints)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--orientation", choices=("left_ccw90", "right_cw90", "raw"), default="raw")
    parser.add_argument("--raw-width", type=int, default=1920)
    parser.add_argument("--raw-height", type=int, default=1080)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refuse to overwrite existing WiLoR output: {args.output}")
    root = _tool_project_root
    repo = root / "third_party" / "WiLoR"
    checkpoint = repo / "pretrained_models" / "wilor_final.ckpt"
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    load_start = time.perf_counter()
    model, cfg = load_model(repo, checkpoint, args.device)
    # The bundled detector is a trusted local Ultralytics checkpoint.  The
    # PyTorch 2.6+ default ``weights_only=True`` rejects its PoseModel class.
    # WiLoR itself uses Lightning state dictionaries; this compatibility shim
    # is scoped to loading that detector.
    import torch
    _torch_load = torch.load
    def _trusted_torch_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return _torch_load(*load_args, **load_kwargs)
    torch.load = _trusted_torch_load
    from ultralytics import YOLO
    detector = YOLO(str(repo / "pretrained_models" / "detector.pt")).to(args.device)
    load_ms = (time.perf_counter() - load_start) * 1000.0
    images = sorted(args.image_dir.glob("pair_*.png"))
    if args.limit is not None:
        images = images[:args.limit]
    if not images:
        raise FileNotFoundError(f"no pair_*.png under {args.image_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for frame_index, path in enumerate(images):
            frame_start = time.perf_counter()
            records = process_image(model, cfg, detector, path, args.device, args.orientation,
                                    (args.raw_width, args.raw_height))
            if args.device == "cuda":
                torch.cuda.synchronize()
            processing_ms = (time.perf_counter() - frame_start) * 1000.0
            handle.write(json.dumps({"frame_index": frame_index, "image": str(path.resolve()),
                                     "processing_ms": processing_ms, "records": records}, ensure_ascii=False) + "\n")
            handle.flush()
            print(frame_index, path.name, len(records), f"{processing_ms:.1f} ms", flush=True)
    print(f"model_and_detector_load_ms={load_ms:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
