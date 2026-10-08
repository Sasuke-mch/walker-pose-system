"""Original P16 volume/RANSAC networks at published size; no training."""
import argparse
import copy
import importlib
import json
import platform
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from yacs.config import CfgNode

from .run_original import ROOT
from .run_probability import install_download_guard


class SequenceRandom:
    """Module-local Python3.12 compatibility; sorted integer sets only."""
    @staticmethod
    def sample(population, k):
        if isinstance(population, set):
            if not all(type(x) is int for x in population):
                raise TypeError("only original integer view sets supported")
            population = sorted(population)
        return random.sample(population, k)


def load_original(kind, device):
    root = ROOT / "LearnableTriangulation"
    sys.path.insert(0, str(root.resolve()))
    module = importlib.import_module("mvn.models.triangulation")
    geometry = importlib.import_module("mvn.utils.multiview")
    config_path = root / "experiments/human36m/eval" / ("human36m_vol_softmax.yaml" if kind == "volume" else "human36m_ransac.yaml")
    config = CfgNode(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    config.model.backbone.init_weights = False
    cls = module.VolumetricTriangulationNet if kind == "volume" else module.RANSACTriangulationNet
    # Allocate original full-width networks directly on GPU, without a CPU copy.
    with torch.device(device):
        model = cls(config, device=device)
    model.to(device).eval()
    return model, module, geometry, config, config_path


def camera_fixture(geometry, device, image_shape):
    height, width = image_shape
    K = np.array([[384., 0, width / 2], [0, 384., height / 2], [0, 0, 1.]])
    centers = ((-300., 0, 0), (300., 0, 0), (0, -300., 0), (0, 300., 0))
    cameras = [[geometry.Camera(np.eye(3), -np.asarray(center), K, name=f"synthetic_{i}")]
               for i, center in enumerate(centers)]
    projection = torch.tensor(np.stack([c[0].projection for c in cameras]), device=device, dtype=torch.float32)[None]
    return cameras, projection


def volume(device):
    model, _module, geometry, cfg, source = load_original("volume", device)
    assert cfg.model.volume_size == 64 and cfg.image_shape == [384, 384]
    assert not cfg.model.use_gt_pelvis and cfg.model.kind == "mpii"
    images = torch.randn(1, 4, 3, *cfg.image_shape, device=device) * 0.1
    cameras, projection = camera_fixture(geometry, device, cfg.image_shape)
    predicted = np.zeros((17, 4), dtype=np.float32)
    predicted[:, 2], predicted[:, 3] = 3000., 1.
    predicted[6, :3] = [80., -40., 3000.]
    batch = {"cameras": cameras, "pred_keypoints_3d": [predicted]}
    original_camera_K = [c[0].K.copy() for c in cameras]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    torch.cuda.reset_peak_memory_stats(device) if str(device).startswith("cuda") else None
    first = model(images, projection, batch)
    xyz, features, probabilities, _confidences, _cuboids, coordinates, center = first
    shapes = [list(x.shape) if isinstance(x, torch.Tensor) else None for x in first]
    assert xyz.shape == (1, 17, 3) and probabilities.shape == (1, 17, 64, 64, 64)
    assert torch.isfinite(xyz).all() and torch.isfinite(probabilities).all()
    assert torch.equal(center[0], torch.as_tensor(predicted[6, :3], device=device))
    probability_error = float((probabilities.flatten(2).sum(-1) - 1).abs().max())
    assert probability_error < 1e-5
    minimum, maximum = coordinates.reshape(-1, 3).amin(0), coordinates.reshape(-1, 3).amax(0)
    assert torch.allclose(minimum, center[0] - 1250) and torch.allclose(maximum, center[0] + 1250)
    repeat = model(images, projection, batch)
    repeat_error = float((repeat[0] - xyz).abs().max())
    assert repeat_error == 0
    del repeat
    # Source overwrites this argument using deep-copied cameras. Confirm behavior.
    ignored = model(images, torch.full_like(projection, torch.nan), batch)
    ignored_error = float((ignored[0] - xyz).abs().max())
    assert ignored_error == 0
    del ignored
    changed_cameras = copy.deepcopy(cameras)
    for camera in changed_cameras:
        camera[0].K[0, 2] += 20
    changed_geometry = model(images, projection, {"cameras": changed_cameras,
                                                "pred_keypoints_3d": [predicted]})
    camera_output_change = float((changed_geometry[0] - xyz).abs().max())
    assert torch.isfinite(changed_geometry[0]).all() and camera_output_change > 1e-5
    del changed_geometry
    shifted_predicted = predicted.copy()
    shifted_predicted[6, 0] += 100
    shifted_batch = {"cameras": cameras, "pred_keypoints_3d": [shifted_predicted]}
    shifted = model(images, projection, shifted_batch)
    center_change = shifted[-1] - center
    assert torch.equal(center_change, torch.tensor([[100., 0., 0.]], device=device))
    output_change = float((shifted[0] - xyz).abs().max())
    del shifted
    # Missing center is an explicit prerequisite, not derived from images here.
    try:
        model(images, projection, {"cameras": cameras})
    except KeyError as error:
        missing_center = str(error)
    else:
        raise AssertionError("expected missing predicted center to fail")
    assert all(np.array_equal(c[0].K, k) for c, k in zip(cameras, original_camera_K))
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    report = {"status": "original_volumetric_full_forward_passed", "config": source.as_posix(),
        "parameter_count": sum(p.numel() for p in model.parameters()), "image_shape": list(images.shape),
        "volume_size": cfg.model.volume_size, "cuboid_side_mm": cfg.model.cuboid_side,
        "aggregation": cfg.model.volume_aggregation_method, "use_gt_pelvis": cfg.model.use_gt_pelvis,
        "predicted_pelvis_index": 6, "output_shapes": shapes,
        "predicted_center_mm": center[0].tolist(), "volume_probability_sum_max_error": probability_error,
        "finite_repeat_max_error": repeat_error, "ignored_nan_projection_argument_output_max_error": ignored_error,
        "actual_camera_cx_shift_px": 20, "actual_camera_shift_xyz_max_change_mm": camera_output_change,
        "actual_geometry_source": "batch cameras, resized deep copies", "input_cameras_not_mutated": True,
        "center_shift_mm": center_change.tolist(), "center_shift_xyz_max_change_mm": output_change,
        "missing_predicted_center_error": missing_center, "state_dict_unchanged": True,
        "gpu_peak_allocated_bytes_including_check_snapshots": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None,
        "coverage": "Original ResNet152/unprojection/64cubed V2V/integration full forward; artificial predicted center/cameras, not original detector/precalculated center/data/checkpoints"}
    arrays = {"xyz_mm": xyz.cpu().numpy(), "features": features.cpu().numpy(),
              "probabilities": probabilities.cpu().numpy(), "coordinate_volume_mm": coordinates.cpu().numpy(),
              "predicted_center_mm": center.cpu().numpy(), "input_predicted_keypoints": predicted,
              "input_projection": projection.cpu().numpy()}
    return report, arrays


def ransac_controls(model, geometry, projection):
    matrices = projection[0].cpu().numpy().astype(np.float64)
    truth = np.array([[120., -40., 3000.]])
    points = np.stack([geometry.project_3d_points_to_image_plane_without_distortion(p, truth)[0] for p in matrices])
    clean, inliers = model.triangulate_ransac(matrices, points, direct_optimization=True)
    clean_error = float(np.linalg.norm(clean - truth[0]))
    assert clean_error < 1e-5 and len(inliers) == 4
    outlier = points.copy()
    outlier[3] += [130., -100.]
    random.seed(51)
    robust, kept = model.triangulate_ransac(matrices, outlier, direct_optimization=True)
    robust_error = float(np.linalg.norm(robust - truth[0]))
    assert robust_error < 1e-5 and kept.tolist() == [0, 1, 2]
    # Two views cannot create a redundant consistency vote; sampled views forced in.
    random.seed(51)
    two, two_kept = model.triangulate_ransac(matrices[[0, 3]], outlier[[0, 3]], direct_optimization=True)
    assert two_kept.tolist() == [0, 1]
    two_error = float(np.linalg.norm(two - truth[0]))
    artificial_offset = points.copy()
    artificial_offset[:, 0] += 20
    error = geometry.calc_reprojection_error_matrix(truth, artificial_offset, matrices)
    assert np.allclose(error, 10)
    return {"known_point_error_mm": clean_error, "clean_inliers": inliers.tolist(),
            "outlier_view": 3, "robust_error_mm": robust_error, "robust_inliers": kept.tolist(),
            "two_view_forced_inliers": two_kept.tolist(), "two_view_outlier_xyz_error_mm_fixture": two_error,
            "original_reprojection_error_for_20px_offset": error.tolist(),
            "declared_threshold": 15, "effective_euclidean_pixel_distance_threshold": 30,
            "boundary": "Analytic geometry/dependency fixture only; no real-person accuracy"}


def ransac(device, compatibility):
    model, module, geometry, cfg, source = load_original("ransac", device)
    original_random = module.random
    if compatibility:
        module.random = SequenceRandom
    try:
        cameras, projection = camera_fixture(geometry, device, cfg.image_shape)
        controls = ransac_controls(model, geometry, projection)
        images = torch.randn(1, 4, 3, *cfg.image_shape, device=device) * 0.1
        before = {k: v.clone() for k, v in model.state_dict().items()}
        random.seed(52)
        xyz, xy, heatmaps, confidence = model(images, projection, {"cameras": cameras})
        assert xyz.shape == (1, 17, 3) and xy.shape == (1, 4, 17, 2)
        assert torch.isfinite(xyz).all() and torch.isfinite(heatmaps).all()
        assert torch.equal(confidence, torch.zeros_like(confidence))
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
        return {"status": "original_ransac_full_forward_with_explicit_python_compatibility_passed" if compatibility else "original_ransac_full_forward_passed",
                "config": source.as_posix(), "parameter_count": sum(p.numel() for p in model.parameters()),
                "image_shape": list(images.shape), "xyz_shape": list(xyz.shape), "xy_shape": list(xy.shape),
                "heatmap_shape": list(heatmaps.shape), "confidence_shape": list(confidence.shape),
                "confidence_is_original_zero_placeholder": True, "direct_optimization": cfg.model.direct_optimization,
                "state_dict_unchanged": True, "geometry_controls": controls,
                "compatibility": "module-local random.sample converts sorted integer set to list, source unchanged" if compatibility else None,
                "coverage": "Original ResNet152/argmax/CPU RANSAC/SciPy direct refinement full forward; no original detector/data/checkpoints, no project strict mask"}, {
                    "xyz_mm": xyz.cpu().numpy(), "xy_pixels": xy.cpu().numpy(), "heatmaps": heatmaps.cpu().numpy(),
                    "confidence_placeholder": confidence.cpu().numpy(), "input_projection": projection.cpu().numpy()}
    finally:
        module.random = original_random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("volume", "ransac"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python-set-compatibility", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError(args.output)
    torch.manual_seed(19)
    torch.set_num_threads(2)
    random.seed(51)
    start = time.perf_counter()
    try:
        install_download_guard()
        with torch.inference_mode():
            result, arrays = volume(args.device) if args.kind == "volume" else ransac(args.device, args.python_set_compatibility)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    except Exception as error:  # noqa: BLE001 -- preserve original compatibility/numeric/resource failures
        result = {"status": "original_multiview_check_failed", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc()}
    result.update(id="P16", branch=args.kind, training_executed=False, weights="random_initialization",
                  protocol="architecture_multiview_protocol.json", python_set_compatibility=args.python_set_compatibility,
                  environment={"python": platform.python_version(), "torch": torch.__version__, "device": args.device,
                               "elapsed_total_seconds_not_latency": time.perf_counter() - start},
                  evidence_boundary="Original structure and artificial geometry only, not real accuracy/fisheye compatibility/realtime")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result["status"].endswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
