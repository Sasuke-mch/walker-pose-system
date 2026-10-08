"""R14 unchanged author DLT numerical and missing-input contracts."""
import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import torch

SOURCE = Path("third_party/architecture_reproduction/LearnableTriangulation/mvn/utils/multiview.py")


def load_geometry():
    spec = importlib.util.spec_from_file_location("r14_original_multiview", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    intrinsic = np.array([[384., 0, 192.], [0, 384., 192.], [0, 0, 1.]])
    cameras = np.stack([intrinsic @ np.column_stack((np.eye(3), [x, 0., 0.])) for x in (300., -300.)])
    point = np.array([80., -40., 3000.])
    homogeneous = cameras @ np.append(point, 1.)
    pixels = homogeneous[:, :2] / homogeneous[:, 2:]
    return cameras, point, pixels


def observe(operation):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = operation()
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            result = {"kind": "finite" if np.isfinite(value).all() else "nonfinite",
                      "shape": list(value.shape), "dtype": str(value.dtype),
                      "values": [float(x) if np.isfinite(x) else str(x) for x in value.ravel()]}
        except Exception as error:  # noqa: BLE001 -- author failures are audit outputs
            result = {"kind": "exception", "exception_type": type(error).__name__, "message": str(error)}
    result["warnings"] = [str(item.message) for item in caught]
    return result


def run():
    geometry = load_geometry()
    cameras, point, pixels = fixture()
    tensor = torch.from_numpy
    original = geometry.triangulate_point_from_multiple_views_linear_torch
    baseline = original(tensor(cameras), tensor(pixels))
    assert np.max(np.abs(baseline.numpy() - point)) < 1e-7
    numpy_baseline = geometry.triangulate_point_from_multiple_views_linear(cameras, pixels)
    assert np.max(np.abs(numpy_baseline - point)) < 1e-7
    # Independent rectified stereo formula; does not call author DLT.
    depth = 384. * 600. / (pixels[0, 0] - pixels[1, 0])
    analytic = np.array([(pixels[0, 0] - 192.) * depth / 384. - 300.,
                         (pixels[0, 1] - 192.) * depth / 384., depth])
    assert np.max(np.abs(analytic - point)) < 1e-7
    weights = torch.ones(2, dtype=torch.float64)
    assert torch.equal(original(tensor(cameras), tensor(pixels), weights), baseline)
    scaled = original(tensor(cameras), tensor(pixels), weights * 7.)
    assert (scaled - baseline).abs().max() < 1e-7
    variants = {"healthy": (cameras, pixels, None),
                "all_zero_confidence": (cameras, pixels, np.zeros(2)),
                "one_zero_confidence": (cameras, pixels, np.array([1., 0.])),
                "coincident_cameras": (cameras[[0, 0]], pixels[[0, 0]], None),
                "single_view": (cameras[:1], pixels[:1], None),
                "zero_views": (cameras[:0], pixels[:0], None)}
    for name, index, data in (("nan_point", (0, 0), pixels), ("nan_camera", (0, 0, 0), cameras)):
        changed = data.copy()
        changed[index] = np.nan
        variants[name] = (cameras, changed, None) if data is pixels else (changed, pixels, None)
    variants["nan_confidence"] = (cameras, pixels, np.array([np.nan, 1.]))
    behind = point.copy()
    behind[2] *= -1
    projected = cameras @ np.append(behind, 1.)
    variants["behind_cameras"] = (cameras, projected[:, :2] / projected[:, 2:], None)
    cases = {}
    for name, (projection, xy, confidence) in variants.items():
        cases[name] = {"numpy": observe(lambda projection=projection, xy=xy: geometry.triangulate_point_from_multiple_views_linear(projection, xy)),
                       "torch": observe(lambda projection=projection, xy=xy, confidence=confidence: original(tensor(projection), tensor(xy),
                                                         None if confidence is None else tensor(confidence)))}
    # NumPy has no confidence argument: its result is an explicitly unweighted control.
    cases["batch_float64_input"] = observe(lambda: geometry.triangulate_batch_of_points(
        tensor(cameras[None]), tensor(pixels[None, :, None])))
    return {"status": "R14_original_DLT_numeric_contract_audited", "protocol": "architecture_r14_protocol.json",
            "source": str(SOURCE), "cases": cases, "numpy_confidence_supported": False,
            "healthy_error_mm": float((baseline - tensor(point)).abs().max()),
            "independent_stereo_error_mm": float(np.max(np.abs(analytic - point))),
            "positive_confidence_scale_error_mm": float((scaled - baseline).abs().max()),
            "training_executed": False, "checkpoint_loaded": False, "network_requests": 0,
            "whole_goal_complete": False,
            "boundary": "Original utility outputs/errors, artificial undistorted pinhole only. Finite result is not a valid geometry/visibility/quality decision. No author network or mainline gate changes."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("retain prior result; choose new version")
    report = run()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
