"""H2: exact author codecs and saved 360-frame contract replay, no training."""
import argparse
import ast
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from realtime_app.learning_flow_mesh.interfaces import (
    PixelDomain,
    SixDLayout,
    camera_observations,
    decode_rotation,
    encode_rotation,
    metric_coordinates,
)
from realtime_app.pose_app.calibration import StereoCalibration
from realtime_app.pose_app.smpl_coco_observation import (
    COCO17_NAMES,
    load_coco17_regressor,
)

ROOT = Path("third_party/architecture_reproduction")
INPUT = Path("research_records/engineering_validation/G20261003_repositioned360_cold_smplh_v2")
STRICT = Path("research_records/engineering_validation/G20261005_reference25_mainline_view_v2/first_instance_strict.jsonl")


def original_function(path, name):
    """Compile exact original function AST, without unrelated package/download facades."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(selected) != 1:
        raise ValueError(f"original function unavailable/ambiguous: {path}:{name}")
    space = {"torch": torch, "F": F}
    # Execute only the inspected, named author function in this standalone check.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), space)  # noqa: S102
    return space[name]


def author_rotation_checks():
    torch.manual_seed(45)
    q, _ = torch.linalg.qr(torch.randn(20, 3, 3, dtype=torch.float64))
    q[:, :, -1] *= torch.linalg.det(q)[:, None]
    sources = (
        ("ProHMR", "prohmr/utils/geometry.py", "rot6d_to_rotmat", SixDLayout.PACKED_COLUMNS),
        ("HMR2", "hmr2/utils/geometry.py", "rot6d_to_rotmat", SixDLayout.PACKED_COLUMNS),
        ("HuManiFlow", "utils/rigid_transform_utils.py", "rot6d_to_rotmat", SixDLayout.INTERLEAVED_COLUMNS),
        ("MotionBERT", "lib/utils/utils_mesh.py", "rot6d_to_rotmat", SixDLayout.INTERLEAVED_COLUMNS),
        ("WHAM", "lib/utils/transforms.py", "rotation_6d_to_matrix", SixDLayout.PACKED_ROWS),
        ("WHAM", "lib/utils/transforms.py", "rot6d_to_rotmat", SixDLayout.INTERLEAVED_COLUMNS),
    )
    results = []
    for repo, relative, function, layout in sources:
        path = ROOT / repo / relative
        six = encode_rotation(q, layout)
        original = original_function(path, function)(six)
        adapted, valid = decode_rotation(six, layout)
        error = float((original - adapted).abs().max())
        assert valid.all() and error < 1e-10 and float((original - q).abs().max()) < 1e-10
        results.append({"source": str(path), "function": function, "layout": layout.value,
                        "max_matrix_error": error, "original_function_ast": True})
    wrong, valid = decode_rotation(encode_rotation(q, SixDLayout.PACKED_ROWS), SixDLayout.PACKED_COLUMNS)
    wrong_error = float((wrong - q).abs().max())
    assert valid.all() and wrong_error > 0.1
    return {"fixtures": 20, "dtype": "float64", "functions": results,
            "wrong_layout_still_proper_rotation": True, "wrong_layout_matrix_max_error": wrong_error}


def source_frames(path):
    images = json.loads(path.read_text(encoding="utf-8"))["images"]
    result = {}
    for row in images:
        fid = row["image_id"]
        if type(fid) is not int or fid in result:
            raise ValueError(f"duplicate/undeclared frame ID: {path}")
        result[fid] = row
    return result


def replay_saved():
    # Existing body reader uses legacy absolute pose_app imports. Expose its
    # unchanged package only within this standalone check process.
    sys.path.insert(0, str(Path("realtime_app").resolve()))
    from pose_app.body_observations import raw_side

    cal = StereoCalibration.load(INPUT / "calibration/stereo_fisheye.json")
    sources = {side: source_frames(INPUT / f"{side}_pmpose_raw.json") for side in ("left", "right")}
    control = {side: raw_side(INPUT / f"{side}_pmpose_raw.json", side) for side in sources}
    rows = [json.loads(line) for line in STRICT.read_text(encoding="utf-8").splitlines()]
    ids = [row["index"] for row in rows]
    if ids != list(range(360)) or any(sorted(frames) != ids for frames in sources.values()):
        raise ValueError("unexpected input range/order or missing/duplicate IDs")
    shape = (360, 2, 17)
    raw = np.full((*shape, 2), np.nan)
    xy = np.full_like(raw, np.nan)
    rays = np.full((*shape, 3), np.nan)
    origins = np.full_like(rays, np.nan)
    observed = np.zeros(shape, bool)
    confidence = np.full(shape, np.nan)
    ray_available = np.zeros(shape, bool)
    gate = np.zeros((360, 17), bool)
    reasons = np.full((360, 17), "no_stereo_person", dtype=object)
    point_reasons = np.full(shape, "no_stereo_person", dtype=object)
    ray_reasons = point_reasons.copy()
    flags = np.full((360, 17), "[]", dtype=object)
    xyz = np.full((360, 17, 3), np.nan)
    selected_people = np.full((360, 2), -1, dtype=np.int64)
    pair_ids = np.asarray([row["original_pair_id"] for row in rows])
    if not np.array_equal(pair_ids, np.arange(1157, 1517)):
        raise ValueError("physical pair IDs differ from frozen input range")
    outcomes = [row["pair_outcome"] for row in rows]
    pixel_control_errors = []
    # Check all existing body-reader entries, including strict unmatched frames.
    for side in sources:
        rotation = "ccw90" if side == "left" else "cw90"
        for fid in ids:
            kp = np.asarray(sources[side][fid]["keypoints"][0], dtype=np.float64)[:17]
            restored = camera_observations(kp[:, :2], cal, side=side,
                                          domain=PixelDomain.MODEL, model_rotation=rotation)
            error = np.nanmax(np.abs(restored.raw_xy - control[side][fid][:, :2]))
            assert error == 0
            pixel_control_errors.append(float(error))
    for fid, row in zip(ids, rows):
        people = row["persons_3d"]
        if len(people) > 1:
            raise ValueError("frozen strict contract requires max_matches=1")
        if not people:
            continue  # retain frame, outcome, all 17 unavailable slots
        person = people[0]
        points = person["keypoints_3d"]
        if len(points) != 17 or any(p["index"] != j or p["name"] != COCO17_NAMES[j]
                                    for j, p in enumerate(points)):
            raise ValueError("strict landmark order/name mismatch")
        for view, side in enumerate(sources):
            selected = person[f"{side}_person_id"]
            if type(selected) is not int or not 0 <= selected < len(sources[side][fid]["keypoints"]):
                raise ValueError("invalid frozen person association")
            selected_people[fid, view] = selected
            kp = np.asarray(sources[side][fid]["keypoints"][selected], dtype=np.float64)
            if kp.shape not in ((17, 3), (23, 3)):
                raise ValueError("unsupported detector channel schema")
            result = camera_observations(kp[:17, :2], cal, side=side, domain=PixelDomain.MODEL,
                                         model_rotation="ccw90" if side == "left" else "cw90")
            raw[fid, view], xy[fid, view] = result.raw_xy, result.image_normalized_xy
            rays[fid, view], origins[fid, view] = result.ray_directions_cam0, result.ray_origins_m
            observed[fid, view], ray_available[fid, view] = result.point_valid, result.ray_valid
            confidence[fid, view] = kp[:17, 2]  # original values; no recalibration/threshold
            point_reasons[fid, view], ray_reasons[fid, view] = result.point_reasons, result.ray_reasons
        for j, point in enumerate(points):
            if type(point["valid"]) is not bool:
                raise TypeError("strict gate is not boolean")
            gate[fid, j], reasons[fid, j] = point["valid"], point["reason"]
            flags[fid, j] = json.dumps(point.get("quality_flags", []))
            if point["xyz"] is not None:
                xyz[fid, j] = metric_coordinates(point["xyz"], camera_kind="metric_xyz",
                                                unit=cal.length_unit, frame="cam0")
        assert np.array_equal(gate[fid], [p["valid"] for p in points])
        assert reasons[fid].tolist() == [p["reason"] for p in points]
        assert int(gate[fid].sum()) == person["valid_keypoints"]
    geometry = ray_available & gate[:, None, :]
    assert np.array_equal(geometry, np.broadcast_to(gate[:, None, :], shape))
    assert int(gate.sum()) == 3031 and np.isfinite(xyz[gate]).all()
    assert np.isfinite(rays[geometry]).all()
    norm_error = float(np.abs(np.linalg.norm(rays[geometry], axis=-1) - 1).max())
    assert norm_error < 1e-12
    # A point lying on a ray must project back to the measured raw pixel.
    projection_errors = {}
    scale = 1.0 if cal.length_unit in ("m", "meter") else 0.001
    for view, side in enumerate(sources):
        mask = ray_available[:, view]
        along_ray = origins[:, view][mask] + rays[:, view][mask] * 2
        projector = cal.project_left if side == "left" else cal.project_right
        error = np.linalg.norm(projector(along_ray / scale) - raw[:, view][mask], axis=-1)
        projection_errors[side] = float(error.max())
        assert projection_errors[side] < 1e-4
    regressor = load_coco17_regressor("models/smpl/J_regressor_coco.npy")
    arrays = {"frame_ids": np.asarray(ids), "original_pair_ids": pair_ids,
              "raw_xy": raw, "image_normalized_xy": xy, "ray_directions_cam0": rays,
              "ray_origins_m": origins, "observation_mask": observed,
              "confidence": confidence,
              "geometry_mask": geometry, "strict_gate": gate, "strict_xyz_m_diagnostic": xyz,
              "selected_people": selected_people, "pair_outcomes": np.asarray(outcomes),
              "strict_reasons_json": np.asarray([json.dumps(r.tolist()) for r in reasons]),
              "quality_flags_json": flags.astype(str),
              "point_reasons_json": np.asarray([json.dumps(r.tolist()) for r in point_reasons]),
              "ray_reasons_json": np.asarray([json.dumps(r.tolist()) for r in ray_reasons])}
    report = {"frames": 360, "point_slots": 6120, "matched_frames": int((selected_people[:, 0] >= 0).sum()),
              "unmatched_frames": int((selected_people[:, 0] < 0).sum()),
              "pair_outcomes": dict(Counter(outcomes)), "strict_accepted": int(gate.sum()),
              "reasons": dict(Counter("accepted" if x is None else x for x in reasons.flat)),
              "finite_rejected_xyz_preserved": int((~gate & np.isfinite(xyz).all(-1)).sum()),
              "strict_mask_and_reason_parity": True, "body_reader_pixel_max_error": max(pixel_control_errors),
              "nonzero_selected_person_indices": int((selected_people > 0).sum()),
              "detector_multi_person_frames": {s: sum(len(r["keypoints"]) > 1 for r in f.values()) for s, f in sources.items()},
              "ray_norm_max_error": norm_error, "ray_projection_max_error_px": projection_errors,
              "point_coordinate_valid": int(observed.sum()), "ray_available": int(ray_available.sum()),
              "confidence_copied_without_clipping_or_threshold": True,
              "matched_confidence_range": [float(np.nanmin(confidence)), float(np.nanmax(confidence))],
              "ray_reference": "cam0", "ray_origin_unit": "meter", "calibration_unit": cal.length_unit,
              "baseline_m": float(np.linalg.norm(cal.T) * scale), "regressor_shape": list(regressor.shape),
              "regressor_is_surface_observation_not_kinematic_identity": True,
              "unmatched_policy": "keep slots unavailable; do not select first detector person on unmatched frames",
              "source_paths": [str(INPUT / f"{s}_pmpose_raw.json") for s in sources] + [str(STRICT)]}
    return report, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError("keep prior result; select a new output version")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "architecture_h2_protocol.json", "training_steps": 0,
              "mainline_modified": False, "boundary": "interface/internal geometry checks; no accuracy claim"}
    try:
        report["author_rotations"] = author_rotation_checks()
        report["saved_replay"], arrays = replay_saved()
        np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
        report["status"] = "passed"
    # Preserve failure evidence for any exception, then exit nonzero.
    except Exception:  # noqa: BLE001
        report["status"], report["failure"] = "failed", traceback.format_exc()
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
