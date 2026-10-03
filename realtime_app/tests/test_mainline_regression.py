import json
from pathlib import Path

import numpy as np
import pytest

from pose_app.mainline_regression import audit_contract, compare_arrays, compare_archives, observation_snapshot
from check_mainline_regression import main


def test_rejected_nan_positions_are_part_of_contract():
    assert compare_arrays(np.array([np.nan, 1.]), np.array([np.nan, 1.]))["passed"]
    assert not compare_arrays(np.array([np.nan, 1.]), np.array([1., np.nan]))["passed"]


@pytest.mark.parametrize("candidate", [np.array([2, 1]), np.array([1, 2], np.int32), np.array([[1, 2]])])
def test_frame_order_dtype_and_shape_changes_fail(candidate):
    assert not compare_arrays(np.array([1, 2], np.int64), candidate)["passed"]


def test_mask_reason_and_topology_changes_fail_even_with_large_float_tolerance():
    reference = {"mask": np.array([True]), "reason": np.array(["negative_depth"]),
                 "faces": np.array([[0, 1, 2]])}
    candidate = {"mask": np.array([False]), "reason": np.array(["None"]),
                 "faces": np.array([[0, 2, 1]])}
    assert all(not row["passed"] for key, row in compare_archives(reference, candidate, 100).items() if key != "keys")


def test_units_and_missing_arrays_cannot_pass():
    assert not compare_arrays(np.array([1000.]), np.array([1.]))["passed"]
    assert not compare_archives({"beta": np.zeros((1, 10))}, {})["keys"]["passed"]


def test_infinite_values_are_not_treated_as_nan():
    assert not compare_arrays(np.array([np.inf]), np.array([-np.inf]))["passed"]


def test_missing_assets_report_unavailable_and_nonzero(tmp_path):
    output = tmp_path / "report"
    assert main(["compare", "--fit", str(tmp_path / "missing"), "--reference", str(tmp_path / "golden"),
                 "--output", str(output)]) == 2
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))["status"] == "unavailable"


def test_existing_output_is_never_overwritten(tmp_path):
    with pytest.raises(SystemExit):
        main(["compare", "--fit", str(tmp_path), "--reference", str(tmp_path), "--output", str(tmp_path)])


def test_compare_mismatch_returns_failure_report(tmp_path):
    result = {key: np.zeros((1, 3)) for key in ("vertices", "betas", "body_pose", "vposer_latent")}
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    for directory in (reference, candidate):
        directory.mkdir()
        np.savez(directory / "result.npz", **result)
        np.savez(directory / "cold_initialization.npz", pair_id=np.array([0]))
    result["vertices"][0, 0] = 0.01
    np.savez(candidate / "result.npz", **result)
    output = tmp_path / "report"
    assert main(["compare", "--fit", str(candidate), "--reference", str(reference), "--output", str(output)]) == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "mismatch"
    assert not report["checks"]["result_parity"]["vertices"]["passed"]


def test_duplicate_frame_ids_detected_before_dictionary_overwrite(tmp_path):
    from types import SimpleNamespace
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"images": [{"image_id": 0}, {"image_id": 0}]}))
    with pytest.raises(ValueError, match="duplicate frame IDs"):
        observation_snapshot(SimpleNamespace(left_raw=path, right_raw=Path("unused")))


def contract_fixture():
    from types import SimpleNamespace
    from pose_app.smplh_fitting.stages import build_stage_schedule
    args = SimpleNamespace(steps=180, base_steps=1, beta_steps=1, joint_steps=1,
                           hand_steps=1, contact_steps=1, contact_refine_steps=0,
                           hand_pca_comps=12, body_temporal_weight=0, body_reprojection_weight=0)
    names, train, budgets = build_stage_schedule(args, False)
    ids = np.arange(2, dtype=np.int64)
    tri = np.zeros((2, 17, 3), dtype=np.float64)
    mask = np.ones((2, 17), dtype=bool)
    quality = np.ones((2, 17), dtype=np.float64)
    result = {"pair_id": ids, "betas": np.zeros((1, 10)), "body_pose": np.zeros((2, 63)),
              "vposer_latent": np.zeros((2, 32)), "raw_triangulated_points": tri,
              "body_accepted": mask, "body_quality": quality}
    for key in ("vertices", "global_orient", "transl", "left_hand_pose", "right_hand_pose",
                "predicted_coco", "smplh_joints"):
        result[key] = np.zeros((2, 3))
    for side in ("left", "right"):
        result[side + "_hand_pca"] = np.zeros((2, 12))
    initial = {"pair_id": ids, "betas": np.zeros((1, 10)), "vposer_latent": np.zeros((2, 32))}
    summary = {"contact_enabled": False, "shared_hand_pose": True, "hand_pose_parameter_frames": 1,
               "body_temporal_weight": 0, "body_reprojection": {"weight": 0},
               "stage_schedule": [{"stage": name, "trainable": parameters, "steps": budget}
                                  for name, parameters, budget in zip(names, train, budgets)]}
    snapshot = {"pair_id": ids, "triangulated_mm": tri, "accepted": mask, "quality": quality}
    return result, initial, summary, snapshot, args


def test_valid_frozen_contract_passes():
    assert all(row["passed"] for row in audit_contract(*contract_fixture()).values())


@pytest.mark.parametrize("fault,failed", [
    ("beta", "shared_beta"), ("hand", "left_shared_pca"),
    ("stage", "stage_trainability_and_budget"), ("warm", "cold_beta_zero"),
    ("frame", "frames"), ("nan", "finite_vertices"),
])
def test_frozen_semantics_tampering_fails(fault, failed):
    result, initial, summary, snapshot, args = contract_fixture()
    if fault == "beta":
        result["betas"] = np.zeros((2, 10))
    elif fault == "hand":
        result["left_hand_pca"][1, 0] = 1
    elif fault == "stage":
        summary["stage_schedule"][3]["trainable"] = ["beta", "transl"]
    elif fault == "warm":
        initial["betas"][0, 0] = 1
    elif fault == "frame":
        result["pair_id"] = result["pair_id"][::-1]
    elif fault == "nan":
        result["vertices"][0, 0] = np.nan
    assert not audit_contract(result, initial, summary, snapshot, args)[failed]["passed"]
