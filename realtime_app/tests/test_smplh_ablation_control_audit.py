import json
from pathlib import Path

import numpy as np
import pytest

from evaluate_smplh_ablations import audit_case_controls
from run_smplh_ablations import modified_options
from walker_tools.body.fit_smplh_wilor_sequence import build_parser
from pose_app.smplh_fitting.stages import build_stage_schedule


def fixture(tmp_path):
    directory = tmp_path / "case"
    fit = directory / "fit"
    fit.mkdir(parents=True)
    values = ["--left-raw", "left", "--right-raw", "right", "--wilor-left", "wl", "--wilor-right", "wr",
              "--calibration-dir", "cal", "--regressor", "reg", "--smplh-model", "model", "--output-dir", "out",
              "--base-steps", "30", "--beta-steps", "30", "--joint-steps", "300", "--hand-steps", "100",
              "--contact-steps", "100", "--contact-refine-steps", "0", "--shared-hand-pose"]
    command = tmp_path / "command.json"
    command.write_text(json.dumps(values), encoding="utf-8")
    case = {"detector": "sapiens2", "changes": {}}
    plan = {"frames": 2, "source_commands": {"sapiens2": str(command)}}
    args = build_parser().parse_args(modified_options(values, {}, fit))
    options = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (directory / "resolved_options.json").write_text(json.dumps(options), encoding="utf-8")
    result = {"pair_id": np.arange(2), "betas": np.zeros((1, 10)),
              "raw_triangulated_points": np.zeros((2, 17, 3)),
              "body_quality": np.ones((2, 17)), "body_accepted": np.ones((2, 17), dtype=bool)}
    for key in ("vertices", "body_pose", "vposer_latent", "transl", "global_orient",
                "left_hand_pose", "right_hand_pose", "predicted_coco", "smplh_joints"):
        result[key] = np.zeros((2, 52, 3))
    for side in ("left", "right"):
        result[side + "_hand_pca"] = np.zeros((2, 12))
    np.savez(fit / "result.npz", **result)
    np.savez(fit / "cold_initialization.npz", pair_id=np.arange(2), betas=np.zeros((1, 10)), vposer_latent=np.zeros((2, 32)))
    names, train, budgets = build_stage_schedule(args, False)
    (fit / "fit_summary.json").write_text(json.dumps({"stage_schedule": [
        {"stage": n, "trainable": t, "steps": b} for n, t, b in zip(names, train, budgets)]}), encoding="utf-8")
    np.savez(directory / "checkpoint_B_shared_beta_30.npz", betas=result["betas"])
    np.savez(directory / "checkpoint_C_body_vposer_refine_300.npz", betas=result["betas"], smplh_joints=result["smplh_joints"])
    observations = {"triangulated_mm": result["raw_triangulated_points"].copy(),
                    "quality": result["body_quality"].copy(), "accepted": result["body_accepted"].copy()}
    return directory, case, plan, observations, result


def test_valid_control_audit_does_not_require_model_assets(tmp_path):
    directory, case, plan, observations, _ = fixture(tmp_path)
    _, result = audit_case_controls(directory, case, plan, observations)
    assert result["betas"].shape == (1, 10)


@pytest.mark.parametrize("fault", ["input", "body_in_hand_stage", "shared_hand", "nonfinite", "frame", "options"])
def test_tampering_fails_even_when_cached_metrics_exist(tmp_path, fault):
    directory, case, plan, observations, result = fixture(tmp_path)
    (directory / "metrics_v2.json").write_text('{"old":"passed"}')
    if fault == "input":
        observations["triangulated_mm"][0, 0, 0] = 1
    elif fault == "body_in_hand_stage":
        result["smplh_joints"][0, 0, 0] = 1
    elif fault == "shared_hand":
        result["left_hand_pca"][1, 0] = 1
    elif fault == "nonfinite":
        result["vertices"][0, 0, 0] = np.nan
    elif fault == "frame":
        result["pair_id"] = np.array([1, 0])
    elif fault == "options":
        case["changes"] = {"beta_policy": "freeze_after_b"}
    np.savez(directory / "fit/result.npz", **result)
    with pytest.raises(ValueError):
        audit_case_controls(directory, case, plan, observations)
