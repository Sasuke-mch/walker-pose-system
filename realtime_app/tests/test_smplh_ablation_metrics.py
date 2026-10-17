import numpy as np
import pytest

from pose_app.smplh_ablation_metrics import leg_diagnostics, stats
from run_smplh_ablations import modified_options, validate_plan


def joints():
    z = np.zeros((6, 52, 3))
    for h, k, a, x in ((1, 4, 7, -.1), (2, 5, 8, .1)):
        z[:, h] = [x, 1, 0]
        z[:, k] = [x, .5, 0]
        z[:, a] = [x, 0, 0]
    return z


def test_common_translation_cannot_be_reported_as_relative_leg_jitter():
    z = joints()
    z += np.arange(6)[:, None, None] * .1
    metrics, _ = leg_diagnostics(z, 30)
    assert metrics["leg_relative_acceleration_m_s2"]["max"] < 1e-12
    assert metrics["leg_motion_excursion_mm"]["max"] < 1e-12
    assert max(metrics["knee_range_deg"]) < 1e-12


def test_motion_retention_and_temporal_variation_are_separate_quantities():
    moving = joints()
    moving[:, 7, 0] += [.0, .01, .02, .03, .04, .05]
    constant, _ = leg_diagnostics(joints(), 30)
    motion, _ = leg_diagnostics(moving, 30)
    assert motion["leg_motion_excursion_mm"]["rms"] > constant["leg_motion_excursion_mm"]["rms"]
    assert motion["leg_relative_acceleration_m_s2"]["max"] < 1e-12
    moving[3, 7, 0] += .1
    noisy, _ = leg_diagnostics(moving, 30)
    assert noisy["leg_relative_acceleration_m_s2"]["rms"] > motion["leg_relative_acceleration_m_s2"]["rms"]


def test_fps_controls_derivative_units():
    z = joints()
    z[3, 7, 0] += .1
    a, _ = leg_diagnostics(z, 15)
    b, _ = leg_diagnostics(z, 30)
    assert b["leg_relative_acceleration_m_s2"]["rms"] == 4 * a["leg_relative_acceleration_m_s2"]["rms"]
    assert b["leg_relative_speed_m_s"]["rms"] == 2 * a["leg_relative_speed_m_s"]["rms"]


def test_invalid_diagnostics_are_not_filtered_into_success():
    with pytest.raises(ValueError, match="nonfinite"):
        stats([1, np.nan])


def test_explicit_zero_weights_and_removed_flags_are_preserved():
    values = modified_options(["python", "script", "--left-raw", "left", "--wrist-reference", "w",
                               "--wrist-reference-weight", "1", "--mano-pose-init", "--output-dir", "old"],
                              {"wrist_reference": None, "wrist_reference_weight": 0, "mano_pose_init": False}, "new")
    assert "--wrist-reference" not in values and "--mano-pose-init" not in values
    assert values[values.index("--wrist-reference-weight") + 1] == "0"
    assert values.count("--output-dir") == 1 and values[-1] == "new"


@pytest.mark.parametrize("fault", ["duplicate", "budget", "path", "detector"])
def test_bad_protocol_is_rejected_before_models(fault):
    row = {"id": "case", "detector": "sapiens2", "changes": {}}
    plan = {"source_commands": {"sapiens2": "command"}, "cases": [row]}
    if fault == "duplicate":
        plan["cases"].append(dict(row))
    elif fault == "budget":
        row["changes"]["joint_steps"] = 30
    elif fault == "path":
        row["id"] = "../outside"
    elif fault == "detector":
        row["detector"] = "missing"
    with pytest.raises(ValueError):
        validate_plan(plan)


def test_saved_command_policy_is_explicit_and_survives_before_left_raw(tmp_path):
    import json
    from check_mainline_regression import fit_arguments
    required = ["--left-raw", "l", "--right-raw", "r", "--wilor-left", "wl", "--wilor-right", "wr",
                "--calibration-dir", "cal", "--regressor", "reg", "--smplh-model", "model", "--output-dir", "out"]
    path = tmp_path / "command.json"
    for prefix, expected in [([], "legacy"), (["--fit-policy", "reliability"], "reliability")]:
        path.write_text(json.dumps(["python", "script", *prefix, *required]))
        assert fit_arguments(path).fit_policy == expected
        values = modified_options(["python", "script", *prefix, *required], {}, "new")
        assert values[values.index("--fit-policy") + 1] == expected


def test_preflight_and_regression_share_one_argv_reader(tmp_path):
    import json
    from pathlib import Path
    from preflight_smplh import fit_arguments as preflight_fit_arguments
    from check_mainline_regression import fit_arguments as regression_fit_arguments
    from walker_tools.body._fit_arguments import read_argv_list

    required = ["--left-raw", "l", "--right-raw", "r", "--wilor-left", "wl", "--wilor-right", "wr",
                "--calibration-dir", "cal", "--regressor", "reg", "--smplh-model", "model", "--output-dir", "out"]
    path = tmp_path / "args.json"
    path.write_text(json.dumps(required))

    args, values = preflight_fit_arguments(path)
    assert values == required
    assert args.left_raw == Path("l")

    assert regression_fit_arguments(path).fit_policy == "legacy"

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError):
        read_argv_list(bad)
