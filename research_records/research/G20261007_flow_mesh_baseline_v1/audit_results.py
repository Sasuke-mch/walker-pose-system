"""Recheck saved artifacts; does not claim numerical checks prove physical accuracy."""
import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch


def main():
    root = Path(__file__).resolve().parent
    repository = root.parents[2]
    outputs = root / "outputs"
    checks = {}
    papers = json.loads((root / "papers.json").read_text(encoding="utf-8"))
    literature = (root / "LITERATURE_ANALYSIS.md").read_text(encoding="utf-8")
    checks["20_distinct_primary_sources_and_written_cards"] = (
        len(papers) == 20 and len({p["arxiv_id"] for p in papers}) == 20
        and all((root / p["text_path"]).is_file() and p["read_anchor_lines"]
                and p["reading_status"] == "methods_main_experiments_analyzed"
                and f'## {p["id"]} ' in literature for p in papers))
    # This confirms the recorded scope/artifacts, not independently that a human read every appendix.
    report = {"checks": checks, "models": {}}
    config = json.loads((root / "config_learning1000.json").read_text())
    for key, path in (("use_graph", "config_no_graph1000.json"),
                      ("use_history", "config_no_history1000.json")):
        other = json.loads((root / path).read_text())
        expected = json.loads(json.dumps(config))
        expected["model"][key] = False
        checks[f"only_{key}_changes_in_control"] = other == expected
    models = [("flow", "flow_v2", "fresh_flow"),
              ("regression", "regression_v2", "fresh_regression"),
              ("no_graph", "flow_no_graph", "fresh_no_graph"),
              ("no_history", "flow_no_history", "fresh_no_history")]
    for name, training, prediction in models:
        train_report = json.loads((outputs / training / "result.json").read_text())
        test_report = json.loads((outputs / prediction / "result.json").read_text())
        checks[f"{name}_loss_decreases"] = (
            np.isfinite(train_report["final_validation_loss"])
            and train_report["final_validation_loss"] < train_report["initial_validation_loss"])
        checks[f"{name}_1000_steps_logged"] = (
            train_report["steps"] == 1000 and train_report["logs"][-1]["step"] == 1000)
        checks[f"{name}_uses_condition"] = (
            test_report["metrics"]["vertex_error_metres_label_consistency"]
            < test_report["metrics"]["shuffled_condition_vertex_error_metres"])
        checks[f"{name}_no_oracle"] = test_report["candidates"] == 1 and not test_report["oracle_selection"]
        with np.load(outputs / prediction / "prediction.npz", allow_pickle=False) as archive:
            rotations = archive["rotations"]
            valid = archive["frame_valid"]
            vertices, faces = archive["vertices"], archive["faces"]
            orthogonal = np.max(np.abs(np.swapaxes(rotations, -1, -2) @ rotations - np.eye(3)))
            determinant = np.max(np.abs(np.linalg.det(rotations) - 1))
            checks[f"{name}_legal_rotation_and_real_topology"] = bool(
                valid.shape == (16, 4) and valid.all() and archive["rotation_valid"].all()
                and np.isfinite(vertices).all() and vertices.shape == (16, 4, 6890, 3)
                and faces.shape == (13776, 3) and faces.min() >= 0 and faces.max() < 6890
                and orthogonal < 1e-5 and determinant < 1e-5)
        report["models"][name] = {
            "training": f"outputs/{training}", "prediction": f"outputs/{prediction}",
            "parameters": train_report["parameters"], "steps": train_report["steps"],
            "initial_validation_loss": train_report["initial_validation_loss"],
            "final_validation_loss": train_report["final_validation_loss"],
            "vertex_consistency_mm": 1000 * test_report["metrics"]["vertex_error_metres_label_consistency"],
            "shuffled_condition_mm": 1000 * test_report["metrics"]["shuffled_condition_vertex_error_metres"],
            "nfe": test_report["steps"], "candidates": test_report["candidates"],
            "max_orthogonality_error": float(orthogonal), "max_determinant_error": float(determinant)}
    original = torch.load(outputs / "flow_v2/checkpoint.pt", map_location="cpu", weights_only=True)
    replay = torch.load(outputs / "flow_v2_replay/checkpoint.pt", map_location="cpu", weights_only=True)
    checks["full_training_replay_bitwise_identical"] = all(
        torch.equal(value, replay["model_state"][key]) for key, value in original["model_state"].items())
    reproduced = json.loads((root / "reproducibility.json").read_text())
    checks["verification_inputs_recreated_exactly"] = reproduced["frozen_verification_inputs_recreated_exactly"]
    with np.load(outputs / "fresh_flow/prediction.npz") as first, np.load(outputs / "fresh_replay/prediction.npz") as second:
        checks["inference_replay_bitwise_identical"] = all(
            np.array_equal(first[key], second[key]) for key in ("state", "rotations", "betas", "translation", "vertices", "joints", "faces"))
    with np.load(outputs / "all_missing_infer/prediction.npz") as missing:
        checks["all_64_missing_frames_retained_invalid"] = bool(
            missing["frame_valid"].shape == (16, 4) and not missing["frame_valid"].any()
            and np.isnan(missing["vertices"]).all() and np.isnan(missing["joints"]).all())
    suite = ET.parse(outputs / "pytest_final.xml").getroot().find("testsuite")
    checks["21_behavior_tests_passed"] = bool(suite is not None and suite.get("tests") == "21"
                                              and suite.get("failures") == "0" and suite.get("errors") == "0")
    for path in (repository / "realtime_app/learning_flow_mesh").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    checks["all_module_python_syntax_valid"] = True
    report["evidence_boundary"] = "Synthetic low-dimensional artificial-motion engineering checks; not real RGB, fisheye or physical accuracy. Reading scope is methods/main experiments, not all appendices or visuals."
    report["all_checks_passed"] = all(checks.values())
    (root / "verification_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
