"""Evaluate all planned cases; failed cases remain visible in the result table."""
from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys
import time
import traceback

import numpy as np
from check_mainline_regression import fit_arguments
from pose_app.independent_wrist_reference import load_reference
from pose_app.mainline_regression import observation_snapshot, load_archive, compare_archives
from pose_app.smplh_ablation_metrics import evaluate, leg_diagnostics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--retries", type=Path, help="explicit bug-fix retry root; first failure stays in records")
    parser.add_argument("--baseline-run", type=Path, help="external reference for a preregistered follow-up family")
    cli = parser.parse_args()
    plan = read(cli.run / "protocol.json")
    root = Path("research_records/engineering_validation/G20261002_sapiens2_body_mainline448_v2")
    soles_path = Path("research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json")
    sets = read(soles_path)["sets"]
    soles = {side: np.unique(sum(sets[side + "_sole_surface_candidate"].values(), [])) for side in ("left", "right")}
    sources = {}
    for detector, command in plan["source_commands"].items():
        args = fit_arguments(command)
        with np.load(root / detector / "scene/scene_transforms.npz", allow_pickle=False) as archive:
            scene = {key: archive[key] for key in ("rotation_ground_from_left", "translation_ground_from_left_mm", "accepted")}
        sources[detector] = {"observations": observation_snapshot(args),
            "wrist": load_reference(args.wrist_reference, args.calibration_dir, True)[0],
            "scene": scene}
    while True:
        rows = []
        for case in plan["cases"]:
            directory = cli.run / case["id"]
            first_failure = None
            if (directory / "FAILURE.json").exists():
                first_failure = read(directory / "FAILURE.json")
                retry = cli.retries / case["id"] if cli.retries else None
                if retry is not None and (retry / "execution.json").exists():
                    directory = retry
                else:
                    rows.append({"id": case["id"], "status": "failed", "failure": first_failure})
                    continue
            if not (directory / "execution.json").exists():
                rows.append({"id": case["id"], "status": "pending"})
                continue
            output = directory / "metrics_v2.json"
            if not output.exists():
                try:
                    source = sources[case["detector"]]
                    result = load_archive(directory / "fit/result.npz")
                    if not np.array_equal(result["pair_id"], np.arange(plan["frames"])) or result["betas"].shape != (1, 10):
                        raise ValueError("not planned full sequence with shared beta")
                    initial = load_archive(directory / "fit/cold_initialization.npz")
                    if np.any(initial["betas"]) or np.any(initial["vposer_latent"]):
                        raise ValueError("not cold beta/latent initialization")
                    baseline = load_archive((cli.baseline_run or cli.run) / (case["detector"] + "_baseline") / "fit/result.npz")
                    metrics, arrays = evaluate(result, source["observations"], source["wrist"], source["scene"], soles, plan["fps"], baseline)
                    metrics["seconds"] = read(directory / "execution.json")["seconds"]
                    metrics["same_body_parameters_as_baseline"] = {key: bool(np.array_equal(result[key], baseline[key]))
                        for key in ("betas", "global_orient", "transl", "vposer_latent", "body_pose")}
                    metrics["body_joint_max_difference_m"] = float(np.max(np.abs(result["smplh_joints"][:, :22] - baseline["smplh_joints"][:, :22])))
                    metrics["checkpoints"] = {}
                    for checkpoint in directory.glob("checkpoint_*.npz"):
                        z = load_archive(checkpoint)
                        values, _ = leg_diagnostics(z["smplh_joints"], plan["fps"])
                        mask = result["body_accepted"]
                        d = np.linalg.norm(z["predicted_coco"] - result["raw_triangulated_points"] / 1000, axis=-1)
                        values["body_rms_mm"] = float(np.sqrt(np.mean(d[mask] ** 2)) * 1000)
                        values["betas"] = z["betas"].tolist()
                        metrics["checkpoints"][checkpoint.stem] = values
                    audit = directory / "grasp_geometry_audit.json"
                    if not audit.exists():
                        with (directory / "geometry_audit.log").open("w", encoding="utf-8") as log:
                            subprocess.run([sys.executable, "-B", "realtime_app/tools/audit_constructed_grasp_body.py",
                                "--result", str(directory / "fit/result.npz"), "--walker-model",
                                str(root / case["detector"] / "scene/static_walker_model.json"),
                                "--output", str(audit)], stdout=log, stderr=subprocess.STDOUT, check=True)
                    metrics["hand_geometry"] = read(audit)["summary"]
                    if case["factor"] == "baseline":
                        frozen_directory = Path(plan["source_commands"][case["detector"]]).parent
                        old = load_archive(frozen_directory / "fit/result.npz")
                        # The old viewer wrapper appends this export-only array.
                        old.pop("vertices_ground_m", None)
                        metrics["frozen_result_parity"] = compare_archives(old, result)
                    np.savez_compressed(directory / "diagnostics_v2.npz", **arrays)
                    write(output, metrics)
                    print(json.dumps({"evaluated": case["id"]}), flush=True)
                except Exception as error:
                    write(directory / "EVALUATION_FAILURE.json", {"error": str(error), "traceback": traceback.format_exc()})
                    rows.append({"id": case["id"], "status": "evaluation_failed"})
                    continue
            rows.append({"id": case["id"], "status": "completed", "result_directory": str(directory),
                         "first_attempt_failure": first_failure, "metrics": read(output)})
        suffix = "_after_fix" if cli.retries else ""
        write(cli.run / f"results{suffix}.json", {"cases": rows, "retry_root": str(cli.retries) if cli.retries else None,
              "evidence": "engineering diagnostics; same diagnostic wrist and source observations; no independent3D truth"})
        fields = ["id", "status", "body_rms_mm", "weighted_body_mm", "wrist_left_p95_mm", "wrist_right_p95_mm", "leg_acceleration_rms_m_s2", "leg_motion_rms_mm", "sole_left_min_mm", "sole_right_min_mm", "hand_left_pass", "hand_right_pass", "seconds"]
        with (cli.run / f"results{suffix}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                values = {"id": row["id"], "status": row["status"]}
                if "metrics" in row:
                    m = row["metrics"]
                    values.update(body_rms_mm=m["body_observation_mm"]["rms"], weighted_body_mm=m["reference_quality_weighted_body_rms_mm"],
                        wrist_left_p95_mm=m["wrist_reference_mm"]["left"]["p95"], wrist_right_p95_mm=m["wrist_reference_mm"]["right"]["p95"],
                        leg_acceleration_rms_m_s2=m["leg_relative_acceleration_m_s2"]["rms"], leg_motion_rms_mm=m["leg_motion_excursion_mm"]["rms"],
                        sole_left_min_mm=m["feet"]["left"]["minimum_z_mm"], sole_right_min_mm=m["feet"]["right"]["minimum_z_mm"],
                        hand_left_pass=m["hand_geometry"]["left"]["passed"], hand_right_pass=m["hand_geometry"]["right"]["passed"], seconds=m["seconds"])
                writer.writerow(values)
        if not cli.watch or (cli.run / "execution_summary.json").exists():
            return any(row["status"] != "completed" for row in rows)
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
