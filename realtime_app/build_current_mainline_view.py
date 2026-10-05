"""Export the selected body with freshly triangulated/replayed mainline scene."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from check_mainline_regression import fit_arguments
from evaluate_repositioned_capture import strict_first_instances
from pose_app.calibration import StereoCalibration
from pose_app.mainline_regression import observation_snapshot
from walker_tools.stereo.evaluate_offline_stereo_predictions import _records, _result

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def execute(out, name, command):
    write(out / (name + "_command.json"), command)
    print("starting", name, flush=True)
    with (out / (name + ".log")).open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    print("completed", name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path,
                        help="explicit capture selection using the current reference25/baseline logic")
    cli = parser.parse_args()
    out = cli.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    cfg = read(ROOT / "realtime_app/configs/current_mainline.json")
    sel = read(cli.selection) if cli.selection else cfg["mainline_selection"]
    body = sel["body"]
    options = fit_arguments(ROOT / body["reproduction_arguments"])
    if (sel["detector"] != "pmpose" or options.body_weighting != "reference"
            or [options.body_reference_reprojection_px, options.body_reference_ray_gap_mm,
                options.body_reference_confidence_floor] != [25.0, 80.0, 0.6]
            or cfg["scene_route"]["default_profile"] != "baseline"):
        raise ValueError("exporter requires selected PMPose reference25/baseline route")
    frames = read(ROOT / sel.get("frames_metadata", str(Path(sel["dataset"]) / "frames.json")))
    if len(frames) != sel["frames"]:
        raise ValueError("frame metadata count differs from selection")
    out.mkdir(parents=True)
    write(out / "protocol.json", {
        "selected_body": body, "frames": sel["frames"], "frame_ids": sel["frame_ids"],
        "source_pair_ids": [int(row["pair_id"]) for row in frames], "scene_profile": "mainline -> baseline",
        "detector_inputs": sel["detector_inputs"],
        "implementation": "realtime_app/build_current_mainline_view.py",
        "variable": "Visualization export of selected body; fresh strict geometry and baseline replay, no fitting changes",
        "success": "Complete real mesh, original/rejected observations, same-frame transforms, solid walker and cameras; file and browser verification",
        "stop": "Missing assets or source/frame/geometry mismatch: preserve failure and stop; no relaxed gates or substituted frames",
        "old_dynamic_inputs_consumed": False, "body_refitted": False,
        "display_smoothing": False, "interpolation": False,
        "limits": "Engineering display; static installation is an assumption, no independent contact or 3D truth"})
    try:
        observed = observation_snapshot(options)
        np.testing.assert_array_equal(observed["pair_id"], np.arange(sel["frames"]))
        calibration_path = options.calibration_dir / "stereo_fisheye.json"
        calibration = StereoCalibration.load(calibration_path)
        records = {s: _records(Path(sel["detector_inputs"][s]), "pmpose", "raw_keypoint")
                   for s in ("left", "right")}
        strict = []
        for i, frame in enumerate(frames):
            views = {}
            for s, size, rotation in (("left", calibration.left_image_size, "ccw90"),
                                      ("right", calibration.right_image_size, "cw90")):
                record = records[s][i]
                if record["id"] != i or record["name"] != Path(frame[s + "_image"]).name:
                    raise ValueError("detector image/frame identity mismatch")
                views[s] = _result(record, "pmpose", size, rotation, i / 30)
                np.testing.assert_allclose(views[s].persons[0].keypoints,
                                           observed[s + "_raw_pixels"][i],
                                           atol=2e-4, rtol=0)
            people = strict_first_instances(views["left"].persons,
                                            views["right"].persons, calibration)
            strict.append({"index": i, "original_pair_id": int(frame["pair_id"]),
                               "pair_outcome": "matched" if people else "association_failed",
                               "persons_3d": [p.to_dict() for p in people]})
        strict_path = out / "first_instance_strict.jsonl"
        strict_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n"
                                       for row in strict), encoding="utf-8")
        command = read(ROOT / sel["scene_reproduction_command"])
        command[0] = sys.executable
        for flag, value in (("--output", str(out / "scene")),
                            ("--strict-jsonl", str(strict_path)),
                            ("--calibration", str(calibration_path)),
                            ("--scene-profile", "mainline")):
            command[command.index(flag) + 1] = value
        execute(out, "scene_replay", command)
        with np.load(ROOT / body["result"], allow_pickle=False) as archive:
            z = {k: archive[k] for k in archive.files}
        for k, reference in (("pair_id", observed["pair_id"]),
                             ("body_accepted", observed["accepted"]),
                             ("raw_triangulated_points", observed["triangulated_mm"]),
                             ("body_quality", observed["quality"])):
            np.testing.assert_array_equal(z[k], reference)
        if z["vertices"].shape != (len(frames), 6890, 3):
            raise ValueError("selected result is not a full same-frame mesh")
        with np.load(out / "scene/scene_transforms.npz", allow_pickle=True) as scene:
            z["vertices_ground_m"] = np.einsum("nij,nvj->nvi", scene["rotation_ground_from_left"],
                                                z["vertices"]) + scene["translation_ground_from_left_mm"][:, None] / 1000
        np.savez_compressed(out / "result.npz", **z)
        write(out / "run_metadata.json", {
            "cold_start": True, "detector": "pmpose", "viewer_bundle_only": True,
            "source_fitted_result": str((ROOT / body["result"]).resolve()),
            "inputs": {"left_raw": str(options.left_raw), "right_raw": str(options.right_raw),
                        "calibration_dir": str(options.calibration_dir)},
            "allow_diagnostic_wrist_reference": False,
            "hand_position_source": "body_fit_native_finger_rotation_only",
            "old_fitted_parameters_consumed": False, "source_scene": str(out / "scene")})
        scene_sources = read(out / "scene/scene_sources.json")
        static_walker = scene_sources["static_walker_model"]
        if read(static_walker) != read(out / "scene/static_walker_model.json"):
            raise ValueError("scene static walker copy differs from declared source")
        execute(out, "geometry_audit", [sys.executable, "-B", "realtime_app/run_tool.py",
                "body.audit_constructed_grasp_body", "--result", str(out / "result.npz"),
                "--walker-model", static_walker,
                "--output", str(out / "grasp_geometry_audit.json")])
        execute(out, "viewer", [sys.executable, "-B", "realtime_app/run_tool.py",
                "visualization.build_grasp_body_canvas_viewer", "--result", str(out / "result.npz"),
                "--scene", str(out / "scene"), "--surface-sets",
                str(ROOT / "research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json"),
                "--geometry-audit", str(out / "grasp_geometry_audit.json"),
                "--strict-jsonl", str(strict_path), "--output", str(out / "body.html")])
        page_path = out / "body.html"
        page = page_path.read_text(encoding="utf-8")
        title = sel.get("visualization_title", "当前主线 · reference25 · baseline")
        page = re.sub(r"<title>.*?</title>", "<title>" + title + "</title>", page)
        page = re.sub(r"<h1>.*?</h1>", "<h1>" + title + "</h1>", page)
        match = re.search(r'(<script id="data" type="application/json">)(.*?)(</script>)', page, re.DOTALL)
        payload = json.loads(match[2])
        payload["geometry_quality_recomputed"] = payload["quality"]
        payload["quality"] = z["body_quality"].tolist()
        payload["fit_weights"] = z["body_fit_weights"].tolist()
        payload["source_pair_ids"] = [int(row["pair_id"]) for row in frames]
        # Keep original quality separate from the selected effective fit weights.
        page = page[:match.start(2)] + json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                                separators=(",", ":")) + page[match.end(2):]
        page = page.replace("window.viewerAudit.frame=fi;", "window.viewerAudit.frame=fi;"
                "document.getElementById('weights').textContent='原始配对 '+d.source_pair_ids[fi]+'\\n关节: 原质量 → 实际拟合权重\\n'+"
                "d.fit_weights[fi].map((w,j)=>j+': '+d.quality[fi][j].toFixed(4)+' → '+w.toFixed(4)).join('\\n');")
        page = page.replace("<section><h2>操作</h2>",
                '<section><h2>reference25 权重</h2><pre id="weights" style="font-size:12px"></pre>'
                '<p>黑线连接身体原始DLT观测；橙叉为严格场景拒绝。拟合候选门与严格场景门分别记录。'
                '足底红点为低于参考地面；静态扶手检查仅为几何代理。</p></section><section><h2>操作</h2>')
        page_path.write_text(page, encoding="utf-8")
        print(page_path, flush=True)
    except Exception as error:
        write(out / "FAILURE.json", {"error": repr(error), "outputs_preserved": True})
        raise


if __name__ == "__main__":
    main()
