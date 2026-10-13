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


def require_exportable_body(detector, options):
    """Keep historical reference25 replay and allow the current reliability fitter."""
    if detector != "pmpose":
        raise ValueError("exporter requires same-source PMPose body observations")
    policy = getattr(options, "fit_policy", "legacy")
    if policy == "reliability":
        from pose_app.smplh_fitting.stages import validate_optimization_options

        validate_optimization_options(options)
        return
    if (policy == "legacy" and options.body_weighting == "reference"
            and [options.body_reference_reprojection_px, options.body_reference_ray_gap_mm,
                 options.body_reference_confidence_floor] == [25.0, 80.0, 0.6]):
        return
    raise ValueError("exporter requires legacy reference25 or a valid reliability body route")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path,
                        help="explicit capture selection using the selected body and scene route")
    parser.add_argument("--resume-current-export", action="store_true",
                        help="resume a failed export with its exact same completed current scene")
    parser.add_argument("--scene-profile", default="mainline",
                        help="explicit isolated scene candidate; default retains formal mainline")
    parser.add_argument("--minimal-text", action="store_true",
                        help="compact title and sidebar; retain diagnostics under details")
    cli = parser.parse_args()
    out = cli.output.resolve()
    if out.exists() and not cli.resume_current_export:
        raise FileExistsError(out)
    cfg = read(ROOT / "realtime_app/configs/current_mainline.json")
    resolved_scene_profile = (cfg["scene_route"]["default_profile"]
                              if cli.scene_profile == "mainline" else cli.scene_profile)
    profiles = read(ROOT / "realtime_app/configs/stage_ground_profiles.json")["profiles"]
    if resolved_scene_profile not in profiles:
        raise ValueError("unknown selected scene profile")
    sel = read(cli.selection) if cli.selection else cfg["mainline_selection"]
    body = sel["body"]
    smoothing = body.get("display_smoothing", False)
    if smoothing and not cli.selection:
        raise ValueError("display smoothing requires an explicit isolated selection")
    options = fit_arguments(ROOT / body["reproduction_arguments"])
    require_exportable_body(sel["detector"], options)
    if options.fit_policy == "reliability":
        fitted_summary = read((ROOT / body["result"]).parent / "fit_summary.json")
        if fitted_summary.get("fit_policy") != options.fit_policy:
            raise ValueError("selected fitted result policy differs from its reproduction arguments")
    frames = read(ROOT / sel.get("frames_metadata", str(Path(sel["dataset"]) / "frames.json")))
    if len(frames) != sel["frames"]:
        raise ValueError("frame metadata count differs from selection")
    if cli.resume_current_export:
        previous = read(out / "protocol.json")
        if (not (out / "FAILURE.json").exists() or previous["selected_body"] != body
                or previous["detector_inputs"] != sel["detector_inputs"]
                or not (out / "scene/scene_transforms.npz").exists()):
            raise ValueError("resume requires same failed export and completed current scene")
    else:
        out.mkdir(parents=True)
    write(out / "protocol.json", {
        "selected_body": body, "frames": sel["frames"], "frame_ids": sel["frame_ids"],
        "source_pair_ids": [int(row["pair_id"]) for row in frames],
        "scene_profile": ("mainline -> " + resolved_scene_profile
                          if cli.scene_profile == "mainline" else cli.scene_profile),
        "detector_inputs": sel["detector_inputs"],
        "implementation": "realtime_app/build_current_mainline_view.py",
        "variable": "Visualization export of selected body; fresh strict geometry and selected scene replay, no fitting changes",
        "success": "Complete real mesh, original/rejected observations, same-frame transforms, solid walker and cameras; file and browser verification",
        "stop": "Missing assets or source/frame/geometry mismatch: preserve failure and stop; no relaxed gates or substituted frames",
        "old_dynamic_inputs_consumed": False, "body_refitted": False,
        "body_fit_policy": options.fit_policy,
        "body_observation_mode": options.body_observation_mode,
        "display_smoothing": smoothing, "interpolation": False,
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
        strict_text = "".join(json.dumps(row, allow_nan=False) + "\n" for row in strict)
        if cli.resume_current_export:
            if strict_path.read_text(encoding="utf-8") != strict_text:
                raise ValueError("current strict observations changed during export recovery")
        else:
            strict_path.write_text(strict_text, encoding="utf-8")
        command = read(ROOT / sel["scene_reproduction_command"])
        command[0] = sys.executable
        for flag, value in (("--output", str(out / "scene")),
                            ("--strict-jsonl", str(strict_path)),
                            ("--calibration", str(calibration_path)),
                            ("--scene-profile", cli.scene_profile)):
            command[command.index(flag) + 1] = value
        if cli.resume_current_export:
            if read(out / "scene_replay_command.json") != command:
                raise ValueError("scene replay command changed during recovery")
        else:
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
        if cli.resume_current_export:
            with np.load(out / "result.npz", allow_pickle=False) as existing:
                for key, value in z.items():
                    if (key not in existing or not np.array_equal(
                            value, existing[key], equal_nan=value.dtype.kind in "biufc")):
                        raise ValueError(f"fitted body changed during export recovery: {key}")
        with np.load(out / "scene/scene_transforms.npz", allow_pickle=True) as scene:
            scene_rotation = scene["rotation_ground_from_left"].copy()
            scene_translation = scene["translation_ground_from_left_mm"].copy()
            z["vertices_ground_m"] = np.einsum("nij,nvj->nvi", scene_rotation,
                                                z["vertices"]) + scene_translation[:, None] / 1000
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
        audit_command = [sys.executable, "-B", "realtime_app/run_tool.py",
                "body.audit_constructed_grasp_body", "--result", str(out / "result.npz"),
                "--walker-model", static_walker,
                "--output", str(out / "grasp_geometry_audit.json")]
        if cli.resume_current_export:
            if read(out / "geometry_audit_command.json") != audit_command:
                raise ValueError("geometry audit command changed during recovery")
            read(out / "grasp_geometry_audit.json")
            if (out / "body.html").exists():
                failed_path = out / "body_failed_initial.html"
                suffix = 1
                while failed_path.exists():
                    failed_path = out / f"body_failed_{suffix}.html"
                    suffix += 1
                (out / "body.html").rename(failed_path)
        else:
            execute(out, "geometry_audit", audit_command)
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
        if smoothing:
            import base64
            for key, source_key in (("raw_v", "raw_vertices"), ("raw_j", "raw_predicted_coco")):
                ground = (np.einsum("nij,nvj->nvi", scene_rotation, z[source_key])
                          + scene_translation[:, None] / 1000)
                payload[key] = base64.b64encode(ground.astype("<f4").tobytes()).decode("ascii")
            payload["display_smoothing"] = smoothing
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
        if getattr(options, "body_observation_mode", "legacy") == "reprojection_only":
            page = page.replace("reference25 权重", "三维诊断权重（不用于二维拟合）")
            page = page.replace("原质量 → 实际拟合权重", "原质量 → 三维诊断权重")
        elif options.fit_policy == "reliability":
            page = page.replace("reference25 权重", "可靠程度权重")
        if resolved_scene_profile == "supported_stop_fixed_lag":
            page = page.replace("held:'保持',", "protocol_support_restored:'协议支撑闭合',"
                                "protocol_support_window_reestimated:'过渡窗口重估',held:'保持',")
            page = page.replace("<section><h2>操作</h2>",
                '<section><h2>相机停止与支撑协议</h2><p>动作Stage标签保留；相机可靠停止时按用户采集协议恢复平地支撑，'
                '不等待人体重新运动。固定延迟为15帧（名义30FPS约0.5秒）；这是先验辅助的位姿重估，不是触地测量。'
                '缺失观测、原失败及修正前位姿均保留。</p></section><section><h2>操作</h2>')
        if smoothing:
            hook = "]).then(([v,w,t,he,c,ankle,model])=>{"
            if hook not in page:
                raise ValueError("viewer smoothing toggle hook missing")
            page = page.replace(hook, hook + "const smoothedV=v,smoothedModel=model,rawV=unpack(d.raw_v),rawModel=unpack(d.raw_j);"
                                "document.getElementById('smoothModel').onchange=()=>{const enabled=document.getElementById('smoothModel').checked;"
                                "v=enabled?smoothedV:rawV;model=enabled?smoothedModel:rawModel;draw();window.viewerAudit.smoothing=enabled;};")
            page = page.replace("<section><h2>操作</h2>",
                                '<section><h2>SMPL-H 显示</h2><label><input id="smoothModel" type="checkbox" checked>少量参数平滑</label>'
                                '<p>邻帧权重各0.1；每关节旋转修正最多2°，平移最多3mm。取消可看原始拟合。'
                                '观测、Stage、地面、相机、助步器不平滑；几何诊断栏针对平滑版本。</p></section><section><h2>操作</h2>')
        if cli.minimal_text:
            from walker_tools.visualization.minimal_viewer_text import minimal_viewer_text
            page = minimal_viewer_text(page, sel.get("minimal_visualization_title", "人体与助步器"))
        page_path.write_text(page, encoding="utf-8")
        # Keep the standalone page and a compact browser entry for every
        # fit, including unsmoothed meshes with a large numeric payload.
        import gzip
        packed = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.DOTALL)
        (out / "body_data.json.gz").write_bytes(gzip.compress(packed[1].encode("utf-8"), compresslevel=1))
        browser_page = page[:packed.start()] + page[packed.end():]
        browser_page = browser_page.replace(
            "const d=JSON.parse(document.getElementById('data').textContent),canvas=",
            "function startViewer(d){const canvas=", 1)
        browser_page = browser_page.replace("</script></html>",
            "}fetch('body_data.json.gz').then(async response=>{if(!response.ok)throw Error('viewer data '+response.status);"
            "return new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).json();})"
            ".then(startViewer).catch(error=>{document.getElementById('qa').textContent='加载失败：'+error;});</script></html>")
        (out / "body_browser.html").write_text(browser_page, encoding="utf-8")
        if smoothing:
            validation_path = out / "viewer_validation.json"
            validation = read(validation_path)
            validation.update(no_display_smoothing=False, explicit_user_requested_smoothing=smoothing,
                              original_fit_toggle_available=True)
            write(validation_path, validation)
        if cli.resume_current_export:
            write(out / "RECOVERY.json", {"status": "export_recovered", "failure_preserved": "FAILURE.json",
                  "same_strict_input_and_scene_command_verified": True,
                  "completed_current_scene_replayed_again": False,
                  "fix": "copy scene R/t before closing NPZ; render the explicit smoothing toggle"})
        print(page_path, flush=True)
    except Exception as error:
        write(out / "FAILURE.json", {"error": repr(error), "outputs_preserved": True})
        raise


if __name__ == "__main__":
    main()
