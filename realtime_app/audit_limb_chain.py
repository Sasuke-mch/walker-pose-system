"""Read-only limb/scene attribution from frozen, explicitly aligned observations.

No detector, fitter, labels, masks or scene estimator are modified. Historical
smoothed points are reported separately and never fed to the current chain.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compare_capture_arm_bias import RECORDS, ROOT, project, read, stats
from experiment_arm_sensitivity import sensitivity
from PIL import Image, ImageDraw, ImageOps
from pose_app.body_observations import raw_side, raw_triangulate
from pose_app.fisheye_camera import load_stereo_fisheye
from scipy.spatial.transform import Rotation

NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
         "right_knee", "left_ankle", "right_ankle"]
NATIVE = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21,
          11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
         (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def transform(x, r, t):
    return np.einsum("tij,tkj->tki", r, x) + t[:, None]


def step_parts(x, r, t):
    """Exact step = current-R camera step + pose step acting on previous x."""
    camera = np.einsum("tij,tkj->tki", r[1:], np.diff(x, axis=0))
    pose = np.einsum("tij,tkj->tki", np.diff(r, axis=0), x[:-1]) + np.diff(t, axis=0)[:, None]
    world = np.diff(transform(x, r, t), axis=0)
    np.testing.assert_allclose(world, camera + pose, atol=1e-9)
    return world, camera, pose


def temporal(x, mask):
    finite = np.isfinite(x).all(-1)
    mask = mask & finite
    m1 = mask[1:] & mask[:-1]
    m2 = mask[2:] & mask[1:-1] & mask[:-2]
    return {"step": stats(np.linalg.norm(np.diff(x, axis=0), axis=-1)[m1]),
            "second_difference": stats(np.linalg.norm(np.diff(x, n=2, axis=0), axis=-1)[m2])}


def bend(x, indices):
    a, b, c = (x[:, k] for k in indices)
    u, v = a - b, c - b
    den = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    angle = np.rad2deg(np.arccos(np.clip(np.sum(u * v, axis=-1) / np.maximum(den, 1e-12), -1, 1)))
    return 180 - angle


def segment_smooth(x, mask):
    """Diagnostic symmetric [0.15,0.7,0.15] only inside valid triples.

    Missing slots stay unavailable. This is not an archived current output.
    """
    y = x.copy()
    m = mask[2:] & mask[1:-1] & mask[:-2]
    filtered = .15 * x[:-2] + .7 * x[1:-1] + .15 * x[2:]
    y[1:-1][m] = filtered[m]
    y[~mask] = np.nan
    return y


def draw_overlay(fid, camera, images, observed, raw_px, model_px, native_px, mask, output):
    im = Image.open(images / ("left_ccw90" if camera == "left" else "right_cw90") / f"pair_{fid:04d}.png").convert("RGB")
    draw = ImageDraw.Draw(im)

    def upright(p):
        return (p[1], 1919 - p[0]) if camera == "left" else (1079 - p[1], p[0])

    colors = ["#00bb00", "#00bddd", "#ff2222", "#ff9900"]
    arrays = [observed[fid, :, :2], raw_px[fid], model_px[fid], native_px[fid]]
    for k, (points, color) in enumerate(zip(arrays, colors, strict=True)):
        for a, b in EDGES:
            if k == 0 and np.isfinite(points[[a, b]]).all():
                draw.line([upright(points[a]), upright(points[b])], fill="black", width=2)
        for j in range(5, 17):
            if not np.isfinite(points[j]).all():
                continue
            u, v = upright(points[j])
            if k == 0:
                draw.ellipse((u-6, v-6, u+6, v+6), outline=color, width=3)
                draw.text((u+7, v+5), str(j), fill=color, stroke_width=1, stroke_fill="black")
            elif k == 1:
                draw.rectangle((u-4, v-4, u+4, v+4), outline=color, width=2)
            else:
                size = 10 if k == 2 else 6
                draw.line((u-size, v, u+size, v), fill=color, width=3)
                draw.line((u, v-size, u, v+size), fill=color, width=3)
            if k == 0 and not mask[fid, j]:
                draw.text((u+7, v-14), "strict rejected", fill="white", stroke_width=1, stroke_fill="black")
    draw.rectangle((0, 0, im.width, 54), fill="white")
    draw.text((10, 6), f"frame {fid} / {camera} | green:2D circle  cyan:DLT square  red:SMPLH COCO +  orange:native +", fill="black")
    draw.text((10, 28), "Frozen observation audit; camera space; no smoothing, relabeling or scene replay", fill="black")
    im.save(output / f"frame_{fid:04d}_{camera}.png")


def legacy_report():
    folder = RECORDS / "G20260923_smpl_clean_full_sequence_v1"
    result = {}
    for name, rp, sp, rawkey, smoothkey, maskkey in [
        ("old448_visual_kernel", folder / "triangulation_skeleton_visualization_v1/skeleton_triangulation.npz",
         folder / "triangulation_skeleton_visualization_v1/skeleton_triangulation.npz", "raw_triangulated_points", "temporally_processed_points", "accepted_mask"),
        ("old448_quadratic_solve", folder / "run_full448_v6/triangulation.npz", folder / "run_full448_v6/temporal_points.npz",
         "raw", "temporally_processed_points", "accepted")]:
        with np.load(rp) as a, np.load(sp) as b:
            x, y, mask = a[rawkey], b[smoothkey], a[maskkey]
            result[name] = {"raw_source": str(rp), "smooth_source": str(sp), "frames": len(x),
                            "not_current_360_input": True, "right_knee_accepted": int(mask[:, 14].sum()),
                            "joints": {NAMES[j]: {"raw": temporal(x[:, j], mask[:, j]),
                                                  "smooth": temporal(y[:, j], mask[:, j]),
                                                  "displacement_mm": stats(np.linalg.norm(y[:, j]-x[:, j], axis=-1)[mask[:, j]])}
                                       for j in range(5, 17)}}
    return result


def prior_report(baseline):
    path = RECORDS / "G20261005_pmpose_prior_ablation_v1/runs/new_no_vposer_prior/fit/result.npz"
    with np.load(path) as a:
        for key in ("pair_id", "raw_triangulated_points", "body_accepted", "body_quality"):
            np.testing.assert_array_equal(a[key], baseline[key])
        candidates = {"baseline": baseline["predicted_coco"].astype(float)*1000,
                      "no_latent_quadratic_prior": a["predicted_coco"].astype(float)*1000}
    result = {"source": str(path), "not_new_fit": True, "scope": "Existing same-capture cold-start single-weight ablation; no independent truth", "metrics": {}}
    for label, points in candidates.items():
        result["metrics"][label] = {"joint_residual_median_mm": {NAMES[j]: stats(np.linalg.norm(points[:, j]-baseline["raw_triangulated_points"][:, j], axis=-1))["median"] for j in range(5, 17)},
            "bend_median_deg": {name: stats(bend(points, indices))["median"] for name, indices in [("left_elbow", (5, 7, 9)), ("right_elbow", (6, 8, 10)), ("left_knee", (11, 13, 15)), ("right_knee", (12, 14, 16))]}}
    return result


def filtered_stage_feet(sc, ground):
    """Reconstruct the archived estimator's actual five-valid-sample median.

    This filter only feeds ground pose. It does not filter the SMPLH targets.
    """
    from collections import deque

    history = deque(maxlen=5)
    filtered = np.full((len(ground), 2, 3), np.nan)
    latched, carry = False, 0
    for f, row in enumerate(ground):
        if row["stage"].startswith("stage2"):
            active, latched, carry = True, True, 0
        elif row["stage"] == "transition" and latched and carry < 4:
            carry += 1
            active = True
        else:
            active = False
            if row["stage"] != "transition" or carry >= 4:
                latched = False
        if not active:
            history.clear()
        elif sc["accepted"][f, 15:17].all():
            history.append(sc["raw_triangulated_points_mm"][f, 15:17])
            filtered[f] = np.median(np.asarray(history), axis=0)
        if "causal_foot_window_frames" in row["pose_audit"]:
            np.testing.assert_equal(len(history), row["pose_audit"]["causal_foot_window_frames"])
    return filtered


def contact_sheet(overlays, output, name, ids, box):
    canvas = Image.new("RGB", (1620, 1400), "white")
    draw = ImageDraw.Draw(canvas)
    for row, camera in enumerate(("left", "right")):
        for col, fid in enumerate(ids):
            source = Image.open(overlays / f"frame_{fid:04d}_{camera}.png").crop(box)
            patch = ImageOps.contain(source, (540, 670))
            canvas.paste(patch, (col*540+(540-patch.width)//2, row*700+30))
            draw.text((col*540+10, row*700+10), f"frame {fid} / {camera}", fill="black")
    canvas.save(output / (name + ".png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    route = read(ROOT / "realtime_app/configs/current_mainline.json")["mainline_selection"]
    scene = ROOT / route["scene_reference"]
    fit = ROOT / route["body"]["result"]
    detector = route["detector_inputs"]
    protocol = {"code": "realtime_app/audit_limb_chain.py", "fit": str(fit), "scene": str(scene),
                "detector_inputs": detector, "frames": 360,
                "objective": "Locate observation, geometry, fit and ground-transform contributions before any method changes",
                "controls": "Frozen current input, calibration, masks, body parameters and archived baseline scene; no inference/refit/replay",
                "diagnostic_variable": "Algebraic removal of ground transform; isolated within-valid-triple kernel as explicit counterfactual only",
                "success": "Recomputed raw XYZ/mask/quality match saved body; exact world-step decomposition and same-frame residual invariance; visual audit of preset and ranked frames",
                "stop": "Frame/mask/source mismatch, non-rigid scene, unavailable images; preserve failures and do not substitute",
                "units": "2D px; 3D mm; indexed step mm/frame and second difference mm/frame^2, not physical acceleration",
                "limits": "No anatomical 3D or stage truth; motion includes actual movement; pose component is an algebraic term, not a percentage of error",
                "scene_usage": "Saved scene is an audited input, not a newly replayed scene or new 3D visualization",
                "visual_selection": "Preset 0,42,74,101,136,139,184,226,268,313,359 plus maximum arm residual, camera raw knee second difference, model pelvis-relative knee second difference and five largest world pose steps; include neighbours for ranked events"}
    save(args.output / "protocol.json", protocol)
    with np.load(fit) as a:
        fields = ["pair_id", "raw_triangulated_points", "body_accepted", "body_quality", "predicted_coco", "smplh_joints", "global_orient", "root_init", "vposer_latent"]
        data = {k: a[k].copy() for k in fields}
    # Local archive's object array contains only stage strings.
    with np.load(scene / "scene_transforms.npz", allow_pickle=True) as a:
        sc = {k: a[k].copy() for k in a.files}
    n = len(data["pair_id"])
    np.testing.assert_array_equal(data["pair_id"], np.arange(360))
    ground = [json.loads(l) for l in (scene / "dynamic_ground_pose.jsonl").read_text(encoding="utf-8").splitlines()]
    np.testing.assert_array_equal([r["pair_id"] for r in ground], np.arange(n))
    r, t = sc["rotation_ground_from_left"], sc["translation_ground_from_left_mm"]
    np.testing.assert_allclose(r @ r.transpose(0, 2, 1), np.broadcast_to(np.eye(3), r.shape), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(r), 1, atol=1e-9)
    np.testing.assert_allclose(r, [a["rotation_world_from_left_camera"] for a in ground], atol=1e-9)
    np.testing.assert_allclose(t, [a["translation_world_from_left_camera_mm"] for a in ground], atol=1e-9)
    command = read(ROOT / route["body"]["source_command"])
    calibration_path = ROOT / command[command.index("--calibration-dir") + 1]
    protocol["fit_source_command"] = str(ROOT / route["body"]["source_command"])
    protocol["calibration"] = str(calibration_path)
    protocol["execution_command"] = [".venv-cuda/Scripts/python.exe", "-B",
                                     "realtime_app/audit_limb_chain.py", "--output", str(args.output)]
    save(args.output / "protocol.json", protocol)
    cal = load_stereo_fisheye(calibration_path)
    obs = {side: np.stack([a[i] for i in range(n)]).astype(np.float32) for side in ("left", "right")
           for a in [raw_side(ROOT / detector[side], side)]}
    tri = raw_triangulate(obs["left"], obs["right"], cal)
    x, mask, q = tri[0], tri[6], tri[8]
    np.testing.assert_allclose(x, data["raw_triangulated_points"], rtol=0, atol=0, equal_nan=True)
    np.testing.assert_array_equal(mask, data["body_accepted"])
    np.testing.assert_array_equal(q, data["body_quality"])
    model = data["predicted_coco"].astype(float) * 1000
    native = np.full_like(model, np.nan)
    for j, k in NATIVE.items():
        native[:, j] = data["smplh_joints"][:, k].astype(float) * 1000
    relative = model - model[:, [11, 12]].mean(1)[:, None]
    roots = Rotation.from_rotvec(data["global_orient"]).as_matrix()
    local = np.einsum("tji,tkj->tki", roots, relative)
    smooth = segment_smooth(x, mask)
    mw, mc, mp = step_parts(model, r, t)
    world_model = transform(model, r, t)
    world_raw = transform(x, r, t)
    residual = np.linalg.norm(model - x, axis=-1)
    invariant_error = float(np.nanmax(np.abs(np.linalg.norm(world_model-world_raw, axis=-1)-residual)))
    np.testing.assert_allclose(invariant_error, 0, atol=1e-8)
    dt = np.diff(sc["pair_timestamp_sec"])
    boundaries = np.r_[False, sc["stage"][1:] != sc["stage"][:-1]]
    pose_status = np.array([a["pose_audit"]["status"] for a in ground])
    report = {"frames": n, "level": "engineering_diagnostic_not_truth", "alignment": {"raw_max_difference_mm": float(np.nanmax(np.abs(x-data["raw_triangulated_points"]))),
              "raw_mask_and_quality_equal": True, "rigid_residual_max_difference_mm": invariant_error},
              "timestamps": {"dt_sec": stats(dt), "left_right_return_offset_ms": stats(1000*np.abs(sc["left_timestamp_sec"]-sc["right_timestamp_sec"])),
                             "interpretation": "Host-return times are not certified exposure timestamps; variable intervals preclude constant30Hz physical acceleration"},
              "stage_counts": dict(Counter(sc["stage"])), "pose_counts": dict(Counter(pose_status)),
              "stage_changes": np.flatnonzero(boundaries).tolist(), "joints": {}, "bones": {}, "angles": {}}
    initial_roots = Rotation.from_rotvec(data["root_init"]).as_matrix()
    report["root_rotation"] = {"initial_step_deg": stats(np.rad2deg(Rotation.from_matrix(initial_roots[1:] @ initial_roots[:-1].transpose(0, 2, 1)).magnitude())),
        "fitted_step_deg": stats(np.rad2deg(Rotation.from_matrix(roots[1:] @ roots[:-1].transpose(0, 2, 1)).magnitude())),
        "fit_minus_initial_deg": stats(np.rad2deg(Rotation.from_matrix(roots @ initial_roots.transpose(0, 2, 1)).magnitude())),
        "source": "Per-frame shoulder/hip rigid basis; root anchored to each individual initialization, not to adjacent frames"}
    arrays = {"raw_mm": x, "counterfactual_smooth_mm": smooth, "model_coco_mm": model, "native_mm": native,
              "mask_body": mask, "mask_strict": sc["accepted"], "quality": q, "model_step_world_mm": mw,
              "model_step_camera_term_mm": mc, "model_step_pose_term_mm": mp, "stage_boundary": boundaries}
    px = {kind: {side: project(points / 1000, cal, side) for side in obs}
          for kind, points in [("raw", x), ("model", model), ("native", native), ("smooth", smooth)]}
    for j in range(5, 17):
        sigmas = [sensitivity(point, cal) for point in x[:, j]]
        entry = {"body_accepted": int(mask[:, j].sum()), "strict_accepted": int(sc["accepted"][:, j].sum()),
                 "quality": stats(q[:, j]), "confidence_left": stats(obs["left"][:, j, 2]), "confidence_right": stats(obs["right"][:, j, 2]),
                 "ray_gap_mm": stats(tri[3][:, j]), "reprojection_left_px": stats(tri[1][:, j]), "reprojection_right_px": stats(tri[2][:, j]),
                 "fit_residual_mm": stats(residual[:, j][mask[:, j]]), "native_minus_coco_mm": stats(np.linalg.norm(native[:, j]-model[:, j], axis=-1)),
                 "fixed_calibration_1px_xyz_rms_mm": stats([np.sqrt(np.trace(s[0])) for s in sigmas if s is not None]),
                 "temporal": {name: temporal(a[:, j], mask[:, j]) for name, a in [("raw_camera", x), ("diagnostic_smooth_camera", smooth),
                               ("model_camera", model), ("model_pelvis_relative", relative), ("model_root_local", local), ("model_world", world_model), ("raw_world", world_raw)]},
                 "pixel": {side: {"observed": temporal(obs[side][:, j, :2], mask[:, j]),
                                    "fit_minus_observation_px": stats(np.linalg.norm(px["model"][side][:, j]-obs[side][:, j, :2], axis=-1)),
                                    "smooth_minus_observation_px": stats(np.linalg.norm(px["smooth"][side][:, j]-obs[side][:, j, :2], axis=-1))} for side in obs},
                 "world_step_decomposition_mm": {name: stats(np.linalg.norm(a[:, j], axis=-1)) for name, a in [("world", mw), ("camera_term", mc), ("pose_term", mp)]}}
        report["joints"][NAMES[j]] = entry
    for side, pairs in {"left": [(5, 7), (7, 9), (11, 13), (13, 15)], "right": [(6, 8), (8, 10), (12, 14), (14, 16)]}.items():
        for label, (a, b) in zip(["upper_arm", "forearm", "thigh", "shank"], pairs, strict=True):
            valid = mask[:, a] & mask[:, b]
            report["bones"][side + "_" + label] = {kind: {"length_mm": stats(np.linalg.norm(points[:, a]-points[:, b], axis=-1)[valid]),
                "length_step_mm": stats(np.abs(np.diff(np.linalg.norm(points[:, a]-points[:, b], axis=-1)))[valid[1:] & valid[:-1]])}
                for kind, points in [("raw", x), ("diagnostic_smooth", smooth), ("model_coco", model), ("native", native)]}
        for label, indices in [("elbow", pairs[0]+(pairs[1][1],)), ("knee", pairs[2]+(pairs[3][1],))]:
            valid = mask[:, indices].all(1)
            report["angles"][side + "_" + label] = {kind: {"bend_deg": stats(bend(points, indices)[valid]),
                "bend_step_deg": stats(np.abs(np.diff(bend(points, indices)))[valid[1:] & valid[:-1]])}
                for kind, points in [("raw", x), ("model_coco", model), ("native", native)]}
    pose_norm = np.linalg.norm(mp[:, 13:17], axis=-1).mean(1)
    for label, subset in {"stage_boundary": boundaries[1:], "not_boundary": ~boundaries[1:],
                          "reanchor": pose_status[1:] == "landed_support_reanchored",
                          "held": pose_status[1:] == "held", "accepted": np.char.startswith(pose_status[1:], "accepted")}.items():
        report.setdefault("scene_groups", {})[label] = {"world_leg_step_mm": stats(np.linalg.norm(mw[:, 13:17], axis=-1)[subset]),
            "camera_leg_step_mm": stats(np.linalg.norm(mc[:, 13:17], axis=-1)[subset]),
            "pose_leg_step_mm": stats(np.linalg.norm(mp[:, 13:17], axis=-1)[subset])}
    ranked = {"arm_residual": int(np.argmax(residual[:, 7:11].mean(1))),
              "raw_knee_second_difference": int(np.argmax(np.linalg.norm(np.diff(x[:, 13:15], n=2, axis=0), axis=-1).mean(1)))+1,
              "model_relative_knee_second_difference": int(np.argmax(np.linalg.norm(np.diff(relative[:, 13:15], n=2, axis=0), axis=-1).mean(1)))+1,
              "scene_pose_leg_step": int(np.argmax(pose_norm))+1}
    pose_ranked_frames = (np.argsort(-pose_norm)[:5]+1).tolist()
    selected = sorted({0, 42, 74, 101, 136, 139, 184, 226, 268, 313, 359} |
                      {f+d for f in list(ranked.values())+pose_ranked_frames for d in (-1, 0, 1) if 0 <= f+d < n})
    report["ranked_visual_events"] = ranked
    report["visual_frames"] = selected
    events = []
    for f in np.argsort(-pose_norm)[:15]+1:
        events.append({"frame": int(f), "stage_before": str(sc["stage"][f-1]), "stage_after": str(sc["stage"][f]),
                       "pose_audit": ground[f]["pose_audit"], "model_world_leg_step_mm": np.linalg.norm(mw[f-1, 13:17], axis=-1).tolist(),
                       "model_camera_leg_step_mm": np.linalg.norm(mc[f-1, 13:17], axis=-1).tolist(),
                       "model_pose_leg_step_mm": np.linalg.norm(mp[f-1, 13:17], axis=-1).tolist()})
    report["largest_pose_events"] = events
    report["historical_smoothing"] = legacy_report()
    report["existing_prior_control"] = prior_report(data)
    filtered_feet = filtered_stage_feet(sc, ground)
    arrays["archived_stage_filtered_feet_mm"] = filtered_feet
    report["actual_stage_foot_filter"] = {"role": "Five latest valid two-foot samples in camera coordinates, active stage2/carry only; not body target smoothing",
        "samples": int(np.isfinite(filtered_feet).all((1, 2)).sum()),
        "displacement_from_current_strict_feet_mm": stats(np.linalg.norm(filtered_feet-sc["raw_triangulated_points_mm"][:, 15:17], axis=-1)),
        "history_counts_match_saved_pose_audit": True,
        "limit": "Median mixes previous camera-frame measurements during camera movement; noise suppression and lag cannot be separated without a controlled estimator replay"}
    report["numeric_visual_events"] = [{"frame": f, "stage": str(sc["stage"][f]), "pose": ground[f]["pose_audit"],
        "raw_knee_step_mm": np.linalg.norm(x[f, 13:15]-x[max(0, f-1), 13:15], axis=-1).tolist(),
        "model_knee_step_mm": np.linalg.norm(model[f, 13:15]-model[max(0, f-1), 13:15], axis=-1).tolist(),
        "raw_left_pixel_knee_step_px": np.linalg.norm(obs["left"][f, 13:15, :2]-obs["left"][max(0, f-1), 13:15, :2], axis=-1).tolist(),
        "raw_right_pixel_knee_step_px": np.linalg.norm(obs["right"][f, 13:15, :2]-obs["right"][max(0, f-1), 13:15, :2], axis=-1).tolist(),
        "quality_hip_knee_ankle": q[f, 11:17].tolist(), "quality_shoulder_elbow_wrist": q[f, 5:11].tolist()}
        for f in selected]
    save(args.output / "summary.json", report)
    np.savez_compressed(args.output / "audit_arrays.npz", **arrays)
    overlays = args.output / "overlays"
    overlays.mkdir()
    for fid in selected:
        for side, observation in obs.items():
            draw_overlay(fid, side, ROOT / detector["images"], observation, px["raw"][side], px["model"][side], px["native"][side], sc["accepted"], overlays)
    for name, ids, box in [("knee_320_322", [320, 321, 322], (380, 1050, 920, 1720)),
                           ("raw_137_139", [137, 138, 139], (380, 1050, 920, 1720)),
                           ("arm_136_231", [136, 184, 231], (0, 170, 1080, 940)),
                           ("stage_298_300", [298, 299, 300], (250, 900, 1000, 1800))]:
        contact_sheet(overlays, args.output, name, ids, box)
    for name, ids in [("early", [0, 42, 74]), ("middle", [101, 139, 226]), ("late", [268, 313, 359])]:
        contact_sheet(overlays, args.output, name+"_arms", ids, (0, 140, 1080, 1000))
        contact_sheet(overlays, args.output, name+"_legs", ids, (250, 1000, 1050, 1850))
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(residual[:, 7:11].mean(1), label="arm mean fit-raw")
    axes[0].plot(residual[:, 13:17].mean(1), label="leg mean fit-raw")
    axes[0].set_ylabel("Residual mm")
    axes[1].plot(np.arange(1, n), np.linalg.norm(np.diff(x[:, 13:17], axis=0), axis=-1).mean(1), label="raw camera")
    axes[1].plot(np.arange(1, n), np.linalg.norm(np.diff(model[:, 13:17], axis=0), axis=-1).mean(1), label="SMPLH camera")
    axes[1].set_ylabel("Leg step mm/frame")
    axes[2].plot(np.arange(1, n), np.linalg.norm(mw[:, 13:17], axis=-1).mean(1), label="world total")
    axes[2].plot(np.arange(1, n), pose_norm, label="pose term")
    axes[2].plot(np.arange(1, n), np.linalg.norm(mc[:, 13:17], axis=-1).mean(1), label="camera term")
    axes[2].set_ylabel("Exact vector terms\n(norms do not add)")
    axes[3].plot(bend(model, (11, 13, 15)), label="SMPLH left knee bend")
    axes[3].plot(bend(model, (12, 14, 16)), label="SMPLH right knee bend")
    axes[3].set_ylabel("Bend deg")
    for ax in axes:
        for f in np.flatnonzero(boundaries):
            ax.axvline(f, color="black", alpha=.12, linewidth=.7)
        ax.legend(loc="upper right", ncol=3)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Fit frame index (0..359); vertical lines are operational Stage changes")
    fig.tight_layout()
    fig.savefig(args.output / "chain_timeline.png", dpi=150)
    plt.close(fig)
    save(args.output / "validation.json", {"alignment_and_rigid_checks": "passed", "decomposition": "passed",
         "images_generated": len(selected)*2, "visual_model_review": "pending", "no_input_mutation": True})
    print(json.dumps({"ranked": ranked, "output": str(args.output), "joints": {name: {"residual": entry["fit_residual_mm"]["median"],
          "raw_d2_p95": entry["temporal"]["raw_camera"]["second_difference"]["p95"],
          "model_d2_p95": entry["temporal"]["model_camera"]["second_difference"]["p95"],
          "world_d2_p95": entry["temporal"]["model_world"]["second_difference"]["p95"]} for name, entry in report["joints"].items()}}, indent=2))


if __name__ == "__main__":
    main()
