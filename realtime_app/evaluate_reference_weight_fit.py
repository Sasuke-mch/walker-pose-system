"""Evaluate confidence-reference fits on immutable observations and all frames."""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit_limb_chain import NAMES, ROOT, bend, temporal
from compare_capture_arm_bias import project, read, stats
from matplotlib.font_manager import FontProperties
from PIL import Image
from pose_app.body_observations import raw_side, raw_triangulate
from pose_app.fisheye_camera import load_stereo_fisheye
from pose_app.smplh_fitting.losses import observation_weights
from run_smplh_ablations import modified_options
from walker_tools.body.fit_smplh_wilor_sequence import build_parser


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def verify_saved_meshes(experiment):
    """Reconstruct all saved parameters without optimization or scene replay."""
    import torch
    from experiment_local_arm_pose import load_model
    from pose_app.smpl_coco_observation import (
        load_coco17_regressor,
        regress_coco17_torch,
    )

    output = experiment / "mesh_reconstruction.json"
    if output.exists():
        raise FileExistsError(output)
    model = load_model(ROOT / "third_party/WiLoR/mano_data/models")
    reg = load_coco17_regressor(ROOT / "models/smpl/J_regressor_coco.npy")
    rows = {}
    for case in read(experiment/"plan.json")["cases"]:
        path = experiment/"runs"/case["id"]/"fit/result.npz"
        if not path.exists():
            rows[case["id"]] = {"status": "unavailable"}
            continue
        with np.load(path) as archive:
            maximum_vertex, maximum_coco, maximum_joint = 0., 0., 0.
            for start in range(0, 360, 32):
                end = min(start+32, 360)
                inputs = {k: torch.as_tensor(archive[k][start:end], device="cuda", dtype=torch.float32)
                          for k in ("global_orient", "body_pose", "transl", "left_hand_pose", "right_hand_pose")}
                beta = torch.as_tensor(archive["betas"], device="cuda", dtype=torch.float32).expand(end-start, -1)
                with torch.no_grad():
                    body = model(betas=beta, return_verts=True, **inputs)
                    coco = regress_coco17_torch(body.vertices, reg).cpu().numpy()
                maximum_vertex = max(maximum_vertex, float(np.max(np.abs(body.vertices.cpu().numpy()-archive["vertices"][start:end]))))
                maximum_coco = max(maximum_coco, float(np.max(np.abs(coco-archive["predicted_coco"][start:end]))))
                maximum_joint = max(maximum_joint, float(np.max(np.abs(body.joints.cpu().numpy()-archive["smplh_joints"][start:end]))))
            if max(maximum_vertex, maximum_coco, maximum_joint) > 2e-6:
                raise ValueError("saved model reconstruction mismatch")
        rows[case["id"]] = {"frames": 360, "status": "passed", "tolerance_m": 2e-6,
            "max_vertex_difference_m": maximum_vertex, "max_coco_difference_m": maximum_coco,
            "max_native_joint_difference_m": maximum_joint}
    save(output, {"scope": "Saved parameter/mesh equality only; no optimization, interpolation or stage replay", "cases": rows})
    return rows


def figure(experiment, report, data, raw, mask, obs, cal, images, selected):
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 12})
    labels = ["原权重", "参考25px", "参考40px", "等权压力对照"]
    keys = ["quality_replay", "reference25", "reference40", "uniform_stress"]
    keys = [k for k in keys if k in data]
    labels = labels[:len(keys)]
    colors = ["#666666", "#188952", "#258aca", "#b88626"][:len(keys)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor="white")
    metrics = [("肘腕观测残差均值", "mm", [report["cases"][k]["groups"]["arms"]["residual_mm"]["mean"] for k in keys]),
               ("膝踝二阶差分P95", "mm/frame²", [report["cases"][k]["groups"]["legs"]["second_difference"]["p95"] for k in keys]),
               ("右膝321→322单步", "mm", [report["cases"][k]["right_knee_321_to_322_mm"] for k in keys])]
    for ax, (title, unit, values) in zip(axes, metrics, strict=True):
        ax.bar(np.arange(len(keys)), values, color=colors, edgecolor="black")
        ax.set_xticks(np.arange(len(keys)), labels, rotation=20)
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.set_ylim(0, max(values)*1.23)
        for i, v in enumerate(values):
            ax.text(i, v+max(values)*.02, f"{v:.2f}", ha="center")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("同源360帧冷启动：只改变观测权重规则", fontsize=18)
    fig.text(.08, .015, "全部帧保留；没有身体时序项或显示平滑。残差衡量贴合观测，不是独立人体精度。", fontsize=11)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(experiment / "fit_comparison.png", dpi=150)
    plt.close(fig)
    error = np.linspace(0, 200, 501)
    terms = (np.full_like(error, .8), np.ones_like(error), np.exp(-error/8), np.ones_like(error), np.ones_like(error))
    quality = .8*np.exp(-error/8)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(error, quality, color=colors[0], label="原指数权重")
    for name, scale, color in [("参考25px", 25, colors[1]), ("参考40px", 40, colors[2])]:
        w = observation_weights(quality, terms, np.ones_like(error, bool), "reference", reference_reprojection_px=scale)
        ax.plot(error, w, color=color, label=name)
    ax.set(xlabel="双目平均回投影偏差（像素）", ylabel="有效拟合权重", ylim=(0, 1), xlim=(0, 200),
           title="权重曲线：普通偏差保留约束，严重冲突仍衰减")
    ax.legend()
    ax.grid(alpha=.2)
    fig.text(.12, .01, "示例固定二维置信度0.8、射线间距0；实际权重还考虑射线间距。", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(experiment / "weight_curves.png", dpi=150)
    plt.close(fig)
    fitted = {name: {side: project(d["predicted_coco"], cal, side) for side in obs}
              for name, d in data.items() if name in ("quality_replay", selected)}
    raw_px = {side: project(raw/1000, cal, side) for side in obs}
    for title, frames, joints, box in [("knee_before_after", [320, 321, 322], [13, 14, 15, 16], (330, 950, 1030, 1820)),
                                       ("arm_before_after", [136, 184, 231], [5, 6, 7, 8, 9, 10], (0, 140, 1080, 980))]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 10), facecolor="white")
        # One common pixel scale across all panels, retaining projections that
        # fall outside the image/crop on the white margin rather than hiding them.
        bounds = []
        x0, y0, x1, y1 = box
        for side in ("left", "right"):
            for f in frames:
                for points in (obs[side][f, :, :2], raw_px[side][f],
                               fitted["quality_replay"][side][f], fitted[selected][side][f]):
                    xy = points[joints]
                    upright = np.stack([xy[:, 1], 1919-xy[:, 0]], axis=-1) if side == "left" else np.stack([1079-xy[:, 1], xy[:, 0]], axis=-1)
                    bounds.extend(upright.tolist())
        bounds = np.asarray(bounds)
        left = min(0, float(bounds[:, 0].min()-x0-20))
        right = max(x1-x0, float(bounds[:, 0].max()-x0+20))
        top = min(0, float(bounds[:, 1].min()-y0-20))
        bottom = max(y1-y0, float(bounds[:, 1].max()-y0+20))
        for row, side in enumerate(("left", "right")):
            for col, f in enumerate(frames):
                ax = axes[row, col]
                folder = "left_ccw90" if side == "left" else "right_cw90"
                im = np.asarray(Image.open(images/folder/f"pair_{f:04d}.png").convert("RGB"))
                x0, y0, x1, y1 = box
                ax.imshow(im[y0:y1, x0:x1])
                for points, color, marker, label in [(obs[side][f, :, :2], "#22cc55", "o", "二维"),
                        (raw_px[side][f], "#00cddd", "s", "三角化"),
                        (fitted["quality_replay"][side][f], "#ff3333", "+", "原拟合"),
                        (fitted[selected][side][f], "#ffcc00", "x", "新拟合")]:
                    xy = points[joints]
                    upright = np.stack([xy[:, 1], 1919-xy[:, 0]], axis=-1) if side == "left" else np.stack([1079-xy[:, 1], xy[:, 0]], axis=-1)
                    for (u, v), j in zip(upright, joints, strict=True):
                        kw = {"facecolors": "none", "edgecolors": color} if marker in ("o", "s") else {"c": color}
                        ax.scatter(u-x0, v-y0, s=75, marker=marker, linewidths=2, **kw)
                        if marker == "o":
                            ax.text(u-x0+5, v-y0, str(j), color=color, fontsize=8)
                    ax.scatter([], [], c=color, marker=marker, label=label)
                ax.set_title(f"{'左' if side == 'left' else '右'}相机 · 第{f}帧")
                ax.set_xlim(left, right)
                ax.set_ylim(bottom, top)
                ax.set_xticks([])
                ax.set_yticks([])
        axes[0, 0].legend(loc="upper left", fontsize=9)
        fig.suptitle(f"原拟合（红＋）与 {selected} 新拟合（黄×）", fontsize=17)
        fig.text(.05, .01, "相机系原图回投影；严格拒绝点仍保留；无插值、无平滑；图像不认证内部关节中心。", fontsize=11)
        fig.tight_layout(rect=(0, .04, 1, .95))
        fig.savefig(experiment / (title+".png"), dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--selected", default="reference25")
    parser.add_argument("--review-output", type=Path, help="new review directory, keeping earlier evidence")
    args = parser.parse_args()
    review = args.review_output or args.experiment
    if args.review_output:
        review.mkdir(parents=True, exist_ok=False)
    output = review / "evaluation.json"
    if output.exists():
        raise FileExistsError(output)
    plan = read(args.experiment/"plan.json")
    command = read(Path(plan["source_commands"]["new"]))
    base_options = vars(build_parser().parse_args(modified_options(command, {}, "baseline-placeholder")))
    base_options = {k: str(v) if isinstance(v, Path) else v for k, v in base_options.items()}
    cal = load_stereo_fisheye(Path(base_options["calibration_dir"]))
    obs = {s: np.stack([rows[i] for i in range(360)]).astype(np.float32) for s in ("left", "right")
           for rows in [raw_side(Path(base_options[s+"_raw"]), s)]}
    tri = raw_triangulate(obs["left"], obs["right"], cal)
    raw, mask, quality = tri[0], tri[6], tri[8]
    error = (tri[1]+tri[2])/2
    ordinary = (error <= 25) & (tri[3] <= 80) & mask
    extreme = ((error >= 100) | (tri[3] >= 160)) & mask
    data = {}
    sets_path = ROOT / "research_records/engineering_validation/G20260930_smplh_contact_set_fix_v1_run3/contact_vertex_sets.json"
    sets = read(sets_path)["sets"]
    baseline_path = Path(plan["control_results"]["new"])
    with np.load(baseline_path) as a:
        original = {k: a[k].copy() for k in a.files if a[k].dtype.kind in "fibu"}
    report = {"units": "3D mm; step mm/frame; second differences mm/frame^2; no physical acceleration/accuracy",
              "selection": args.selected, "source": str(baseline_path), "surface_sets": str(sets_path), "cases": {}}
    for case in plan["cases"]:
        directory = args.experiment/"runs"/case["id"]
        if not (directory/"execution.json").exists():
            report["cases"][case["id"]] = {"status": "failed_or_incomplete"}
            continue
        options = read(directory/"resolved_options.json")
        changed = {k: [base_options[k], v] for k, v in options.items() if v != base_options[k]}
        expected = {"output_dir"} | {k for k, v in case["changes"].items() if v != base_options[k]}
        if set(changed) != expected:
            raise ValueError("unexpected controls changed")
        with np.load(baseline_path.parent/"cold_initialization.npz") as a, np.load(directory/"fit/cold_initialization.npz") as b:
            parity = {k: bool(np.array_equal(a[k], b[k], equal_nan=True)) for k in a.files if a[k].dtype.kind in "fibu"}
        if not all(parity.values()):
            raise ValueError("cold initialization differs")
        with np.load(directory/"fit/result.npz") as a:
            d = {k: a[k].copy() for k in ("predicted_coco", "vertices", "body_fit_weights", "betas", "vposer_latent")}
            for k, v in [("raw_triangulated_points", raw), ("body_quality", quality), ("body_accepted", mask), ("pair_id", np.arange(360))]:
                np.testing.assert_array_equal(a[k], v)
            np.testing.assert_array_equal(a["faces"], original["faces"])
            for k in ("vertices", "predicted_coco", "smplh_joints", "body_pose", "global_orient", "transl", "vposer_latent"):
                if not np.isfinite(a[k]).all():
                    raise ValueError("nonfinite candidate")
        data[case["id"]] = d
        model = d["predicted_coco"].astype(float)*1000
        residual = np.linalg.norm(model-raw, axis=-1)
        w = d["body_fit_weights"]
        weights_options = {"reference_reprojection_px": options["body_reference_reprojection_px"],
                           "reference_ray_gap_mm": options["body_reference_ray_gap_mm"],
                           "reference_confidence_floor": options["body_reference_confidence_floor"]}
        np.testing.assert_array_equal(w, observation_weights(quality, tri[9], mask, options["body_weighting"], **weights_options))
        row = {"status": "completed_not_promoted", "options_changed": changed, "initialization_parity": parity,
               "original_quality_weighted_rms_mm": float(np.sqrt(np.sum(quality[mask]*residual[mask]**2)/quality[mask].sum())),
               "uniform_supported_rms_mm": float(np.sqrt(np.mean(residual[mask]**2))),
               "own_weighted_rms_mm_not_cross_method_score": float(np.sqrt(np.sum(w[mask]*residual[mask]**2)/w[mask].sum())),
               "groups": {}, "joints": {}, "surface_movement_mm": {},
               "effective_weights": {"all_body": stats(w[:, 5:17][mask[:, 5:17]]), "ordinary": stats(w[ordinary]), "extreme": stats(w[extreme])},
               "right_knee_321_to_322_mm": float(np.linalg.norm(model[322, 14]-model[321, 14])),
               "knee_weights_320_322": w[320:323, 14].tolist(),
               "beta_norm": float(np.linalg.norm(d["betas"])), "latent_mean_square": float(np.mean(d["vposer_latent"]**2)),
               "execution": read(directory/"execution.json")}
        for label, js in [("arms", [7, 8, 9, 10]), ("legs", [13, 14, 15, 16])]:
            valid2 = mask[2:, js] & mask[1:-1, js] & mask[:-2, js]
            row["groups"][label] = {"residual_mm": stats(residual[:, js][mask[:, js]]),
                "second_difference": stats(np.linalg.norm(np.diff(model[:, js], n=2, axis=0), axis=-1)[valid2]),
                "worse_over_1mm": int(np.sum((residual[:, js]-np.linalg.norm(original["predicted_coco"][:, js]*1000-raw[:, js], axis=-1))[mask[:, js]] > 1))}
        for j in range(5, 17):
            row["joints"][NAMES[j]] = {"residual_mm": stats(residual[:, j][mask[:, j]]), "temporal": temporal(model[:, j], mask[:, j]),
                "weight": stats(w[:, j][mask[:, j]]),
                "projected_residual_px": {side: stats(np.linalg.norm(project(d["predicted_coco"], cal, side)[:, j]-obs[side][:, j, :2], axis=-1)[mask[:, j]]) for side in obs}}
        row["bend_deg"] = {name: stats(bend(model, js)) for name, js in [("right_knee", (12, 14, 16)), ("right_elbow", (6, 8, 10))]}
        for side in ("left", "right"):
            for part in ("sole", "palm"):
                ids = sorted({i for group in sets[f"{side}_{part}_surface_candidate"].values() for i in group})
                row["surface_movement_mm"][side+"_"+part] = stats(np.linalg.norm(d["vertices"][:, ids]-original["vertices"][:, ids], axis=-1)*1000)
        report["cases"][case["id"]] = row
    replay = data["quality_replay"]["predicted_coco"]
    report["quality_replay_vs_saved_max_mm"] = float(np.max(np.abs(replay-original["predicted_coco"]))*1000)
    base = report["cases"]["quality_replay"]
    for row in report["cases"].values():
        if row.get("status") != "completed_not_promoted":
            continue
        motion_ok = all(row["joints"][NAMES[j]]["temporal"]["second_difference"]["p95"] <= 1.1*base["joints"][NAMES[j]]["temporal"]["second_difference"]["p95"] for j in range(13, 17))
        row["preferred_candidate_checks"] = {"arm_mean_improves": row["groups"]["arms"]["residual_mm"]["mean"] < base["groups"]["arms"]["residual_mm"]["mean"],
            "right_knee_event_improves": row["right_knee_321_to_322_mm"] < base["right_knee_321_to_322_mm"],
            "all_knee_ankle_d2_p95_within_10percent": motion_ok}
    save(output, report)
    image_root = ROOT / "research_records/engineering_validation/G20261003_repositioned360_cold_smplh_v2/input"
    figure(review, report, data, raw, mask, obs, cal, image_root, args.selected)
    print(json.dumps({k: {"arm_mean": v["groups"]["arms"]["residual_mm"]["mean"], "leg_d2_p95": v["groups"]["legs"]["second_difference"]["p95"],
         "knee_event": v["right_knee_321_to_322_mm"], "checks": v["preferred_candidate_checks"]} for k, v in report["cases"].items() if "groups" in v}, indent=2))


if __name__ == "__main__":
    main()
