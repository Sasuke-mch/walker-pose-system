"""Export a focused scientific figure of the frozen frame321->322 knee event.

This is a historical audit, not a new scene replay or a body/ground estimate.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit_limb_chain import ROOT
from compare_capture_arm_bias import project
from matplotlib.font_manager import FontProperties
from PIL import Image
from pose_app.body_observations import raw_side
from pose_app.fisheye_camera import load_stereo_fisheye


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    protocol = json.loads((args.audit / "protocol.json").read_text(encoding="utf-8"))
    with np.load(args.audit / "audit_arrays.npz") as a:
        raw = a["raw_mm"][321:323].copy()
        model = a["model_coco_mm"][321:323].copy()
        quality = a["quality"][321:323, 14].copy()
        strict = a["mask_strict"][321:323, 14].copy()
        body = a["mask_body"][321:323, 14].copy()
        pose_term = a["model_step_pose_term_mm"][321, 14].copy()
    with np.load(protocol["fit"]) as a:
        np.testing.assert_array_equal(a["pair_id"][321:323], [321, 322])
        np.testing.assert_array_equal(raw, a["raw_triangulated_points"][321:323])
        np.testing.assert_array_equal(model, a["predicted_coco"][321:323].astype(float)*1000)
    with np.load(Path(protocol["scene"]) / "scene_transforms.npz", allow_pickle=True) as a:
        r = a["rotation_ground_from_left"][321:323]
        t = a["translation_ground_from_left_mm"][321:323]
        stage = a["stage"][321:323].tolist()
    np.testing.assert_array_equal(r[0], r[1])
    np.testing.assert_array_equal(t[0], t[1])
    np.testing.assert_array_equal(pose_term, np.zeros(3))
    delta_raw, delta_model = np.diff(raw[:, 14], axis=0)[0], np.diff(model[:, 14], axis=0)[0]
    distance_raw, distance_model = np.linalg.norm(delta_raw), np.linalg.norm(delta_model)
    cal = load_stereo_fisheye(Path(protocol["calibration"]))
    pixels = {}
    for camera in ("left", "right"):
        observations = raw_side(ROOT / protocol["detector_inputs"][camera], camera)
        obs = np.stack([observations[f][14, :2] for f in (321, 322)])
        tri = project(raw/1000, cal, camera)[:, 14]
        fitted = project(model/1000, cal, camera)[:, 14]
        points = np.stack([obs, tri, fitted], axis=1)
        if camera == "left":
            upright = np.stack([points[..., 1], 1919-points[..., 0]], axis=-1)
        else:
            upright = np.stack([1079-points[..., 1], points[..., 0]], axis=-1)
        pixels[camera] = upright
    metrics = {
        "frames": [321, 322], "joint": "COCO right knee index14", "coordinates": "left camera XYZ, mm",
        "raw_step_vector_mm": delta_raw.tolist(), "model_step_vector_mm": delta_model.tolist(),
        "raw_step_mm": float(distance_raw), "model_step_mm": float(distance_model),
        "step_ratio_not_accuracy_ratio": float(distance_model/distance_raw),
        "quality": quality.tolist(), "strict_accepted": strict.tolist(), "body_candidate_accepted": body.tolist(),
        "stage": stage, "ground_rotation_equal": True, "ground_translation_equal": True,
        "pose_step_vector_mm": pose_term.tolist(),
        "inputs": protocol, "code": "realtime_app/visualize_knee_jump.py",
        "scope": "Frozen historical audit only; no new inference, fitting, triangulation or stage replay",
        "success": "Exact source/ID alignment and unchanged ground R/t, zero pose term, visual legibility",
        "limits": "Real movement included; no anatomical truth; arrows compare displacement vectors from their own starting positions",
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 12, "axes.unicode_minus": False})
    green, red, cyan = "#16814a", "#d62d34", "#0398c5"
    fig = plt.figure(figsize=(16, 12), facecolor="white")
    grid = fig.add_gridspec(2, 4, height_ratios=[.95, 1.05], top=.85, bottom=.14, hspace=.42, wspace=.32)
    ax = fig.add_subplot(grid[0, :2])
    bars = ax.barh([1, 0], [distance_raw, distance_model], color=[green, red], edgecolor="black", height=.42)
    ax.set_yticks([1, 0], ["原始三角化", "SMPL-H回归点"])
    ax.set_xlim(0, 200)
    ax.set_ylim(-.7, 1.7)
    ax.set_xlabel("相机系右膝的完整三维位移（毫米）")
    ax.set_title("① 同一帧间隔，模型位置变化更大", pad=16, loc="left", fontweight="bold")
    for bar, value in zip(bars, [distance_raw, distance_model], strict=True):
        ax.text(value+3, bar.get_y()+bar.get_height()/2, f"{value:.2f} mm", va="center", fontweight="bold", fontsize=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax = fig.add_subplot(grid[0, 2:], projection="3d")
    for vector, color, label in [(delta_raw, green, "原始三角化"), (delta_model, red, "SMPL-H")]:
        ax.quiver(0, 0, 0, *vector, color=color, linewidth=3, arrow_length_ratio=.12, label=label)
        ax.scatter(*vector, color=color, s=35)
    ax.scatter(0, 0, 0, color="black", s=30)
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-145, 65)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("ΔX (mm)")
    ax.set_ylabel("ΔY (mm)")
    ax.set_zlabel("ΔZ (mm)")
    ax.view_init(elev=24, azim=-62)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_title("② 位移方向也不同：各自前帧位置设为零", pad=12, fontsize=13)
    for col, (camera, row, fid) in enumerate([("left", 0, 321), ("left", 1, 322), ("right", 0, 321), ("right", 1, 322)]):
        ax = fig.add_subplot(grid[1, col])
        folder = "left_ccw90" if camera == "left" else "right_cw90"
        source = ROOT / protocol["detector_inputs"]["images"] / folder / f"pair_{fid:04d}.png"
        im = np.asarray(Image.open(source).convert("RGB"))
        points = pixels[camera]
        center = points.reshape(-1, 2).mean(0)
        width, height = 330, 440
        x0 = int(np.clip(center[0]-width/2, 0, im.shape[1]-width))
        y0 = int(np.clip(center[1]-height/2, 0, im.shape[0]-height))
        ax.imshow(im[y0:y0+height, x0:x0+width])
        for k, (color, marker) in enumerate([(green, "o"), (cyan, "s"), (red, "+")]):
            u, v = points[row, k]-[x0, y0]
            if marker == "+":
                ax.scatter(u, v, s=250, c=color, marker=marker, linewidths=3)
            else:
                ax.scatter(u, v, s=160, facecolors="none", edgecolors=color, marker=marker, linewidths=2.5)
        ax.set_title(f"{'左相机' if camera == 'left' else '右相机'} · 第{fid}帧", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(.05, .965, "右膝：原始点移动27毫米，拟合点移动166毫米", fontsize=23, fontweight="bold")
    fig.text(.05, .925, "第321 → 322帧  |  Stage1 → Stage1  |  地面旋转完全相同、平移完全相同", fontsize=14)
    fig.text(.05, .885, "地面位姿引入的额外位移 = 0 mm；该次放大已出现在相机系身体结果中。", fontsize=14)
    fig.text(.05, .105, "③ 原图同尺度裁剪：绿色○ 二维检测；青色□ 原三角化回投影；红色＋ SMPL-H回投影", fontsize=13)
    fig.text(.05, .071, f"严格几何门：两帧均拒绝该点；身体候选：两帧均保留。质量权重 {quality[0]:.6f} → {quality[1]:.7f}。", fontsize=12)
    fig.text(.05, .037, "注意：27/166毫米是三维轨迹变化，不是图上像素距离或真实误差；未平滑、未插值、未改变拟合或Stage。", fontsize=11)
    fig.savefig(args.output / "right_knee_jump.png", dpi=150)
    fig.savefig(args.output / "right_knee_jump.pdf")
    plt.close(fig)
    print(json.dumps({k: metrics[k] for k in ("raw_step_mm", "model_step_mm", "pose_step_vector_mm", "quality")}, indent=2))


if __name__ == "__main__":
    main()
