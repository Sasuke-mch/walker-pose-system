"""Plot saved contact-route joint derivatives without rerunning the tracker.

These are raw model-state derivatives in the current left-camera coordinates,
not finite differences of the frame-to-frame positions or ground-frame motion.
The numerical reporting formula is the existing tracker's +/-10 ms formula.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from pose_app.smplh_motion_tracker import SMPLHObservationMap

DEFAULT_RUN = APP.parent / (
    "research_records/engineering_validation/G20261014_smplh_motion448_v1/"
    "hip_pose_repair_20261016/foot_direction_20261016/"
    "autograd_contact_shoe25_full448_20261017"
)


def reconstruct(run: Path, output: Path, device: str) -> None:
    """Reuse saved q/v/a, frozen assets and the frame-specific hip regressor."""
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    parameters = json.loads(
        (Path(protocol["source_body_run"]) / "run_parameters.json").read_text(encoding="utf-8")
    )
    with np.load(run / "result.npz", allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files if key != "covariance"}
    times = state["timestamp_s"]
    if len(times) != 448 or not np.all(np.diff(times) > 0):
        raise ValueError("Expected the saved full448 run with increasing timestamps")
    model = SMPLHObservationMap(
        Path(parameters["effective_initial_fit"]),
        *[Path(parameters["inputs"][key]) for key in
          ("smplh_model", "regressor", "vposer_dir", "mano_left", "mano_right")],
        device=device,
    )
    model.torch.set_num_threads(2)
    if not np.array_equal(model.betas.cpu().numpy(), state["betas"]):
        raise ValueError("Saved shape differs from the source initialization")
    eps = float(protocol["configuration"]["derivative_step_s"])
    velocity = np.full((len(times), 12, 3), np.nan)
    acceleration = np.full_like(velocity, np.nan)
    center_error = 0.0
    for frame in range(len(times)):
        # run_smplh_contact_route switches the observation mapping at frame11.
        if frame == 11:
            model.regressor = model.torch.as_tensor(
                state["regressor"], dtype=model.torch.float32, device=device
            )
        if not state["pose_available"][frame]:
            continue
        translation = state["transl"][frame]
        root = state["root_rotation"][frame]
        latent = state["vposer_latent"][frame]
        center = model.evaluate_batch(translation[None], root[None], latent[None])["coco_m"][0]
        error = float(np.max(np.abs(center - state["predicted_coco"][frame])))
        center_error = max(center_error, error)
        if not np.isfinite(error) or error > 2e-6:
            raise ValueError(f"Frame {frame}: saved/model observation mismatch {error} m")
        v, a = state["velocity"][frame], state["acceleration"][frame]
        offsets = np.stack((eps*v + .5*eps**2*a, -eps*v + .5*eps**2*a))
        nearby = model.evaluate_batch(
            translation + offsets[:, :3],
            root @ Rotation.from_rotvec(offsets[:, 3:6]).as_matrix(),
            latent + offsets[:, 6:],
        )["coco_m"][:, 5:17]
        points = state["predicted_coco"][frame, 5:17]
        velocity[frame] = (nearby[0] - nearby[1]) / (2*eps)
        acceleration[frame] = (nearby[0] + nearby[1] - 2*points) / eps**2
        if frame % 100 == 0 or frame == len(times)-1:
            print(f"Reconstructed saved-state derivatives {frame + 1}/{len(times)}", flush=True)
    rows = [json.loads(line) for line in
            (run / "trajectory.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if [row["frame"] for row in rows] != list(range(len(times))):
        raise ValueError("Optimizer states are not aligned to the saved frames")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "joint_derivatives.npz", timestamp_s=times,
        velocity_m_s=velocity, acceleration_m_s2=acceleration,
        derivative_valid=state["derivative_valid"], segment_id=state["segment_id"],
        pose_available=state["pose_available"], coco17_indices=np.arange(5, 17),
        optimizer_converged=np.array([row["optimizer"]["status"] == "converged" for row in rows]),
        source_result=str((run / "result.npz").resolve()),
        coordinate_frame="current_left_camera", derivative_step_s=eps,
        source_observation_max_difference_m=center_error,
        method="saved q/v/a; original +/-step model reporting formula; no temporal smoothing",
    )
    print(f"All448 saved/model point agreement: max {center_error:.3g} m", flush=True)


def draw(run: Path, output: Path) -> None:
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=font).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 10, "axes.unicode_minus": False,
        "axes.edgecolor": "#BBBBBB", "axes.linewidth": .65,
        "text.color": "#292929", "xtick.color": "#777777", "ytick.color": "#777777",
        "xtick.labelsize": 9, "ytick.labelsize": 9, "svg.fonttype": "none",
        "path.simplify": False, "savefig.facecolor": "white",
    })
    with np.load(output / "joint_derivatives.npz", allow_pickle=False) as data:
        if str(data["source_result"]) != str((run / "result.npz").resolve()):
            raise ValueError("Derived data belongs to another run")
        time = data["timestamp_s"] - data["timestamp_s"][0]
        values = [np.linalg.norm(data[key], axis=-1) for key in
                  ("velocity_m_s", "acceleration_m_s2")]
        valid = data["derivative_valid"]
        segments, available = data["segment_id"], data["pose_available"]
        converged = int(data["optimizer_converged"].sum())
    colors = ("#202020", "#979797")
    styles = ("-", (0, (4, 2.4)))
    fig, axes = plt.subplots(3, 2, figsize=(11.8, 8.8), sharex=True)
    fig.subplots_adjust(left=.085, right=.97, bottom=.175, top=.77, hspace=.47, wspace=.22)
    fig.text(.085, .957, "下肢运动曲线", fontsize=22, va="top")
    fig.text(.085, .899, "SMPL-H  ·  自动微分 + 接触  ·  鞋高参考 25 mm", fontsize=10, color="#737373")
    fig.text(.085, .865, "448 帧  /  16.04 s    ·    模型原始估计，含未收敛帧", fontsize=9, color="#737373")
    fig.legend(handles=[
        Line2D([], [], color=colors[side], lw=1.5, ls=styles[side], label=label)
        for side, label in enumerate(("左侧", "右侧"))
    ], loc="upper right", bbox_to_anchor=(.98, .957), frameon=False, ncol=2, fontsize=10,
        handlelength=3, columnspacing=1.6)
    fig.text(.085, .817, "速度幅值  |v|  (m/s)", fontsize=12)
    fig.text(.565, .817, "加速度幅值  |a|  (m/s²)", fontsize=12)
    for row, label in enumerate(("髋观察点", "膝观察点", "踝观察点")):
        for col, magnitudes in enumerate(values):
            ax = axes[row, col]
            for side in range(2):
                joint = 6 + row*2 + side
                y = magnitudes[:, joint].copy()
                y[~available] = np.nan
                # Never bridge unavailable states or a tracker restart with a line.
                breaks = np.flatnonzero(np.diff(segments) != 0) + 1
                plot_time, plot_y = time.copy(), y.copy()
                for frame in breaks[::-1]:
                    plot_time = np.insert(plot_time, frame, np.nan)
                    plot_y = np.insert(plot_y, frame, np.nan)
                ax.plot(plot_time, plot_y, color=colors[side], ls=styles[side], lw=1.25,
                        solid_capstyle="round", dash_capstyle="round")
                accepted = valid[:, joint] & available & np.isfinite(y)
                ax.scatter(time[accepted], y[accepted], s=8, color=colors[side], linewidths=0, zorder=3)
            ax.set_title(label, loc="left", fontsize=10.5, pad=9)
            ax.set_xlim(time[0], time[-1])
            ax.set_ylim(bottom=0)
            ax.margins(y=.10)
            ax.yaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.grid(axis="y", color="#ECECEC", lw=.6)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", length=3, width=.6)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 2:
                ax.set_xlabel("时间 / s", fontsize=10, labelpad=9)
    counts = valid[:, 6:12].sum(axis=0).reshape(3, 2)
    fig.text(.085, .105, "连续线：全部有限帧估计，逐帧直连，无平滑；小圆点：通过原导数有效性门。",
             fontsize=8.5, color="#737373")
    fig.text(.085, .078, "有效帧（左 / 右）：" + "  ·  ".join(
        f"{name} {pair[0]}/{pair[1]}" for name, pair in zip(("髋", "膝", "踝"), counts)
    ) + f"；每侧分母 448。优化收敛 {converged}/447（首帧仅初始化）。", fontsize=8, color="#737373")
    fig.text(.085, .051,
             "量为当前左相机坐标下的模型状态导数；非地面系真实运动。髋为映射观察点，11 帧起采用虚拟髋映射。",
             fontsize=8, color="#737373")
    for suffix in ("png", "svg"):
        path = output / f"lower_limb_velocity_acceleration.{suffix}"
        fig.savefig(path, dpi=240)
        print(f"Saved {path}", flush=True)
    plt.close(fig)
    for name, value in zip(("speed_m_s", "acceleration_m_s2"), values):
        print(name, "lower-limb maximum", float(np.nanmax(value[:, 6:12])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--render-only", action="store_true", help="Restyle the existing derived data")
    args = parser.parse_args()
    run = args.run.resolve()
    output = run / "motion_curves"
    if not args.render_only:
        reconstruct(run, output, args.device)
    draw(run, output)


if __name__ == "__main__":
    main()
