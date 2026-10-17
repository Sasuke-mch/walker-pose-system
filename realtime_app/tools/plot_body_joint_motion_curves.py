"""Plot existing body estimates using aligned capture host-return timestamps.

This diagnoses the saved baseline; it does not run tracking or learned control.
Host read-return time is not sensor exposure time. No interpolation or smoothing
is applied, and derivatives do not cross unavailable ground-pose intervals.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

APP = Path(__file__).resolve().parents[1]
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))
from pose_app.lower_limb_kinematics import three_point_derivatives  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research_records/engineering_validation/G20261013_reliability448_view_v1"
INPUT = ROOT / "research_records/engineering_validation/V20260908_people0_1_2_pmpose_c3_chain/people_1/input_448pairs"
JOINTS = tuple(range(5, 17))
NAMES = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
         "right_knee", "left_ankle", "right_ankle")
GROUPS = (("左臂", (5, 7, 9), ("肩", "肘", "腕")),
          ("右臂", (6, 8, 10), ("肩", "肘", "腕")),
          ("左腿", (11, 13, 15), ("髋", "膝", "踝")),
          ("右腿", (12, 14, 16), ("髋", "膝", "踝")))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_sources(args: argparse.Namespace) -> dict:
    manifest = read_csv(args.manifest)
    if args.capture_pairs is None:
        metadata = json.loads((args.manifest.parent / "metadata.json").read_text(encoding="utf-8"))
        args.capture_pairs = Path(metadata["source_session"]) / "stereo_pairs.csv"
    capture_metadata = json.loads((args.capture_pairs.parent / "metadata.json").read_text(encoding="utf-8"))
    if capture_metadata["timestamp_type"] != "host_read_return_perf_counter_ns":
        raise ValueError("unexpected capture time definition")
    captured = read_csv(args.capture_pairs)
    with np.load(args.fit, allow_pickle=False) as fitted:
        pair_ids = fitted["pair_id"].copy()
        camera_points_m = fitted["predicted_coco"].astype(np.float64)
    with np.load(args.scene, allow_pickle=True) as scene:
        rotation = scene["rotation_ground_from_left"].copy()
        translation_mm = scene["translation_ground_from_left_mm"].copy()
        stages = scene["stage"].astype(str)
        observed = scene["accepted"].copy()
    ground_rows = [json.loads(line) for line in args.ground.read_text(encoding="utf-8").splitlines()]
    if not (len(manifest) == len(pair_ids) == len(rotation) == len(stages) == len(ground_rows)):
        raise ValueError("source frame counts differ")
    if camera_points_m.shape != (len(pair_ids), 17, 3) or observed.shape != (len(pair_ids), 17):
        raise ValueError("expected COCO-17 points and strict observation mask")
    left_ns, right_ns, capture_ids = [], [], []
    for pair, row, ground_row in zip(pair_ids, manifest, ground_rows):
        if int(row["sequence_index"]) != int(pair) or int(ground_row["pair_id"]) != int(pair):
            raise ValueError("frame identities do not match")
        original = captured[int(row["source_pair_index"])]
        if any(row[f"{side}_frame_id"] != original[f"{side}_frame_id"] for side in ("left", "right")):
            raise ValueError("capture frame mapping mismatch")
        left_ns.append(int(original["left_host_return_timestamp_ns"]))
        right_ns.append(int(original["right_host_return_timestamp_ns"]))
        capture_ids.append(int(original["pair_id"]))
    left_ns, right_ns = np.asarray(left_ns, dtype=np.int64), np.asarray(right_ns, dtype=np.int64)
    origin_twice_ns = int(left_ns[0] + right_ns[0])
    time_s = (left_ns + right_ns - origin_twice_ns) / 2e9
    if not np.all(np.diff(time_s) > 0):
        raise ValueError("capture pair times must increase strictly")
    points_m = np.einsum("tij,tkj->tki", rotation, camera_points_m) + translation_mm[:, None] / 1000
    statuses = np.asarray([row["pose_audit"]["status"] for row in ground_rows])
    ground_valid = (~np.isin(statuses, ("unavailable", "rejected")) & (stages != "warming_up") &
                    np.isfinite(rotation).all(axis=(1, 2)) & np.isfinite(translation_mm).all(axis=1))
    model_valid = np.isfinite(points_m).all(axis=2) & ground_valid[:, None]
    relative = points_m - points_m[:, (11, 12)].mean(axis=1)[:, None]
    relative_valid = model_valid & model_valid[:, (11, 12)].all(axis=1)[:, None]
    return dict(pair_ids=pair_ids, capture_ids=capture_ids, time_s=time_s,
                left_time_s=(2 * left_ns - origin_twice_ns) / 2e9,
                right_time_s=(2 * right_ns - origin_twice_ns) / 2e9,
                left_ns=left_ns, right_ns=right_ns, stages=stages, statuses=statuses,
                observed=observed, frames=(("world", points_m, model_valid),
                                         ("pelvis_relative", relative, relative_valid)))


def plot_curves(data: dict, frame: str, velocity: np.ndarray,
                acceleration: np.ndarray, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 12})
    fig, axes = plt.subplots(4, 2, figsize=(17, 12), sharex=True, layout="constrained")
    style = ("-", "--", ":")
    for row, (limb, joints, labels) in enumerate(GROUPS):
        for col, (value, ylabel) in enumerate(((velocity, "速度模长 (m/s)"),
                                             (acceleration, "加速度模长 (m/s²)"))):
            ax = axes[row, col]
            for joint, label, line_style in zip(joints, labels, style):
                ax.plot(data["time_s"], np.linalg.norm(value[:, joint], axis=-1),
                        color="black", linestyle=line_style, linewidth=1.1, label=label)
            ax.set_title(limb)
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
            ax.grid(color="0.85", linewidth=0.6)
            ax.legend(loc="upper right", ncols=3, framealpha=0.9)
    for ax in axes[-1]:
        ax.set_xlabel("相对主机读取返回时间 (s)")
    caption = "固定地面系" if frame == "world" else "固定地面系减去双髋中心（保留世界轴方向）"
    fig.suptitle(f"现有 448 帧 SMPL-H 的 COCO 身体回归点：{caption}\n"
                 "非跟踪 / 强化学习新结果；主机返回时间非曝光时间；三点非均匀差分，无平滑/插值，失效段与端点断开",
                 fontsize=15)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    data = load_sources(args)
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ("pair_id", "capture_pair_id", "joint_coco_id", "joint_name", "coordinate_frame",
              "pair_host_time_s", "left_host_time_s", "right_host_time_s",
              "left_host_return_timestamp_ns", "right_host_return_timestamp_ns", "timestamp_kind",
              "stage", "ground_pose_status", "model_valid", "strict_observed_supported",
              "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s", "speed_m_s",
              "ax_m_s2", "ay_m_s2", "az_m_s2", "acceleration_m_s2")
    with (args.output / "joint_motion.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for frame, points, valid in data["frames"]:
            velocity, acceleration = three_point_derivatives(data["time_s"], points, valid)
            plot_curves(data, frame, velocity, acceleration,
                        args.output / f"{frame}_motion.png")
            for index, pair_id in enumerate(data["pair_ids"]):
                for joint, name in zip(JOINTS, NAMES):
                    p, v, a = points[index, joint], velocity[index, joint], acceleration[index, joint]
                    writer.writerow(dict(pair_id=int(pair_id), capture_pair_id=data["capture_ids"][index],
                        joint_coco_id=joint, joint_name=name, coordinate_frame=frame,
                        pair_host_time_s=data["time_s"][index], left_host_time_s=data["left_time_s"][index],
                        right_host_time_s=data["right_time_s"][index],
                        left_host_return_timestamp_ns=data["left_ns"][index],
                        right_host_return_timestamp_ns=data["right_ns"][index],
                        timestamp_kind="host_read_return_perf_counter_ns_not_exposure",
                        stage=data["stages"][index], ground_pose_status=data["statuses"][index],
                        model_valid=bool(valid[index, joint]),
                        strict_observed_supported=bool(valid[index, joint] and data["observed"][index, joint]),
                        x_m=p[0], y_m=p[1], z_m=p[2], vx_m_s=v[0], vy_m_s=v[1], vz_m_s=v[2],
                        speed_m_s=np.linalg.norm(v), ax_m_s2=a[0], ay_m_s2=a[1], az_m_s2=a[2],
                        acceleration_m_s2=np.linalg.norm(a)))
    for name in ("fit", "scene", "ground", "manifest", "capture_pairs", "output"):
        print(f"{name}: {getattr(args, name).resolve()}")
    print(f"mapped={len(data['pair_ids'])}, span={data['time_s'][-1]:.9f}s; "
          "host-return timestamps, not exposure timestamps; saved baseline only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, default=BASE / "fit/result.npz")
    parser.add_argument("--scene", type=Path, default=BASE / "view/scene/scene_transforms.npz")
    parser.add_argument("--ground", type=Path, default=BASE / "view/scene/dynamic_ground_pose.jsonl")
    parser.add_argument("--manifest", type=Path, default=INPUT / "selection_manifest.csv")
    parser.add_argument("--capture-pairs", type=Path)
    parser.add_argument("--output", type=Path, default=BASE / "motion_curves")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
