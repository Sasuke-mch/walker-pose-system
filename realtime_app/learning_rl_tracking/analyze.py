"""Failure-aware 448 comparison; numerical survival is not tracking accuracy."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .data import ROOT, load_real
from .run import save_json, stats


def camera_project(points, dataset):
    cal = dataset.calibration
    from pose_app.fisheye_camera import fisheye_project_numpy

    right = points @ cal.R_cam0_to_cam1.T + cal.T_cam0_to_cam1_mm / 1000
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = np.stack(
            [
                fisheye_project_numpy(points, cal.K0, cal.D0),
                fisheye_project_numpy(right, cal.K1, cal.D1),
            ],
            axis=1,
        )
    positive = (points[..., 2] > 0) & (right[..., 2] > 0)
    projected = np.where(positive[:, None, :, None], projected, np.nan)
    return projected


def score(name, projected, available, dataset):
    error = np.linalg.norm(projected - dataset.pixels, axis=-1)
    observed = dataset.valid
    finite = np.isfinite(error) & available[:, None, None]
    slots = observed.sum()
    correct25 = (error <= 25) & finite & observed
    correct50 = (error <= 50) & finite & observed
    per_frame = np.asarray(
        [
            np.median(error[i][observed[i] & finite[i]])
            if np.any(observed[i] & finite[i])
            else np.nan
            for i in range(448)
        ]
    )
    return dict(
        name=name,
        complete_frames=448,
        numerically_available_frames=int(available.sum()),
        observed_view_joint_slots=int(slots),
        available_observed_slots=int((finite & observed).sum()),
        full_sequence_pixel_fraction_within25=float(correct25.sum() / slots),
        full_sequence_pixel_fraction_within50=float(correct50.sum() / slots),
        projection_error_px_given_available=stats(error[finite & observed]),
        per_joint_pck25=[
            float(correct25[:, :, j].sum() / observed[:, :, j].sum()) for j in range(12)
        ],
        per_frame_median_error_px=per_frame,
        numerically_available_frame_mask=available,
        interpretation="same detector observations, failures included in full-sequence denominator; not independent accuracy",
    )


def analyze(root):
    dataset = load_real()
    results = []
    dynamics = []
    for target in sorted(root.glob("*/trajectory.npz")):
        with np.load(target, allow_pickle=False) as z:
            available = np.isin(z["status"], ["tracked", "initialized"])
            result = score(
                target.parent.name,
                camera_project(z["p_camera"], dataset),
                available,
                dataset,
            )
            qpos = z["qpos"]
            body_finite = np.isfinite(qpos).all(axis=1)
            root_up = np.full(448, np.nan)
            root_up[body_finite] = Rotation.from_quat(
                qpos[body_finite][:, [4, 5, 6, 3]]
            ).as_matrix()[:, 2, 1]
            result["root_local_up_dot_world_up"] = stats(root_up)
            result["root_axis_tilt_proxy_below_dot05_frames"] = int(
                np.sum(root_up < 0.5)
            )
            dyn = stats(np.linalg.norm(z["a_world"], axis=-1))
            finite_diff = stats(np.linalg.norm(z["finite_difference_a"], axis=-1))
            summary = json.loads(
                (target.parent / "summary.json").read_text(encoding="utf-8")
            )
            result["first_simulation_failure_frame"] = summary["first_failed_frame"]
            result["heel_numerical_max_m"] = summary["heel_error_m"]["max"]
            result["heel_simulated_frames"] = summary["hard_heel_evaluated_frames"]
            result["wrist_force_max_n"] = summary["support_force_n"]["max"]
            result["simulation_step_ms"] = summary["step_ms"]
            result["dynamics_acceleration_m_s2"] = dyn
            result["finite_difference_acceleration_m_s2"] = finite_diff
            results.append(result)
            dynamics.append(
                dict(
                    name=target.parent.name, dynamics=dyn, finite_difference=finite_diff
                )
            )
    rows = [
        json.loads(line)
        for line in (
            ROOT
            / "research_records/engineering_validation/G20261014_stereo_pva_baseline448_v2/tracking_final/trajectory.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    camera = np.asarray([r["position_mm"] for r in rows], float) / 1000
    projected = camera_project(camera, dataset)
    # Per-joint unavailable states remain failures; no low-error subset selection.
    results.append(score("pva_existing", projected, np.ones(448, bool), dataset))
    with np.load(
        ROOT
        / "research_records/engineering_validation/G20261013_reliability448_view_v1/fit/result.npz"
    ) as fitted:
        offline = fitted["predicted_coco"][:, 5:17].astype(float)
    reference = score(
        "smplh_offline_reference",
        camera_project(offline, dataset),
        np.ones(448, bool),
        dataset,
    )
    reference["interpretation"] = (
        "offline full-sequence fitted reference; future/shared-shape information, not a fair causal controller"
    )
    results.append(reference)
    synthetic = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(root.glob("synthetic_*.json"))
    ]
    report = dict(
        real=results,
        synthetic=synthetic,
        dynamics_sampling_comparison=dynamics,
        observation_denominator=int(dataset.valid.sum()),
        decision="Stop this PPO candidate: all tested PPO real448 pixel hit rates within25px are below1%; synthetic policies show no material improvement over PD hold. This rejects the current training/model configuration, not reinforcement learning in general.",
        scope="PPO trained only on synthetic clips. Stage scene/support/rigid heel are conditional model assumptions. Large numerical state coverage is not tracking success.",
    )
    annotation_path = root / "first_frame_annotation_check.json"
    if annotation_path.exists():
        report["independent_first_frame_reference"] = json.loads(
            annotation_path.read_text(encoding="utf-8")
        )
    report["heel_revision"] = (
        "Final real hard-heel outputs use actual SMPL-H surface3387/6787. Policies and held-out synthetic evaluation were trained/evaluated in historical rigid mode; constraint-only correction kept weights fixed. Old incorrect rigid mapping outputs retained."
    )
    save_json(root / "analysis.json", report)
    for result in results:
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "name",
                        "numerically_available_frames",
                        "full_sequence_pixel_fraction_within25",
                        "projection_error_px_given_available",
                    )
                }
            ),
            flush=True,
        )
    plot(root, results)
    return report


def plot(root, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = load_real().time
    selected = [
        r
        for r in results
        if r["name"]
        in (
            "classic_free",
            "classic_heel",
            "ppo0_free",
            "ppo0_heel",
            "pva_existing",
            "smplh_offline_reference",
        )
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, layout="constrained")
    for i, row in enumerate(selected):
        style = ("-", "--", ":", "-.")[i % 4]
        axes[0].plot(
            t,
            row["per_frame_median_error_px"],
            color=str(0.12 * (i % 6)),
            linestyle=style,
            label=row["name"],
            lw=1,
        )
        axes[1].step(
            t,
            np.asarray(row["numerically_available_frame_mask"], float) + i * 0.08,
            color=str(0.12 * (i % 6)),
            linestyle=style,
            where="post",
            label=row["name"],
        )
    axes[0].axhline(25, color="black", lw=0.6)
    axes[0].set_ylabel("Median detector reprojection error (px)")
    axes[1].set_ylabel("Numerically available (vertically offset)")
    axes[1].set_xlabel("Host-return midpoint time (s)")
    for ax in axes:
        ax.legend(fontsize=8, ncols=2)
        ax.grid(color=".85")
    fig.suptitle(
        "All448 conditional-scene experiment: no sequence-level manual or 3D accuracy truth; failed tails retained"
    )
    fig.savefig(root / "failure_aware_comparison.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args().output)
