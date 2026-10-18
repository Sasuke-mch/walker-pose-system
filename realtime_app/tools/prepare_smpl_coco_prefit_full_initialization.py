#!/usr/bin/env python3
"""Prepare full-sequence male-SMPL initialization from current fitted donors.

For every target frame, choose the donor whose pelvis-centred, scale-normalized
temporal-prefit COCO skeleton is closest.  Reuse only that donor's male-SMPL
root/body pose, then recompute translation by aligning the regressed COCO hip
midpoint to the target temporal-prefit hip midpoint in the left-camera frame.
Raw stereo 2-D observations and validity masks are copied from the target frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pose_app.fisheye_camera import load_stereo_fisheye
from pose_app.smpl_coco_observation import load_coco17_regressor, load_smpl_male, regress_coco17_torch
from tools.fit_smpl_coco_single_frame import valid_2d
from tools.fit_smpl_coco_temporal_guardrail import read_jsonl_by_pair
from tools.fit_smpl_coco_prefit_trajectory import load_ground_transforms, load_prefit


MATCH_JOINTS = np.asarray([5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16], dtype=np.int64)


def normalized_skeleton(points: np.ndarray) -> np.ndarray:
    pelvis = 0.5 * (points[:, 11] + points[:, 12])
    centered = points[:, MATCH_JOINTS] - pelvis[:, None]
    scale = np.sqrt(np.mean(np.square(centered), axis=(1, 2), keepdims=True))
    return centered / np.maximum(scale, 1e-6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor-window", type=Path, required=True)
    parser.add_argument("--stereo-jsonl", type=Path, required=True)
    parser.add_argument("--prefit-jsonl", type=Path, required=True)
    parser.add_argument("--ground-motion", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--regressor", type=Path, required=True)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--end-pair", type=int, default=447)
    parser.add_argument("--minimum-score", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    (output / "frames").mkdir(parents=True)

    pair_ids = list(range(args.start_pair, args.end_pair + 1))
    donor_dirs = sorted((args.donor_window.resolve() / "frames").glob("frame_*"))
    donor_ids = [int(path.name.split("_")[-1]) for path in donor_dirs]
    if not donor_ids:
        raise FileNotFoundError(f"no donor frames under {args.donor_window}")

    target_prefit = load_prefit(args.prefit_jsonl.resolve(), pair_ids)
    donor_prefit = load_prefit(args.prefit_jsonl.resolve(), donor_ids)
    target_norm = normalized_skeleton(target_prefit)
    donor_norm = normalized_skeleton(donor_prefit)
    distances = np.mean(
        np.linalg.norm(target_norm[:, None] - donor_norm[None], axis=3),
        axis=2,
    )
    donor_choice = np.argmin(distances, axis=1)

    donor_roots, donor_poses = [], []
    faces = None
    for directory in donor_dirs:
        with np.load(directory / "result.npz") as result:
            donor_roots.append(np.asarray(result["root_orient"][0], dtype=np.float32))
            donor_poses.append(np.asarray(result["body_pose"][0], dtype=np.float32))
            current_faces = np.asarray(result["faces"], dtype=np.uint32)
        if faces is None:
            faces = current_faces
        elif not np.array_equal(faces, current_faces):
            raise ValueError("donor SMPL topology changed")
    assert faces is not None
    roots = np.asarray(donor_roots, dtype=np.float32)[donor_choice]
    poses = np.asarray(donor_poses, dtype=np.float32)[donor_choice]

    rotations_world, translations_world = load_ground_transforms(
        args.ground_motion.resolve(), pair_ids
    )
    target_left = np.einsum(
        "nji,nkj->nki",
        rotations_world,
        target_prefit - translations_world[:, None],
    )

    import torch

    model = load_smpl_male(args.model_path, batch_size=len(pair_ids), device=args.device)
    regressor = load_coco17_regressor(args.regressor)
    with torch.no_grad():
        result = model(
            betas=torch.zeros(len(pair_ids), 10, device=args.device),
            body_pose=torch.tensor(poses, device=args.device),
            global_orient=torch.tensor(roots, device=args.device),
            transl=torch.zeros(len(pair_ids), 3, device=args.device),
            return_verts=True,
        )
        zero_coco = regress_coco17_torch(result.vertices, regressor).cpu().numpy()
    target_pelvis = 0.5 * (target_left[:, 11] + target_left[:, 12])
    zero_pelvis = 0.5 * (zero_coco[:, 11] + zero_coco[:, 12])
    translations = (target_pelvis - zero_pelvis).astype(np.float32)

    rows = read_jsonl_by_pair(args.stereo_jsonl.resolve(), pair_ids[0], pair_ids[-1])
    cal = load_stereo_fisheye(args.calibration_dir)
    report = []
    for index, pair in enumerate(pair_ids):
        row = rows[pair]
        left = np.asarray(row["left"]["persons"][0]["keypoints"], dtype=np.float32)
        right = np.asarray(row["right"]["persons"][0]["keypoints"], dtype=np.float32)
        valid_left = valid_2d(left, cal.image_width, cal.image_height, args.minimum_score)
        valid_right = valid_2d(right, cal.image_width, cal.image_height, args.minimum_score)
        frame_dir = output / "frames" / f"frame_{pair:04d}"
        frame_dir.mkdir()
        np.savez_compressed(
            frame_dir / "result.npz",
            faces=faces,
            observed_left=left,
            observed_right=right,
            valid_left=valid_left,
            valid_right=valid_right,
            root_orient=roots[index:index + 1],
            body_pose=poses[index:index + 1],
            translation_m=translations[index:index + 1],
            betas=np.zeros((1, 10), dtype=np.float32),
        )
        report.append({
            "pair_id": pair,
            "donor_pair_id": donor_ids[int(donor_choice[index])],
            "normalized_prefit_distance": float(distances[index, donor_choice[index]]),
            "translation_m": translations[index].tolist(),
        })

    summary = {
        "schema_version": "smpl_coco_prefit_full_initialization_v1",
        "frame_count": len(pair_ids),
        "range": [pair_ids[0], pair_ids[-1]],
        "donor_range": [min(donor_ids), max(donor_ids)],
        "method": "nearest normalized temporal-prefit skeleton donor pose; target prefit pelvis translation alignment",
        "uses_old_smplx_or_neutral_parameters": False,
        "records": report,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "frames": len(pair_ids),
        "donors": len(donor_ids),
        "maximum_normalized_prefit_distance": float(np.max(np.min(distances, axis=1))),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
