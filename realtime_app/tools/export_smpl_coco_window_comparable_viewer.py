#!/usr/bin/env python3
"""Export fitted male SMPL frames using the established fixed-ground viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def world_from_left(points_m: np.ndarray, frame: dict) -> np.ndarray:
    rotation = np.asarray(frame["rotation_world_from_left"], dtype=np.float64)
    translation_mm = np.asarray(frame["translation_world_from_left_mm"], dtype=np.float64)
    return (rotation @ (points_m * 1000.0).T).T + translation_mm


def viewer_from_world_mm(points_mm: np.ndarray) -> np.ndarray:
    return points_mm[..., [0, 2, 1]] * np.asarray([0.001, 0.001, -0.001])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-dir", type=Path, required=True)
    parser.add_argument("--template-motion", type=Path, required=True)
    parser.add_argument("--motion-output", type=Path, required=True)
    parser.add_argument("--mesh-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--title", default="PMPose COCO-17 → male SMPL 固定地面对照")
    parser.add_argument(
        "--limitations",
        default=(
            "Male SMPL per-frame fits using PMPose stereo 2-D and the fixed Pose2Mesh COCO-17 regressor. "
            "No temporal, contact, stored 3-D, or display smoothing. The walker is the existing coarse context model."
        ),
    )
    parser.add_argument(
        "--fit-contract",
        default="independent clean-zero 2-D fits; Pose2Mesh COCO-17 regressor; beta=0",
    )
    args = parser.parse_args()

    template = json.loads(args.template_motion.read_text(encoding="utf-8"))
    template_by_pair = {int(frame["pair"]): frame for frame in template["frames"]}
    fitted_dirs = sorted((args.window_dir / "frames").glob("frame_*"))
    if not fitted_dirs:
        raise FileNotFoundError(f"no fitted frames under {args.window_dir / 'frames'}")

    frames: list[dict] = []
    vertices_viewer: list[np.ndarray] = []
    faces = None
    for fitted_dir in fitted_dirs:
        pair = int(fitted_dir.name.split("_")[-1])
        if pair not in template_by_pair:
            raise KeyError(f"pair {pair} missing from fixed-ground template")
        frame = dict(template_by_pair[pair])
        with np.load(fitted_dir / "result.npz") as result:
            vertices_left_m = np.asarray(result["vertices_m"], dtype=np.float64)
            coco_left_m = np.asarray(result["coco_left_m"], dtype=np.float64)
            observed_left = np.asarray(result["observed_left"], dtype=np.float64)
            observed_right = np.asarray(result["observed_right"], dtype=np.float64)
            valid_left = np.asarray(result["valid_left"], dtype=bool)
            valid_right = np.asarray(result["valid_right"], dtype=bool)
            projected_left = np.asarray(result["projected_left_px"], dtype=np.float64)
            projected_right = np.asarray(result["projected_right_px"], dtype=np.float64)
            current_faces = np.asarray(result["faces"], dtype=np.uint32)
        if vertices_left_m.shape != (6890, 3) or coco_left_m.shape != (17, 3):
            raise ValueError(f"unexpected SMPL result shape at pair {pair}")
        if faces is None:
            faces = current_faces
        elif not np.array_equal(faces, current_faces):
            raise ValueError(f"SMPL topology changed at pair {pair}")

        vertices_world_mm = world_from_left(vertices_left_m, frame)
        coco_world_mm = world_from_left(coco_left_m, frame)
        vertices_viewer.append(viewer_from_world_mm(vertices_world_mm))

        left_error = np.linalg.norm(projected_left - observed_left[:, :2], axis=1)
        right_error = np.linalg.norm(projected_right - observed_right[:, :2], axis=1)
        supervised_left = valid_left & (observed_left[:, 2] >= 0.20)
        supervised_right = valid_right & (observed_right[:, 2] >= 0.20)
        supervised_left[14] = False
        supervised_right[14] = False
        errors = []
        for joint in range(17):
            values = []
            if supervised_left[joint]:
                values.append(float(left_error[joint]))
            if supervised_right[joint]:
                values.append(float(right_error[joint]))
            errors.append(None if not values else float(np.mean(values)))
        valid = (supervised_left | supervised_right).tolist()
        observation_state = [
            {
                "left_accepted": bool(supervised_left[j]),
                "right_accepted": bool(supervised_right[j]),
                "any_accepted": bool(supervised_left[j] or supervised_right[j]),
                "used_by_route": bool(supervised_left[j] or supervised_right[j]),
                "held_out": False,
            }
            for j in range(17)
        ]
        frame.update(
            points_mm=coco_world_mm.tolist(),
            errors_px=errors,
            valid=valid,
            strict=valid,
            observation_state=observation_state,
            wrist_handle_mm={"left": None, "right": None},
        )
        frames.append(frame)

    assert faces is not None
    motion = dict(template)
    motion.update(
        title=args.title,
        frame_count=len(frames),
        display_smoothing="none",
        line_style="solid_only",
        coordinate_frame="fixed_ground_z_up",
        limitations=args.limitations,
        frames=frames,
    )
    args.motion_output.parent.mkdir(parents=True, exist_ok=True)
    args.motion_output.write_text(json.dumps(motion, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    vertices_array = np.ascontiguousarray(vertices_viewer, dtype=np.float32)
    np.savez_compressed(args.mesh_output, vertices_viewer_m=vertices_array, faces=faces)
    metadata = {
        "schema_version": 1,
        "model_type": "SMPL",
        "gender": "male",
        "frame_count": len(frames),
        "vertex_count": 6890,
        "face_count": int(faces.shape[0]),
        "coordinate_frame": "fixed-ground viewer metres: [x_world, z_world, -y_world]",
        "topology_constant_across_frames": True,
        "fit_contract": args.fit_contract,
        "display_smoothing": "none",
    }
    args.metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": len(frames), "mesh": str(args.mesh_output), "motion": str(args.motion_output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
