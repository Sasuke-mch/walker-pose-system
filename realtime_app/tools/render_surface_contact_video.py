#!/usr/bin/env python3
"""Render a grounded SMPL/contact result as an auditable fixed-view MP4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1),
         (0, 2), (1, 3), (2, 4))


def display(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    y[..., 1] *= 1.0
    return y


def cylinder_faces(a: np.ndarray, b: np.ndarray, radius: float = 0.018,
                   sides: int = 8) -> list[np.ndarray]:
    direction = np.asarray(b - a, np.float64)
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return []
    direction /= length
    helper = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(direction, helper); u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    angles = np.linspace(0, 2 * np.pi, sides + 1)
    rings = [[center + radius * (np.cos(t) * u + np.sin(t) * v) for t in angles]
             for center in (a, b)]
    return [np.asarray([rings[0][k], rings[0][k + 1], rings[1][k + 1], rings[1][k]])
            for k in range(sides)]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--result-grounded", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--walker", type=Path, required=True)
    p.add_argument("--surface-sets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--trail-frames", type=int, default=12)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--title", default="SMPL surface contact fit")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    d = np.load(args.result_grounded, allow_pickle=True)
    V = np.asarray(d["vertices"], np.float32)
    J = np.asarray(d["predicted_coco"], np.float32)
    T = np.asarray(d["raw_triangulated_points"], np.float32)
    W = np.asarray(d["walker_nodes"], np.float32)
    faces = np.asarray(d["faces"], np.int32)
    accepted = np.asarray(d["accepted_mask"], bool)
    full_n = V.shape[0]
    n = full_n if args.max_frames is None else min(full_n, args.max_frames)
    if V.shape[1:] != (6890, 3) or faces.shape != (13776, 3) or T.shape != (full_n, 17, 3):
        raise ValueError(f"unexpected result shapes: V={V.shape}, faces={faces.shape}, T={T.shape}")
    scene = np.load(args.scene, allow_pickle=True)
    R = np.asarray(scene["rotation_ground_from_left"], np.float32)[:n]
    trans = np.asarray(scene["translation_ground_from_left_mm"], np.float32)[:n] / 1000.0
    static = json.loads(args.walker.read_text(encoding="utf-8-sig"))
    names = list(static["nodes_walker_mm"])
    idx = {name: i for i, name in enumerate(names)}
    edges = [(idx[a], idx[b]) for a, b in static["edges"]]
    handle_edges = [(idx[a], idx[b]) for a, b in static["handle_segments"].values()]
    right_left = np.asarray(static["camera_centers_left_camera_mm"]["right"], np.float32) / 1000.0
    cameras = np.zeros((n, 2, 3), np.float32)
    cameras[:, 0] = trans
    cameras[:, 1] = np.einsum("nij,j->ni", R, right_left) + trans
    Vd, Td, Wd, Cd = display(V), display(T), display(W), display(cameras)
    flat = Vd.reshape(-1, 3)
    all_points = np.concatenate((flat, Wd.reshape(-1, 3), Td.reshape(-1, 3), Cd.reshape(-1, 3)), axis=0)
    lo = np.nanpercentile(all_points, 0.5, axis=0) - np.array([0.7, 0.7, 0.15])
    hi = np.nanpercentile(all_points, 99.5, axis=0) + np.array([0.7, 0.7, 0.45])
    lo[2] = min(lo[2], -0.12); hi[2] = max(hi[2], 1.8)
    side_x = float(hi[0])

    fig = plt.figure(figsize=(12.8, 7.2), dpi=75)
    ax = fig.add_subplot(111, projection="3d")
    canvas = FigureCanvasAgg(fig)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi - lo).tolist())
    # Use reference-video physical [X,Y,Z] coordinates without reflection.
    # Derive the same oblique camera view from the walking displacement.
    movement = Cd[-1, 0, :2] - Cd[0, 0, :2]
    view_azim = float(np.degrees(np.arctan2(movement[1], movement[0])) - 35.0) if np.linalg.norm(movement) else -58.0
    ax.view_init(elev=23, azim=view_azim)
    ax.set_xlabel("Ground X (m)"); ax.set_ylabel("Ground Y (m)"); ax.set_zlabel("Height Z (m)")
    ax.set_title(args.title)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
    gx = np.linspace(lo[0], hi[0], 12); gy = np.linspace(lo[1], hi[1], 12)
    xx, yy = np.meshgrid(gx, gy)
    ax.plot_surface(xx, yy, np.zeros_like(xx), color="#718279", alpha=0.72, linewidth=0, shade=False)
    for x in gx:
        ax.plot([x, x], [lo[1], hi[1]], [0, 0], color="#b9c8c0", lw=0.5, alpha=0.8)
    for y in gy:
        ax.plot([lo[0], hi[0]], [y, y], [0, 0], color="#b9c8c0", lw=0.5, alpha=0.8)
    # Two open room walls, as in the supplied reference video. Their gridlines
    # meet the Z=0 floor; the space elsewhere stays light and unfilled.
    wall_color = "#a9b4b9"
    wall_z = np.linspace(0, hi[2], 8)
    for y in gy:
        ax.plot([side_x, side_x], [y, y], [0, hi[2]], color=wall_color, lw=0.7, alpha=0.7)
    for z in wall_z:
        ax.plot([side_x, side_x], [lo[1], hi[1]], [z, z], color=wall_color, lw=0.7, alpha=0.7)
    # The second wall is behind the subject walking along negative ground Y.
    for x in gx:
        ax.plot([x, x], [hi[1], hi[1]], [0, hi[2]], color=wall_color, lw=0.7, alpha=0.7)
    for z in wall_z:
        ax.plot([lo[0], hi[0]], [hi[1], hi[1]], [z, z], color=wall_color, lw=0.7, alpha=0.7)
    axis_len = max(2.5, float(max(hi[0] - lo[0], hi[1] - lo[1]) * 0.72))
    ax.plot([0, axis_len], [0, 0], [0, 0], color="#ba3030", lw=3)
    ax.plot([0, 0], [0, axis_len], [0, 0], color="#1c7a43", lw=3)
    ax.plot([0, 0], [0, 0], [0, 2.2], color="#365bb5", lw=3)
    # Camera trails will be updated per frame on the side wall; no future path leaks into a frame.
    camera_side_trails = [ax.plot([], [], [], color=color, lw=2.2, alpha=0.9)[0]
                          for color in ("#aa4a18", "#7b3f98")]
    mesh = Poly3DCollection([Vd[0][f] for f in faces], facecolor="#bd8060", edgecolor="none", alpha=0.68)
    ax.add_collection3d(mesh)
    sk_lines = []
    for a, b in EDGES:
        line, = ax.plot([], [], [], color="#101820", lw=2.5)
        sk_lines.append(line)
    walker_tubes = Poly3DCollection([], facecolor="#176b7e", edgecolor="#0d3640", linewidth=0.2)
    ax.add_collection3d(walker_tubes)
    handle_tubes = Poly3DCollection([], facecolor="#f57c00", edgecolor="#55305f", linewidth=0.2)
    ax.add_collection3d(handle_tubes)
    model_sc = ax.scatter([], [], [], s=13, color="#101820", depthshade=False)
    tri_sc = ax.scatter([], [], [], s=10, color="#c62828", depthshade=False)
    rejected_tri_sc = ax.scatter([], [], [], s=18, color="#b36b1e", depthshade=False, marker="x")
    foot_sc = ax.scatter([], [], [], s=22, color="#1b7f4a", depthshade=False)
    foot_below_sc = ax.scatter([], [], [], s=26, color="#c62828", depthshade=False)
    rejected_lines = []
    for _ in EDGES:
        line, = ax.plot([], [], [], color="#b36b1e", lw=1.8, ls="--")
        rejected_lines.append(line)
    camera_sc = [ax.scatter([], [], [], s=34, color=c, depthshade=False) for c in ("#aa4a18", "#7b3f98")]
    ankle_sc = [ax.scatter([], [], [], s=12, color=c, depthshade=False) for c in ("#1e6b8b", "#8b5a20")]
    camera_drop = [ax.plot([], [], [], color="#819199", lw=1, ls=":")[0] for _ in range(2)]

    sets_doc = json.loads(args.surface_sets.read_text(encoding="utf-8"))
    sole_parts = []
    for side in ("left", "right"):
        item = sets_doc["sets"][f"{side}_sole_surface_candidate"]
        sole_parts.extend(np.asarray(item[k], np.int32) for k in ("heel", "ball", "toe"))
    sole_idx = np.unique(np.concatenate(sole_parts)).astype(int)
    sole_idx = sole_idx[(sole_idx >= 0) & (sole_idx < 6890)]
    frame_label = ax.text2D(0.02, 0.96, "", transform=ax.transAxes)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (960, 540))
    if not writer.isOpened(): raise RuntimeError(f"cannot open video writer: {args.output}")
    try:
        for i in range(n):
            mesh.set_verts([Vd[i][f] for f in faces])
            for k, (a, b) in enumerate(EDGES):
                A, B = Td[i, a], Td[i, b]
                finite = np.isfinite(A).all() and np.isfinite(B).all()
                accepted_edge = bool(accepted[i, a] and accepted[i, b])
                target = sk_lines[k] if accepted_edge else rejected_lines[k]
                target.set_data_3d(([A[0], B[0]], [A[1], B[1]], [A[2], B[2]]) if finite else ([], [], []))
                (sk_lines[k] if not accepted_edge else rejected_lines[k]).set_data_3d(([], [], []))
            walker_tubes.set_verts([face for a, b in edges for face in cylinder_faces(Wd[i, a], Wd[i, b])])
            handle_tubes.set_verts([face for a, b in handle_edges
                                    for face in cylinder_faces(Wd[i, a], Wd[i, b], radius=0.022)])
            model_sc._offsets3d = (J[i, :, 0], J[i, :, 1], J[i, :, 2])
            valid_tri = np.isfinite(T[i]).all(axis=1) & accepted[i]
            tri = Td[i, valid_tri]
            tri_sc._offsets3d = (tri[:, 0], tri[:, 1], tri[:, 2])
            finite_rejected = np.isfinite(T[i]).all(axis=1) & ~accepted[i]
            rejected_tri = Td[i, finite_rejected]
            rejected_tri_sc._offsets3d = (rejected_tri[:, 0], rejected_tri[:, 1], rejected_tri[:, 2])
            fp = Vd[i, sole_idx]
            below = fp[:, 2] < 0
            foot_sc._offsets3d = (fp[~below, 0], fp[~below, 1], fp[~below, 2])
            foot_below_sc._offsets3d = (fp[below, 0], fp[below, 1], fp[below, 2])
            for side in range(2):
                c = Cd[i, side]
                camera_sc[side]._offsets3d = ([c[0]], [c[1]], [c[2]])
                camera_drop[side].set_data_3d(([c[0], side_x], [c[1], c[1]], [c[2], c[2]]))
                q0 = max(0, i - args.trail_frames)
                trail = Cd[q0:i + 1, side]
                camera_side_trails[side].set_data_3d((np.full(len(trail), side_x), trail[:, 1], trail[:, 2]))
                ankle = Td[max(0, i - args.trail_frames):i + 1, 15 + side]
                ankle[:, 2] = 0
                ankle_sc[side]._offsets3d = (ankle[:, 0], ankle[:, 1], ankle[:, 2])
            frame_label.set_text(f"frame {i:03d}/{n-1:03d}  |  surface mesh + COCO skeleton + solid walker")
            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba())
            frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release(); plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps({"frames": n, "fps": args.fps, "surface_vertices": 6890, "faces": 13776, "walker_members": len(edges), "walker_geometry": "eight-sided 3D cylinder meshes from static topology", "skeleton": "raw triangulated COCO-17; rejected finite edges dashed", "camera_projection": "YZ side wall, short past-only trail", "ankle_trail_frames": args.trail_frames, "ground": "Z=0 with two gridded room walls; rear wall behind walking direction", "display_coordinates": "X,Y,Z physical ground (reference video)", "view_elevation": 23, "view_azimuth": view_azim, "sole_vertices": int(len(sole_idx)), "input": str(args.result_grounded.resolve())}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
