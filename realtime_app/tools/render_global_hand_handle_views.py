"""Render three static views of the solved hand/handle interaction candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pose_app.global_hand_handle_pose import _frame_from_points, _handle_frame


def display(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    return np.stack((p[..., 0], -p[..., 1], p[..., 2]), axis=-1)


def cylinder(ax, a, b, radius=0.018, color="#4d5963", alpha=.95):
    a, b = np.asarray(a), np.asarray(b)
    axis = b - a; length = np.linalg.norm(axis)
    if length < 1e-9: return
    axis /= length
    ref = np.array([0., 0., 1.]) if abs(axis[2]) < .9 else np.array([1., 0., 0.])
    u = np.cross(axis, ref); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    t = np.linspace(0, 2*np.pi, 20)
    z = np.linspace(0, length, 2)
    X = np.zeros((2, len(t), 3))
    for i, zz in enumerate(z): X[i] = a + axis*zz + radius*(np.cos(t)[:, None]*u + np.sin(t)[:, None]*v)
    ax.plot_surface(X[..., 0], X[..., 1], X[..., 2], color=color, alpha=alpha, linewidth=0)


def solved_cloud(palm, ends, prior):
    hf = _handle_frame(*ends)
    if hf is None: raise ValueError("degenerate handle")
    hc, Rh = hf
    pf = _frame_from_points(palm)
    if pf is None: raise ValueError("degenerate palm")
    pc, Rp = pf
    signs = np.sign(np.diag(Rh.T @ Rp)); signs[signs == 0] = 1
    Rp = Rp @ np.diag(signs)
    if np.linalg.det(Rp) < 0: Rp[:, -1] *= -1
    local = (Rp.T @ (palm - pc).T).T
    Rmean = np.asarray(prior["hand_frame_to_handle_rotation"], float)
    offset = np.asarray(prior["palm_offset_handle_m"], float)
    return hc + (Rh @ (local @ Rmean.T + offset).T).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--pose", type=Path, required=True)
    ap.add_argument("--walker-model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--frame", type=int, default=None)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.result, allow_pickle=False)
    pose = json.loads(args.pose.read_text(encoding="utf-8"))
    model = json.loads(args.walker_model.read_text(encoding="utf-8"))
    nodes = {k: np.asarray(v, float)/1000 for k,v in model["nodes_initial_ground_mm"].items()}
    edges = model["edges"]
    handles = model["handle_segments"]
    frame = int(args.frame if args.frame is not None else z["left_palm_ground_m"].shape[0]//2)
    clouds = {}
    for side, key in (("left", "right"), ("right", "left")):
        ends = np.asarray([nodes[n] for n in handles[side]])
        clouds[side] = solved_cloud(z[f"{side}_palm_ground_m"][frame], ends, pose["hands"][side])
    walker_edges = [(nodes[a], nodes[b]) for a,b in edges]
    # Local interaction view: include only handle segments and solved palms.
    # Keeping separation visible is intentional for this diagnostic.
    pts = [nodes[n] for side in handles.values() for n in side] + [p for c in clouds.values() for p in c]
    lim = display(np.asarray(pts)); lo, hi = lim.min(axis=0), lim.max(axis=0); center=(lo+hi)/2
    span=max(hi-lo)*1.35; center[2]=max(center[2], .75)
    views = {"front": (18, -72), "side": (12, 18), "top": (78, -72)}
    for name, (elev, azim) in views.items():
        fig = plt.figure(figsize=(12, 8), dpi=180); ax=fig.add_subplot(111, projection="3d")
        for a,b in walker_edges: cylinder(ax, display(a), display(b), .014, "#69747e")
        for side, color in (("left", "#d94b4b"), ("right", "#2d73c8")):
            q=display(clouds[side]); ax.scatter(q[:,0],q[:,1],q[:,2],s=3,c=color,alpha=.72,label=f"{side} solved palm")
            h=display(np.asarray([nodes[n] for n in handles[side]])); ax.plot(h[:,0],h[:,1],h[:,2],lw=4,c=color,alpha=.9)
            mid=h.mean(axis=0); ax.scatter(*mid,s=45,c="#111111",marker="x")
        ax.set_xlim(center[0]-span/2,center[0]+span/2); ax.set_ylim(center[1]-span/2,center[1]+span/2); ax.set_zlim(max(0,center[2]-span/2),center[2]+span/2)
        ax.set_xlabel("X display (m)"); ax.set_ylabel("-Y display (m)"); ax.set_zlabel("Z display (m)")
        ax.view_init(elev=elev, azim=azim); ax.set_title(f"Global fixed hand–walker pose diagnostic | frame {frame} | {name}\nstatic walker geometry; no external handle trajectory")
        ax.legend(loc="upper left"); ax.grid(True, alpha=.25); fig.tight_layout()
        fig.savefig(args.output_dir/f"hand_handle_{name}.png", facecolor="white"); plt.close(fig)


if __name__ == "__main__": main()
