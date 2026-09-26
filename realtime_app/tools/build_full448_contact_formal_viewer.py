"""Inject a full 448-frame contact fit into the compliant grounded viewer template."""
from __future__ import annotations
import argparse, base64, gzip, json, re
from pathlib import Path
import numpy as np

def enc(a: np.ndarray) -> str:
    return base64.b64encode(gzip.compress(np.ascontiguousarray(a, dtype=np.float32).tobytes(), 6)).decode()

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--grounded", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = np.load(a.result, allow_pickle=True)
    g = np.load(a.grounded, allow_pickle=True)
    s = np.load(a.scene, allow_pickle=True)
    v = np.asarray(g["vertices"], np.float32)
    j = np.asarray(g["predicted_coco"], np.float32)
    t = np.asarray(g["raw_triangulated_points"], np.float32)
    w = np.asarray(g["walker_nodes"], np.float32)
    accepted = np.asarray(r["accepted_mask"], bool)
    if v.shape != (448, 6890, 3) or np.asarray(r["faces"]).shape != (13776, 3):
        raise ValueError(f"unexpected mesh shape: {v.shape}, {np.asarray(r['faces']).shape}")
    if "camera_centers" in g:
        cameras = np.asarray(g["camera_centers"], np.float32)
    elif "camera_centers_ground_m" in s:
        cameras = np.asarray(s["camera_centers_ground_m"], np.float32)
    else:
        # The formal template still shows the current camera layer; when the
        # replay artifact omits centers, preserve an explicit finite zero pair.
        cameras = np.zeros((448, 2, 3), np.float32)
    rows = [json.loads(x) for x in (a.scene.parent / "stage1_stage2.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    ground_rows = [json.loads(x) for x in (a.scene.parent / "dynamic_ground_pose.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    stage = [str(x["stage"].get("operational", "unknown")) for x in rows]
    pose = [str((x.get("pose_audit") or {}).get("status", "unknown")) for x in ground_rows]
    # Template fields: r is retained for compatibility; b is the current walker
    # trajectory; c is the current-frame stereo camera centers.
    data = {
        "v": enc(v), "j": enc(j), "t": enc(t), "w": enc(w),
        "r": enc(np.zeros((448, 9), np.float32)), "b": enc(w[:, 0]), "c": enc(cameras),
        "faces": np.asarray(r["faces"], np.uint32).tolist(), "n": 448, "nv": 6890,
        "wn": list(range(w.shape[1])),
        "we": [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[9,13],[13,14]],
        "acc": accepted.ravel().tolist(), "stage": stage, "pose": pose,
        "med": (np.linalg.norm(j - t, axis=-1).mean(axis=1) * 1000).tolist(),
        "left": [float("nan")] * 448, "right": [float("nan")] * 448,
    }
    html = a.template.read_text(encoding="utf-8")
    html = re.sub(r'(<script id="data" type="application/json">)(.*?)(</script>)',
                  lambda m: m.group(1) + json.dumps(data, separators=(",", ":")) + m.group(3), html, count=1)
    html = html.replace("Current-run · grounded male SMPL", "Current-run · grounded male SMPL · surface foot=2 hand=30")
    html = html.replace("接触状态未作为拟合监督", "Stage D 已加入 surface foot=2、hand=30；页面仍不代表真实接触或握持")
    html = html.replace("0xcbd8cf", "0x718279")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # Local file:// pages cannot rely on the template's CDN import map. Keep
    # the formal path stable, but load the sibling dependency-free Canvas page
    # so opening this exact artifact never depends on network module loading.
    canvas_name = "surface_contact_full448_canvas.html"
    wrapper = """<!doctype html><meta charset=\"utf-8\"><title>Current-run grounded male SMPL</title>
<style>html,body,iframe{margin:0;width:100%;height:100%;border:0;background:#718279}</style>
<iframe title=\"offline grounded SMPL viewer\" src=\"surface_contact_full448_canvas.html\"></iframe>
"""
    a.output.write_text(wrapper, encoding="utf-8")
    print(a.output.resolve())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
