#!/usr/bin/env python3
"""Build a self-contained no-smoothing viewer for an SMPL fitting window."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np


COCO_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
)


HTML = r'''<!doctype html><meta charset="utf-8"><title>male SMPL COCO-17 fit window</title>
<style>
html,body{margin:0;height:100%;background:#fff;color:#111;font:14px sans-serif;overflow:hidden}
#c{width:100%;height:100%;display:block}
#ui{position:fixed;z-index:2;left:16px;top:16px;background:#fffffff0;padding:12px;border:1px solid #000;line-height:1.45;max-width:430px}
#row{display:flex;gap:8px;align-items:center;margin-top:8px}input[type=range]{width:260px}.note{font-size:12px}
button{background:#fff;border:1px solid #000;padding:4px 10px}
</style>
<div id="ui"><b>male SMPL · PMPose COCO-17 fit · frames 60–90</b><br>
<span class="note">6890 vertices / 13776 faces · body-local audit · no temporal/contact/display smoothing<br>
orange: fitted SMPL surface · black points/lines: regressor-produced model-side COCO-17</span>
<div id="row"><button id="play">Play</button><input id="time" type="range" min="0" max="30" value="0" step="1"><span id="label"></span></div>
<div class="note" id="metric"></div></div><canvas id="c"></canvas>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/"}}</script>
<script id="d" type="application/json">DATA</script>
<script type="module">
import * as THREE from'three';import{OrbitControls}from'three/addons/controls/OrbitControls.js';
const d=JSON.parse(document.querySelector('#d').textContent),ungzip=async x=>new Response(new Blob([Uint8Array.from(atob(x),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
const[vb,jb,fb]=await Promise.all([ungzip(d.v),ungzip(d.j),ungzip(d.f)]),verts=new Float32Array(vb),joints=new Float32Array(jb),faces=new Uint32Array(fb);
const r=new THREE.WebGLRenderer({canvas:document.querySelector('#c'),antialias:true});r.setPixelRatio(Math.min(devicePixelRatio,2));r.setSize(innerWidth,innerHeight);
const sc=new THREE.Scene();sc.background=new THREE.Color(0xffffff);const cam=new THREE.PerspectiveCamera(38,innerWidth/innerHeight,.01,20);cam.position.set(2.5,1.3,3.2);const ctl=new OrbitControls(cam,r.domElement);ctl.target.set(0,.85,0);ctl.update();
sc.add(new THREE.HemisphereLight(0xffffff,0x777777,2.5));const dl=new THREE.DirectionalLight(0xffffff,2);dl.position.set(2,4,3);sc.add(dl);sc.add(new THREE.GridHelper(4,20,0x000000,0xaaaaaa));sc.add(new THREE.AxesHelper(.4));
const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(d.nv*3),3));g.setIndex(new THREE.BufferAttribute(faces,1));const mesh=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:0xb98262,roughness:.75,side:THREE.DoubleSide}));sc.add(mesh);
const jm=new THREE.InstancedMesh(new THREE.SphereGeometry(.018,12,8),new THREE.MeshBasicMaterial({color:0x000000}),17);sc.add(jm);const linePos=new Float32Array(d.edges.length*6),lg=new THREE.BufferGeometry();lg.setAttribute('position',new THREE.BufferAttribute(linePos,3));const lines=new THREE.LineSegments(lg,new THREE.LineBasicMaterial({color:0x000000}));sc.add(lines);const tmp=new THREE.Object3D();
const slider=document.querySelector('#time'),label=document.querySelector('#label'),metric=document.querySelector('#metric');let playing=false,last=0;
function setFrame(i){i=Number(i);const vo=i*d.nv*3,jo=i*17*3;g.attributes.position.array.set(verts.subarray(vo,vo+d.nv*3));g.attributes.position.needsUpdate=true;g.computeVertexNormals();for(let k=0;k<17;k++){tmp.position.set(joints[jo+3*k],joints[jo+3*k+1],joints[jo+3*k+2]);tmp.updateMatrix();jm.setMatrixAt(k,tmp.matrix)}jm.instanceMatrix.needsUpdate=true;for(let e=0;e<d.edges.length;e++){const[a,b]=d.edges[e];for(let q=0;q<3;q++){linePos[e*6+q]=joints[jo+a*3+q];linePos[e*6+3+q]=joints[jo+b*3+q]}}lg.attributes.position.needsUpdate=true;label.textContent='frame '+d.frames[i];const m=d.metrics[i];metric.textContent='strict reprojection P95: left '+m.left.toFixed(2)+' px / right '+m.right.toFixed(2)+' px';slider.value=i}
slider.addEventListener('input',()=>{playing=false;document.querySelector('#play').textContent='Play';setFrame(slider.value)});document.querySelector('#play').onclick=()=>{playing=!playing;document.querySelector('#play').textContent=playing?'Pause':'Play'};
function loop(t){if(playing&&t-last>100){setFrame((Number(slider.value)+1)%d.frames.length);last=t}r.render(sc,cam);requestAnimationFrame(loop)}setFrame(0);loop(0);addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();r.setSize(innerWidth,innerHeight)});
</script>'''


def axis_angle_matrix(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def encode(array: np.ndarray, level: int = 6) -> str:
    return base64.b64encode(gzip.compress(np.ascontiguousarray(array).tobytes(), level)).decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame_dirs = sorted((args.window_dir / "frames").glob("frame_*"))
    if not frame_dirs:
        raise FileNotFoundError(f"no fitted frames under {args.window_dir / 'frames'}")

    vertices_all, joints_all, frame_ids, metrics = [], [], [], []
    faces = None
    for frame_dir in frame_dirs:
        result_path = frame_dir / "result.npz"
        metric_path = frame_dir / "metrics.json"
        if not result_path.is_file() or not metric_path.is_file():
            raise FileNotFoundError(f"incomplete fitted frame: {frame_dir}")
        result = np.load(result_path)
        vertices = np.asarray(result["vertices_m"], dtype=np.float64)
        joints = np.asarray(result["coco_left_m"], dtype=np.float64)
        root = np.asarray(result["root_orient"], dtype=np.float64).reshape(3)
        translation = np.asarray(result["translation_m"], dtype=np.float64).reshape(3)
        rotation = axis_angle_matrix(root)
        vertices = (rotation.T @ (vertices - translation).T).T
        joints = (rotation.T @ (joints - translation).T).T
        floor = float(vertices[:, 1].min())
        center_x = float(np.median(vertices[:, 0]))
        center_z = float(np.median(vertices[:, 2]))
        shift = np.asarray([center_x, floor, center_z])
        vertices_all.append(vertices - shift)
        joints_all.append(joints - shift)
        frame_ids.append(int(frame_dir.name.split("_")[-1]))
        report = json.loads(metric_path.read_text(encoding="utf-8"))
        metrics.append({
            "left": float(report["final"]["left_p95_px"]),
            "right": float(report["final"]["right_p95_px"]),
        })
        current_faces = np.asarray(result["faces"], dtype=np.uint32)
        if faces is None:
            faces = current_faces
        elif not np.array_equal(faces, current_faces):
            raise ValueError(f"face topology changed at {frame_dir}")

    vertices_f32 = np.asarray(vertices_all, dtype=np.float32)
    joints_f32 = np.asarray(joints_all, dtype=np.float32)
    assert faces is not None
    payload = {
        "v": encode(vertices_f32),
        "j": encode(joints_f32),
        "f": encode(faces, 9),
        "nv": 6890,
        "frames": frame_ids,
        "edges": COCO_EDGES,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML.replace("DATA", json.dumps(payload, separators=(",", ":"))), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
