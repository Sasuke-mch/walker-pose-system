#!/usr/bin/env python3
"""Build a self-contained interactive viewer for one fitted SMPL mesh."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np


HTML = '''<!doctype html><meta charset="utf-8"><title>male SMPL single-frame audit</title>
<style>html,body{margin:0;height:100%;background:#fff;color:#111;font:14px sans-serif}#c{width:100%;height:100%;display:block}#ui{position:fixed;z-index:2;left:16px;top:16px;background:#fffffff0;padding:12px;border:1px solid #000;line-height:1.45}.note{font-size:12px}</style>
<div id="ui"><b>male SMPL · single-frame geometry audit</b><br><span class="note">6890 vertices / 13776 faces · body-local display only · no display smoothing</span></div><canvas id="c"></canvas>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/"}}</script>
<script id="d" type="application/json">DATA</script><script type="module">import * as THREE from'three';import{OrbitControls}from'three/addons/controls/OrbitControls.js';const d=JSON.parse(document.querySelector('#d').textContent),ungzip=async x=>new Response(new Blob([Uint8Array.from(atob(x),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();const[vb,fb]=await Promise.all([ungzip(d.v),ungzip(d.f)]),v=new Float32Array(vb),f=new Uint32Array(fb),r=new THREE.WebGLRenderer({canvas:document.querySelector('#c'),antialias:true});r.setPixelRatio(Math.min(devicePixelRatio,2));r.setSize(innerWidth,innerHeight);const sc=new THREE.Scene();sc.background=new THREE.Color(0xffffff);const cam=new THREE.PerspectiveCamera(38,innerWidth/innerHeight,.01,20);cam.position.set(2.5,1.25,3.2);const ctl=new OrbitControls(cam,r.domElement);ctl.target.set(0,.85,0);ctl.update();sc.add(new THREE.HemisphereLight(0xffffff,0x777777,2.5));const dl=new THREE.DirectionalLight(0xffffff,2);dl.position.set(2,4,3);sc.add(dl);const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(v,3));g.setIndex(new THREE.BufferAttribute(f,1));g.computeVertexNormals();sc.add(new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:0xb98262,roughness:.75,side:THREE.DoubleSide})));const grid=new THREE.GridHelper(4,20,0x000000,0xaaaaaa);sc.add(grid);sc.add(new THREE.AxesHelper(.4));function loop(){r.render(sc,cam);requestAnimationFrame(loop)}loop();addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();r.setSize(innerWidth,innerHeight)});</script>'''


def axis_angle_matrix(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = np.load(args.result)
    vertices = np.asarray(result["vertices_m"], dtype=np.float64)
    if vertices.ndim == 3:
        if len(vertices) != 1:
            raise ValueError("single-frame viewer received multiple frames")
        vertices = vertices[0]
    if vertices.shape != (6890, 3):
        raise ValueError(f"expected (6890, 3), got {vertices.shape}")
    translation = np.asarray(result["translation_m"], dtype=np.float64).reshape(-1, 3)[0]
    root = np.asarray(result["root_orient"], dtype=np.float64).reshape(-1, 3)[0]
    rotation = axis_angle_matrix(root)
    display = (rotation.T @ (vertices - translation).T).T
    display[:, 1] -= display[:, 1].min()
    display[:, 0] -= np.median(display[:, 0])
    display[:, 2] -= np.median(display[:, 2])
    vertices_f32 = np.ascontiguousarray(display, dtype=np.float32)
    faces = np.ascontiguousarray(result["faces"], dtype=np.uint32)
    payload = {
        "v": base64.b64encode(gzip.compress(vertices_f32.tobytes(), 6)).decode(),
        "f": base64.b64encode(gzip.compress(faces.tobytes(), 9)).decode(),
    }
    args.output.write_text(
        HTML.replace("DATA", json.dumps(payload, separators=(",", ":"))),
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
