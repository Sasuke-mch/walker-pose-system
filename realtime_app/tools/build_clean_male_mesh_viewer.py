#!/usr/bin/env python3
from __future__ import annotations
import argparse,base64,gzip,json
from pathlib import Path
import numpy as np
HTML='''<!doctype html><meta charset="utf-8"><title>独立 male SMPL-X（身体对齐显示）</title><style>html,body{margin:0;height:100%;background:#101418;color:#eee;font:14px sans-serif}#c{width:100%;height:100%;display:block}#ui{position:fixed;z-index:2;left:16px;top:16px;background:#000c;padding:12px;border-radius:8px;line-height:1.45}input{width:360px}.note{color:#f0c674;font-size:12px}</style><div id="ui"><b>独立 male SMPL-X：身体对齐诊断视图</b><br><span id="t"></span><br><input id="s" type="range" min="0" max="MAX" value="0"><button id="p">播放</button><br><span class="note">固定身体对齐坐标；地面仅为显示基准；未伪造助步器</span></div><canvas id="c"></canvas><script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/"}}</script><script id="d" type="application/json">DATA</script><script type="module">import * as THREE from'three';import{OrbitControls}from'three/addons/controls/OrbitControls.js';const d=JSON.parse(document.querySelector('#d').textContent),ungzip=async x=>new Response(new Blob([Uint8Array.from(atob(x),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();const[vb,fb]=await Promise.all([ungzip(d.v),ungzip(d.f)]),v=new Float32Array(vb),f=new Uint32Array(fb),r=new THREE.WebGLRenderer({canvas:document.querySelector('#c'),antialias:true});r.setPixelRatio(Math.min(devicePixelRatio,2));r.setSize(innerWidth,innerHeight);const sc=new THREE.Scene();sc.background=new THREE.Color(0x101418);const cam=new THREE.PerspectiveCamera(42,innerWidth/innerHeight,.01,20);cam.position.set(2.3,1.45,3.0);const ctl=new OrbitControls(cam,r.domElement);ctl.target.set(0,0.85,0);ctl.update();sc.add(new THREE.HemisphereLight(0xffffff,0x334455,3));const dl=new THREE.DirectionalLight(0xffffff,2);dl.position.set(2,4,3);sc.add(dl);const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(d.n*3),3));g.setIndex(new THREE.BufferAttribute(f,1));const m=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:0xb98262,roughness:.75,side:THREE.DoubleSide}));sc.add(m);const grid=new THREE.GridHelper(4,20,0x777777,0x3f474f);grid.position.y=0;sc.add(grid);sc.add(new THREE.AxesHelper(.4));let i=0,play=false,last=0;const frameMs=1000/d.fps,s=document.querySelector('#s'),t=document.querySelector('#t');function set(k){i=k|0;const a=g.attributes.position.array,o=i*d.n*3;a.set(v.subarray(o,o+a.length));g.attributes.position.needsUpdate=true;g.computeVertexNormals();s.value=i;t.textContent=`帧 ${i+1}/${d.frames} · ${d.fps} fps`};s.oninput=()=>set(+s.value);document.querySelector('#p').onclick=()=>play=!play;set(0);function loop(ts){if(play&&ts-last>=frameMs){set((i+1)%d.frames);last=ts}r.render(sc,cam);requestAnimationFrame(loop)}requestAnimationFrame(loop);addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();r.setSize(innerWidth,innerHeight)});</script>'''

def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("cannot construct body-aligned display axis")
    return vector / norm

def body_aligned_vertices(z: np.lib.npyio.NpzFile) -> tuple[np.ndarray, dict]:
    vertices = np.asarray(z['vertices_m'], dtype=np.float64)
    tri = np.asarray(z['triangulated_m'], dtype=np.float64)
    accepted = np.asarray(z['triangulated_accepted'], dtype=bool)
    required = accepted[:, [5, 6, 11, 12]].all(axis=1)
    required &= np.isfinite(tri[:, [5, 6, 11, 12]]).all(axis=(1, 2))
    if int(required.sum()) < 3:
        raise ValueError("not enough fresh shoulder/hip observations for display alignment")
    selected = tri[required]
    shoulder_mid = .5 * (selected[:, 5] + selected[:, 6])
    hip_mid = .5 * (selected[:, 11] + selected[:, 12])
    up_samples = shoulder_mid - hip_mid
    up_samples /= np.linalg.norm(up_samples, axis=1, keepdims=True).clip(1e-8)
    up = unit(np.median(up_samples, axis=0))
    right_samples = .5 * ((selected[:, 6] - selected[:, 5]) + (selected[:, 12] - selected[:, 11]))
    right_samples -= (right_samples @ up)[:, None] * up[None]
    right_samples /= np.linalg.norm(right_samples, axis=1, keepdims=True).clip(1e-8)
    right = unit(np.median(right_samples, axis=0))
    right = unit(right - up * float(np.dot(right, up)))
    forward = unit(np.cross(right, up))
    rotation = np.stack([right, up, forward], axis=0)
    origin = np.median(hip_mid, axis=0)
    aligned = np.einsum('ij,fvj->fvi', rotation, vertices - origin)
    frame_bottom = aligned[:, :, 1].min(axis=1)
    display_ground = float(np.percentile(frame_bottom, 5))
    aligned[:, :, 1] -= display_ground
    metadata = {
        'coordinate_note': 'fixed body-aligned display frame derived only from fresh shoulder/hip triangulation; not calibrated gravity or ground',
        'accepted_alignment_frames': int(required.sum()),
        'rotation_display_from_left_camera': rotation.tolist(),
        'origin_left_camera_m': origin.tolist(),
        'display_ground_before_shift_m': display_ground,
    }
    return np.ascontiguousarray(aligned, dtype=np.float32), metadata

def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fps',type=float,default=30.0);a=p.parse_args();z=np.load(a.result);v,display=body_aligned_vertices(z);f=np.ascontiguousarray(z['faces'],dtype=np.uint32);d={'frames':len(v),'n':v.shape[1],'fps':a.fps,'display':display,'v':base64.b64encode(gzip.compress(v.tobytes(),6)).decode(),'f':base64.b64encode(gzip.compress(f.tobytes(),9)).decode()};a.output.write_text(HTML.replace('MAX',str(len(v)-1)).replace('DATA',json.dumps(d,separators=(',',':'))),encoding='utf-8');print(a.output)
if __name__=='__main__':main()
