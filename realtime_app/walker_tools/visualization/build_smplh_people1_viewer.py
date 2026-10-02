#!/usr/bin/env python3
"""Export a self-contained SMPL-H people1 audit viewer from one fit result."""
from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()
import argparse, base64, gzip, json, pickle, sys, inspect
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

def b64(a: np.ndarray) -> str:
    return base64.b64encode(gzip.compress(np.ascontiguousarray(a).tobytes(), 6)).decode("ascii")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--ground", type=Path, required=True)
    ap.add_argument("--smplh", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    with np.load(args.result) as z:
        vertices = z["vertices"].astype(np.float32)
        faces = z["faces"].astype(np.uint32)
        pred = z["predicted_coco"].astype(np.float32)
        raw = z["raw_triangulated_points"].astype(np.float32) / 1000.0
        accepted = z["body_accepted"].astype(bool)
        betas = z["betas"].astype(np.float32)
        root = z["global_orient"].astype(np.float32)
        transl = z["transl"].astype(np.float32)
        body = z["body_pose"].astype(np.float32)
        lh = z["left_hand_pose"].astype(np.float32)
        rh = z["right_hand_pose"].astype(np.float32)
        wl = z["wilor_left_2d_full"].astype(np.float32)
        wr = z["wilor_right_2d_full"].astype(np.float32)
        wlv = z["wilor_left_valid_full"].astype(bool)
        wrv = z["wilor_right_valid_full"].astype(bool)
    g = json.loads(args.ground.read_text(encoding="utf-8"))
    R = np.asarray(g["rotation_ground_from_left"], np.float32)
    t = np.asarray(g["translation_ground_from_left_mm"], np.float32) / 1000.0
    def xf(x): return (R @ x.reshape(-1, 3).T).T.reshape(x.shape) + t
    vertices_g = xf(vertices); pred_g = xf(pred); raw_g = xf(raw)

    # Recompute the SMPL-H internal joints from the exact saved parameters.
    import torch
    for _n, _v in {"bool": np.bool_, "int": np.int64, "float": np.float64, "complex": np.complex128, "object": np.object_, "unicode": np.str_, "str": np.str_}.items():
        if not hasattr(np, _n): setattr(np, _n, _v)
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    import smplx
    from smplx.utils import Struct
    with args.smplh.open("rb") as h: md = pickle.load(h, encoding="latin1")
    md.setdefault("hands_componentsl", np.eye(45, dtype=np.float32)); md.setdefault("hands_componentsr", np.eye(45, dtype=np.float32))
    md.setdefault("hands_meanl", np.zeros(45, dtype=np.float32)); md.setdefault("hands_meanr", np.zeros(45, dtype=np.float32))
    model = smplx.SMPLH(str(args.smplh), data_struct=Struct(**md), gender="male", use_pca=False, flat_hand_mean=True, batch_size=len(vertices))
    with torch.no_grad():
        out = model(betas=torch.tensor(np.repeat(betas, len(vertices), axis=0)), global_orient=torch.tensor(root), body_pose=torch.tensor(body), left_hand_pose=torch.tensor(lh), right_hand_pose=torch.tensor(rh), transl=torch.tensor(transl), return_verts=False)
    joints_g = xf(out.joints.numpy().astype(np.float32))
    payload = {
        "vertices": b64(vertices_g), "faces": b64(faces), "pred": b64(pred_g), "raw": b64(raw_g), "accepted": b64(accepted.astype(np.uint8)),
        "joints": b64(joints_g), "wl": b64(wl), "wr": b64(wr), "wlv": b64(wlv.astype(np.uint8)), "wrv": b64(wrv.astype(np.uint8)),
        "frames": len(vertices), "verts": vertices.shape[1], "faces_n": len(faces), "joints_n": joints_g.shape[1], "ground": g.get("schema_version", "unknown")
    }
    data = json.dumps(payload, separators=(",", ":"))
    html = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>people1 SMPL-H 工程候选可视化</title>
<style>body{margin:0;background:#f4f6f5;color:#17201d;font-family:Arial,sans-serif}#bar{padding:10px 14px;background:#fff;border-bottom:1px solid #bbb}#view{width:100vw;height:calc(100vh - 112px);display:block}#msg{font-size:13px;color:#39443f}input{vertical-align:middle}#frame{width:55vw}</style></head><body>
<div id="bar"><b>people1 · SMPL-H + WiLoR</b>　帧 <span id="fi">0</span>/<span id="fn"></span>　<input id="frame" type="range" min="0" value="0"> <button id="play">播放</button><br><span id="msg">固定地面坐标；真实 6890 顶点 / 13776 三角面；手部姿态来自保存的 SMPL-H 参数。</span></div><canvas id="view"></canvas>
<script id="payload" type="application/json">PAYLOAD</script><script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script><script>
const P=JSON.parse(document.getElementById('payload').textContent); const dec=s=>{let b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Uint8Array(new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer())};
const f32=s=>new Float32Array(dec(s).buffer), u32=s=>new Uint32Array(dec(s).buffer), u8=s=>new Uint8Array(dec(s).buffer); const V=f32(P.vertices),F=u32(P.faces),J=f32(P.joints),C=f32(P.pred),T=f32(P.raw),A=u8(P.accepted); let frame=0;
const scene=new THREE.Scene();scene.background=new THREE.Color(0xf4f6f5);const cam=new THREE.PerspectiveCamera(42,innerWidth/(innerHeight-112),.01,100);cam.position.set(2.7,-3.8,2.1);cam.up.set(0,0,1);const ren=new THREE.WebGLRenderer({canvas:document.getElementById('view'),antialias:true});ren.setPixelRatio(devicePixelRatio);ren.setSize(innerWidth,innerHeight-112);scene.add(new THREE.HemisphereLight(0xffffff,0x6f7b76,2));const dl=new THREE.DirectionalLight(0xffffff,2);dl.position.set(2,-2,5);scene.add(dl);
const ground=new THREE.Mesh(new THREE.PlaneGeometry(8,8),new THREE.MeshStandardMaterial({color:0x718279,side:THREE.DoubleSide}));scene.add(ground);scene.add(new THREE.GridHelper(8,16,0xd8dedb,0xaeb9b3));
function axis(a,b,c){let g=new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a),new THREE.Vector3(...b)]);scene.add(new THREE.Line(g,new THREE.LineBasicMaterial({color:c})));} axis([0,0,0],[1,0,0],0xb22);axis([0,0,0],[0,1,0],0x2a5);axis([0,0,0],[0,0,1],0x25a);
const geo=new THREE.BufferGeometry(); const pos=new Float32Array(P.verts*3); geo.setAttribute('position',new THREE.BufferAttribute(pos,3));geo.setIndex(new THREE.BufferAttribute(F,1));const mesh=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({color:0xc8896b,transparent:true,opacity:.62,side:THREE.DoubleSide,depthWrite:false}));scene.add(mesh);
const handMat=new THREE.MeshBasicMaterial({color:0x8b2be2});const handGeo=new THREE.SphereGeometry(.018,8,6);const hands=[];for(let i=0;i<P.joints_n;i++){let m=new THREE.Mesh(handGeo,handMat);scene.add(m);hands.push(m)}
const cocoGeo=new THREE.BufferGeometry();const cocoPos=new Float32Array(17*3);cocoGeo.setAttribute('position',new THREE.BufferAttribute(cocoPos,3));const coco=new THREE.Points(cocoGeo,new THREE.PointsMaterial({color:0x0b6e4f,size:.035}));scene.add(coco);
function setFrame(n){frame=n;let vo=n*P.verts*3;pos.set(V.subarray(vo,vo+P.verts*3));geo.attributes.position.needsUpdate=true;geo.computeVertexNormals();for(let i=0;i<P.joints_n;i++)hands[i].position.fromArray(J,n*P.joints_n*3+i*3);for(let i=0;i<17;i++)cocoPos.set(C.subarray(n*17*3+i*3,n*17*3+i*3+3),i*3);cocoGeo.attributes.position.needsUpdate=true;document.getElementById('fi').textContent=n;document.getElementById('frame').value=n}
document.getElementById('fn').textContent=P.frames-1;document.getElementById('frame').max=P.frames-1;document.getElementById('frame').oninput=e=>setFrame(+e.target.value);let playing=false;document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'暂停':'播放'};function loop(){if(playing)setFrame((frame+1)%P.frames);ren.render(scene,cam);requestAnimationFrame(loop)}setFrame(0);loop();addEventListener('resize',()=>{cam.aspect=innerWidth/(innerHeight-112);cam.updateProjectionMatrix();ren.setSize(innerWidth,innerHeight-112)});
</script></body></html>'''.replace("PAYLOAD", data)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(html, encoding="utf-8")
    print(json.dumps({"output":str(args.output),"frames":len(vertices),"vertices":vertices.shape[1],"faces":len(faces),"joints":joints_g.shape[1]}, ensure_ascii=False))
    return 0
if __name__ == '__main__': raise SystemExit(main())
