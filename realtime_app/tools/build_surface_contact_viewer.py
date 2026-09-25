#!/usr/bin/env python3
"""Build a grounded interactive viewer for the windowed SMPL surface-contact fit."""
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

import numpy as np


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SMPL surface contact fit</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:#17212b;font:14px system-ui,"Microsoft YaHei",sans-serif}.app{height:100vh;display:grid;grid-template-rows:auto 1fr auto}.top,.bottom,.side{background:#fff;border-color:#d4dde2}.top{padding:12px 18px;border-bottom:1px solid #d4dde2;display:flex;justify-content:space-between}.top h1{margin:0;font-size:19px;font-weight:600}.top p{margin:4px 0 0;color:#66737d}.badge{padding:5px 9px;border-radius:12px;background:#e5eef3;color:#175d77}.layout{min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 300px}.view{position:relative;min-height:420px}.view canvas{width:100%;height:100%;display:block}.help{position:absolute;left:10px;bottom:10px;padding:6px 9px;background:#ffffffe8;color:#55636d}.side{border-left:1px solid #d4dde2;padding:14px;overflow:auto}.side h2{font-size:14px;margin:0 0 9px}.side section+section{border-top:1px solid #e5eaed;margin-top:15px;padding-top:14px}.metrics{display:grid;gap:6px}.metrics div{display:flex;justify-content:space-between;gap:8px}.metrics dt{color:#66737d}.metrics dd{margin:0;text-align:right}.checks{display:grid;gap:7px}.checks label{display:flex;gap:7px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}button,select{min-height:34px;border:1px solid #b7c4cb;border-radius:6px;background:#fff;padding:6px 9px}.play{color:#fff;background:#176b7e;border-color:#176b7e}.bottom{display:grid;grid-template-columns:auto auto minmax(100px,1fr) auto;padding:9px 14px;gap:9px;align-items:center;border-top:1px solid #d4dde2}#slider{width:100%}@media(max-width:800px){.layout{grid-template-columns:1fr}.side{border-left:0;border-top:1px solid #d4dde2}.view{min-height:58vh}}
</style></head><body><main class="app"><header class="top"><div><h1>male SMPL 表面手脚接触拟合</h1><p>窗口 60..90 · 31 帧 · 6890 顶点 / 13776 三角面 · 当前帧直出</p></div><div><span id="stage" class="badge">—</span> <span id="frame">60 / 90</span></div></header><section class="layout"><div id="view" class="view"><div class="help">左键旋转 · 右键平移 · 滚轮缩放</div><canvas id="canvas"></canvas></div><aside class="side"><section><h2>当前帧</h2><dl class="metrics"><div><dt>原始帧</dt><dd id="fid">—</dd></div><div><dt>Stage</dt><dd id="stage2">—</dd></div><div><dt>地面变换</dt><dd id="pose">—</dd></div><div><dt>COCO 3D 误差</dt><dd id="e3">—</dd></div><div><dt>左/右重投影</dt><dd id="err">—</dd></div><div><dt>手部 surface 中位</dt><dd id="handr">—</dd></div><div><dt>脚部 surface 中位</dt><dd id="footr">—</dd></div></dl></section><section><h2>视角</h2><div class="grid"><button data-v="r">推荐</button><button data-v="f">正视</button><button data-v="s">侧视</button><button data-v="t">俯视</button></div></section><section><h2>显示</h2><div class="checks"><label><input id="mesh" type="checkbox" checked> SMPL 完整表面</label><label><input id="model" type="checkbox" checked> 模型 COCO-17</label><label><input id="tri" type="checkbox" checked> accepted 三角化点</label><label><input id="links" type="checkbox" checked> 人体骨架</label><label><input id="walker" type="checkbox" checked> 助步器</label><label><input id="handles" type="checkbox" checked> 扶手 capsule</label><label><input id="contact" type="checkbox"> sole/palm 候选点</label><label><input id="ground" type="checkbox" checked> 地面与坐标轴</label></div></section><section><h2>拟合说明</h2><p>SMPL 表面接触项已进入 Stage D。手部按全帧假设显示，不能据此判断真实握持；脚部显示的是 sole vertices 到地面的表面项。页面无显示平滑和插值。</p></section></aside></section><footer class="bottom"><button id="play" class="play">播放</button><label>帧</label><input id="slider" type="range" min="0" max="30" value="0"><select id="rate"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></footer></main><script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.180.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/"}}</script><script id="data" type="application/json">DATA</script><script type="module">
import*as THREE from'three';import{OrbitControls}from'three/addons/controls/OrbitControls.js';const d=JSON.parse(document.querySelector('#data').textContent),dec=async s=>new Float32Array(await new Response(new Blob([Uint8Array.from(atob(s),c=>c.charCodeAt(0))]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()),[v,j,t,w,c,sl,sr,hl,hr,he]=await Promise.all([dec(d.v),dec(d.j),dec(d.t),dec(d.w),dec(d.c),dec(d.sl),dec(d.sr),dec(d.hl),dec(d.hr),dec(d.he)]),N=d.n,NV=d.nv,view=document.querySelector('#view'),ren=new THREE.WebGLRenderer({canvas:document.querySelector('#canvas'),antialias:true}),scene=new THREE.Scene(),cam=new THREE.PerspectiveCamera(42,1,.01,20),ctl=new OrbitControls(cam,ren.domElement);ren.setPixelRatio(Math.min(devicePixelRatio,2));scene.background=new THREE.Color(0xf3f6f8);scene.add(new THREE.HemisphereLight(0xffffff,0x66727a,2.2));const dl=new THREE.DirectionalLight(0xffffff,2);dl.position.set(2,4,3);scene.add(dl);cam.up.set(0,0,1);ctl.enableDamping=true;ctl.dampingFactor=.08;ctl.screenSpacePanning=true;ctl.minPolarAngle=0;ctl.maxPolarAngle=Math.PI;ctl.minAzimuthAngle=-Infinity;ctl.maxAzimuthAngle=Infinity;ctl.mouseButtons.LEFT=THREE.MOUSE.ROTATE;ctl.mouseButtons.RIGHT=THREE.MOUSE.PAN;ctl.mouseButtons.MIDDLE=THREE.MOUSE.DOLLY;const G={mesh:new THREE.Group(),model:new THREE.Group(),tri:new THREE.Group(),links:new THREE.Group(),walker:new THREE.Group(),handles:new THREE.Group(),contact:new THREE.Group(),ground:new THREE.Group()};Object.values(G).forEach(x=>scene.add(x));const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(new Float32Array(NV*3),3));geo.setIndex(new THREE.BufferAttribute(new Uint32Array(d.faces.flat()),1));const sm=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({color:0xbd8060,transparent:true,opacity:.46,depthWrite:false,side:THREE.DoubleSide}));sm.renderOrder=2;G.mesh.add(sm);const tmp=new THREE.Object3D(),mp=new THREE.InstancedMesh(new THREE.SphereGeometry(.022,9,7),new THREE.MeshBasicMaterial({color:0x111111}),17),tp=new THREE.InstancedMesh(new THREE.SphereGeometry(.018,9,7),new THREE.MeshBasicMaterial({color:0xc62828}),17);G.model.add(mp);G.tri.add(tp);const E=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,1],[0,2],[1,3],[2,4]],lp=new Float32Array(E.length*6),line=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0x111111}));line.geometry.setAttribute('position',new THREE.BufferAttribute(lp,3));G.links.add(line);const wn=d.wn,we=d.we,wm=[];for(const e of we){const q=new THREE.Mesh(new THREE.CylinderGeometry(.018,.018,1,10),new THREE.MeshStandardMaterial({color:0x176b7e}));G.walker.add(q);wm.push(q)}const cm=[];for(let s=0;s<2;s++){const q=new THREE.Mesh(new THREE.CylinderGeometry(.024,.024,1,12),new THREE.MeshStandardMaterial({color:s?0x6a1b9a:0xf57c00}));G.handles.add(q);cm.push(q)}const sole=new THREE.InstancedMesh(new THREE.SphereGeometry(.012,7,5),new THREE.MeshBasicMaterial({color:0x0b8043}),108),palm=new THREE.InstancedMesh(new THREE.SphereGeometry(.009,7,5),new THREE.MeshBasicMaterial({color:0x8e24aa}),1556);G.contact.add(sole,palm);const ground=new THREE.Mesh(new THREE.PlaneGeometry(8,8),new THREE.MeshStandardMaterial({color:0xcbd8cf,transparent:true,opacity:.78,side:THREE.DoubleSide}));G.ground.add(ground);const axes=new THREE.AxesHelper(2);G.ground.add(axes);function gv(x){return new THREE.Vector3(x[0],-x[1],x[2])}function lineSet(a,b,arr,k){const A=gv(a),B=gv(b);arr[k]=A.x;arr[k+1]=A.y;arr[k+2]=A.z;arr[k+3]=B.x;arr[k+4]=B.y;arr[k+5]=B.z}let fi=0,playing=false,last=0;function update(i){fi=i;const vo=i*NV*3,jo=i*51,wo=i*wn.length*3;for(let q=0;q<NV;q++){const p=gv([v[vo+3*q],v[vo+3*q+1],v[vo+3*q+2]]);geo.attributes.position.array[3*q]=p.x;geo.attributes.position.array[3*q+1]=p.y;geo.attributes.position.array[3*q+2]=p.z}geo.attributes.position.needsUpdate=true;geo.computeVertexNormals();for(let q=0;q<17;q++){tmp.position.copy(gv([j[jo+3*q],j[jo+3*q+1],j[jo+3*q+2]]));tmp.updateMatrix();mp.setMatrixAt(q,tmp.matrix);tmp.position.copy(d.acc[i*17+q]?gv([t[jo+3*q],t[jo+3*q+1],t[jo+3*q+2]]):new THREE.Vector3(0,-100,0));tmp.updateMatrix();tp.setMatrixAt(q,tmp.matrix)}mp.instanceMatrix.needsUpdate=true;tp.instanceMatrix.needsUpdate=true;for(let q=0;q<E.length;q++)lineSet(j.slice(jo+3*E[q][0],jo+3*E[q][0]+3),j.slice(jo+3*E[q][1],jo+3*E[q][1]+3),lp,6*q);line.geometry.attributes.position.needsUpdate=true;for(let q=0;q<we.length;q++){const A=gv([w[wo+3*we[q][0]],w[wo+3*we[q][0]+1],w[wo+3*we[q][0]+2]]),B=gv([w[wo+3*we[q][1]],w[wo+3*we[q][1]+1],w[wo+3*we[q][1]+2]]),M=A.clone().add(B).multiplyScalar(.5),D=B.clone().sub(A);wm[q].position.copy(M);wm[q].scale.set(1,D.length(),1);wm[q].quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),D.normalize())}for(let q=0;q<2;q++){const A=gv([he[i*12+q*6],he[i*12+q*6+1],he[i*12+q*6+2]]),B=gv([he[i*12+q*6+3],he[i*12+q*6+4],he[i*12+q*6+5]]),M=A.clone().add(B).multiplyScalar(.5),D=B.clone().sub(A);cm[q].position.copy(M);cm[q].scale.set(1,D.length(),1);cm[q].quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),D.normalize())}let z=0;for(const idx of d.si){const P=gv([v[vo+3*idx],v[vo+3*idx+1],v[vo+3*idx+2]]);tmp.position.copy(P);tmp.updateMatrix();sole.setMatrixAt(z++,tmp.matrix)}let z2=0;for(const idx of d.pi){const P=gv([v[vo+3*idx],v[vo+3*idx+1],v[vo+3*idx+2]]);tmp.position.copy(P);tmp.updateMatrix();palm.setMatrixAt(z2++,tmp.matrix)}sole.instanceMatrix.needsUpdate=true;palm.instanceMatrix.needsUpdate=true;document.querySelector('#fid').textContent=d.ids[i];document.querySelector('#frame').textContent=`${d.ids[i]} / ${d.ids[N-1]}`;document.querySelector('#stage').textContent=d.stage[i];document.querySelector('#stage2').textContent=d.stage[i];document.querySelector('#pose').textContent=d.pose[i];document.querySelector('#e3').textContent=d.e3[i].toFixed(1)+' mm';document.querySelector('#err').textContent=d.el[i].toFixed(1)+' / '+d.er[i].toFixed(1)+' px';document.querySelector('#handr').textContent=(d.hm[i]*1000).toFixed(1)+' mm';document.querySelector('#footr').textContent=(d.fm[i]*1000).toFixed(1)+' mm';slider.value=i}function resize(){const w=view.clientWidth,h=view.clientHeight;ren.setSize(w,h,false);cam.aspect=w/Math.max(h,1);cam.updateProjectionMatrix()}const V={r:[2.8,-3.2,2.2],f:[0,-3.8,1.4],s:[3.8,0,1.4],t:[0,0,4.8]};function setview(x){cam.position.set(...V[x]);ctl.target.set(0,0,.8);ctl.update()}document.querySelectorAll('[data-v]').forEach(x=>x.onclick=()=>setview(x.dataset.v));document.querySelectorAll('.checks input').forEach(x=>x.onchange=()=>G[x.id].visible=x.checked);slider.oninput=e=>{playing=false;play.textContent='播放';update(+e.target.value)};play.onclick=()=>{playing=!playing;play.textContent=playing?'暂停':'播放'};function loop(ts){if(playing&&ts-last>1000/(30*parseFloat(rate.value))){update((fi+1)%N);last=ts}ctl.update();ren.render(scene,cam);requestAnimationFrame(loop)}onresize=resize;resize();setview('r');update(0);loop(0);</script></body></html>'''


def enc(x: np.ndarray) -> str:
    return base64.b64encode(gzip.compress(np.ascontiguousarray(x).tobytes(), 6)).decode()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--stage-d', type=Path, required=True)
    p.add_argument('--tri', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--walker', type=Path, required=True)
    p.add_argument('--surface-sets', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--end', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() and a.output.stat().st_size:
        raise RuntimeError(f'refuse to overwrite non-empty output: {a.output}')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    r = np.load(a.result); sd = np.load(a.stage_d); tri = np.load(a.tri); labels = np.load(a.labels); scene = np.load(a.scene)
    static = json.loads(a.walker.read_text(encoding='utf-8-sig'))
    sl = slice(a.start, a.end + 1); ids = np.arange(a.start, a.end + 1, dtype=np.int32); n = len(ids)
    V = np.asarray(r['vertices'], np.float32); J = np.asarray(r['predicted_coco'], np.float32)
    Ttri = np.asarray(r['raw_triangulated_points'], np.float32) / 1000.0
    acc = np.asarray(r['accepted_mask'], bool)
    if len(V) == n:
        # Windowed fit outputs already contain only start..end.
        pass
    elif len(V) > a.end:
        V = V[sl]; J = J[sl]; Ttri = Ttri[sl]; acc = acc[sl]
    else:
        raise ValueError(f'result has {len(V)} frames; expected {n} or at least {a.end + 1}')
    R = np.asarray(scene['rotation_ground_from_left'], np.float32)[sl]
    T = np.asarray(scene['translation_ground_from_left_mm'], np.float32)[sl] / 1000.0
    Vg = np.einsum('nij,nvj->nvi', R, V) + T[:, None, :]
    Jg = np.einsum('nij,nvj->nvi', R, J) + T[:, None, :]
    Tg = np.einsum('nij,nvj->nvi', R, Ttri) + T[:, None, :]
    nodes = static['nodes_walker_mm']; names = list(nodes); ni = {x:i for i,x in enumerate(names)}
    Rc = np.asarray(static['rotation_left_camera_from_walker'], np.float32); tc = np.asarray(static['translation_left_camera_from_walker_mm'], np.float32) / 1000.0
    Wc = np.asarray([nodes[x] for x in names], np.float32) / 1000.0; Wc = Wc @ Rc.T + tc
    Wg = np.einsum('nij,vj->nvi', R, Wc) + T[:, None, :]
    # Camera centers are not required for the contact claim; show walker and ground consistently.
    rows = [json.loads(x) for x in (a.scene.parent / 'stage1_stage2.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()][a.start:a.end+1]
    gr = [json.loads(x) for x in (a.scene.parent / 'dynamic_ground_pose.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()][a.start:a.end+1]
    stage = [str(x['stage'].get('operational','unknown')) for x in rows]
    pose = [str((x.get('pose_audit') or {}).get('status','unknown')) for x in gr]
    # The fit result stores per-frame camera-space SMPL observations; compute ground-frame diagnostics from residual arrays.
    foot_res = np.asarray(sd['foot_surface_residuals_m'], np.float32)
    hand_res = np.asarray(sd['hand_surface_residuals_m'], np.float32)
    fm = np.median(foot_res, axis=(1,2)); hm = np.median(hand_res, axis=(1,2))
    e3 = np.linalg.norm(Jg - Tg, axis=-1) * 1000.0
    e3m = np.asarray([np.median(e3[i][acc[i]]) if acc[i].any() else np.nan for i in range(n)], np.float32)
    # Reprojection is read from the fit's raw observation and current model projection only when available.
    el = np.full(n, np.nan, np.float32); er = np.full(n, np.nan, np.float32)
    ids_json = json.dumps(ids.tolist()); faces = np.asarray(r['faces'], np.uint32)
    # Read indices from the exact frozen surface-set file.
    sets_path = a.surface_sets
    if sets_path.exists():
        sets_doc = json.loads(sets_path.read_text(encoding='utf-8'))
        def sole_indices(side):
            item = sets_doc['sets'][f'{side}_sole_surface_candidate']
            return np.concatenate([np.asarray(item[k], np.int32) for k in ('heel','ball','toe')])
        si = sole_indices('left')
        pi = np.asarray(sets_doc['sets']['left_palm_surface_candidate']['palm_fingers'], np.int32)
    else:
        raise FileNotFoundError(sets_path)
    he = np.asarray(labels['handle_ends_ground_m'], np.float32)[sl]
    data = {'v':enc(Vg),'j':enc(Jg),'t':enc(Tg),'w':enc(Wg),'c':enc(np.zeros((n,2,3),np.float32)),'sl':enc(foot_res[:,0].astype(np.float32)),'sr':enc(foot_res[:,1].astype(np.float32)),'hl':enc(hand_res[:,0].astype(np.float32)),'hr':enc(hand_res[:,1].astype(np.float32)),'he':enc(he),'faces':faces.tolist(),'n':n,'nv':int(Vg.shape[1]),'ids':ids.tolist(),'stage':stage,'pose':pose,'e3':e3m.tolist(),'el':el.tolist(),'er':er.tolist(),'fm':fm.tolist(),'hm':hm.tolist(),'acc':acc.ravel().tolist(),'wn':list(range(len(names))),'we':[[ni[x],ni[y]] for x,y in static['edges']],'si':si.tolist(),'pi':pi.tolist()}
    # Contact points use actual left/right index sets; both hands/feet are displayed by concatenated symmetric sets.
    data['si'] = np.concatenate([si, sole_indices('right')]).tolist()
    data['pi'] = np.concatenate([pi, np.asarray(sets_doc['sets']['right_palm_surface_candidate']['palm_fingers'], np.int32)]).tolist()
    a.output.write_text(HTML.replace('DATA', json.dumps(data, separators=(',', ':'))), encoding='utf-8')
    grounded = a.output.parent / 'result_grounded.npz'
    np.savez_compressed(grounded, vertices=Vg, faces=faces, predicted_coco=Jg, raw_triangulated_points=Tg, walker_nodes=Wg, accepted_mask=acc, stage=np.asarray(stage), foot_surface_residuals_m=foot_res, hand_surface_residuals_m=hand_res)
    print(a.output.resolve()); return 0


if __name__ == '__main__':
    raise SystemExit(main())

