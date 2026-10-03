"""Camera-coordinate diagnostic only; no fabricated ground or walker replay."""
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from check_mainline_regression import fit_arguments
from pose_app.fisheye_camera import load_stereo_fisheye
from pose_app.mainline_regression import observation_snapshot


PAGE = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>身体</title>
<style>body{margin:0;background:#fafafa;font:14px system-ui}header{padding:12px;background:white}canvas{width:100%;height:76vh;display:block}button,input{margin:5px}#frame{min-width:100px;display:inline-block}#slider{width:65%}</style>
<header><b>PMPose · SMPL-H</b>　<span id="frame"></span><br>相机坐标 · 左相机转正视角 · 没有地面或助步器重放。拖动旋转，滚轮缩放。<br>
<label><input id="mesh" type="checkbox" checked>人体</label><label><input id="obs" type="checkbox">三角化观测：通过红色，拒绝橙色</label><label><input id="model" type="checkbox">模型COCO点：青色</label><span id="status"></span></header>
<canvas id="canvas"></canvas><button id="play">播放</button><input id="slider" type="range" min="0" max="MAX" value="0"><span id="error"></span>
<script id="data" type="application/json">DATA</script><script>
try {
const d=JSON.parse(document.getElementById('data').textContent),N=d.n,F=d.faces,NV=d.nv;
function decode(s){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return new Float32Array(a.buffer)}
const V=decode(d.vertices),T=decode(d.observed),C=decode(d.coco),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
let fi=0,yaw=.15,pitch=.1,zoom=1,drag=false,playing=false,lastTime=0;
const center=d.center;
function rot(p){const x=p[1]-center[1],y=p[0]-center[0],z=-(p[2]-center[2]),xx=x*Math.cos(yaw)+z*Math.sin(yaw),zz=-x*Math.sin(yaw)+z*Math.cos(yaw);return[xx,y*Math.cos(pitch)-zz*Math.sin(pitch),y*Math.sin(pitch)+zz*Math.cos(pitch)]}
function draw(){canvas.width=canvas.clientWidth*devicePixelRatio;canvas.height=canvas.clientHeight*devicePixelRatio;const w=canvas.width,h=canvas.height,s=Math.min(w,h)*.75*zoom/d.extent;ctx.fillStyle='#fafafa';ctx.fillRect(0,0,w,h);
function screen(p){const q=rot(p);return[w/2+q[0]*s,h/2-q[1]*s,q[2]]}
const p=[];for(let i=0;i<NV;i++)p.push(screen(V.subarray((fi*NV+i)*3,(fi*NV+i+1)*3)));
if(document.getElementById('mesh').checked){const triangles=F.map(f=>({f,z:(p[f[0]][2]+p[f[1]][2]+p[f[2]][2])/3})).sort((a,b)=>b.z-a.z);for(const {f} of triangles){const a=p[f[0]],b=p[f[1]],c=p[f[2]],u=[b[0]-a[0],b[1]-a[1],(b[2]-a[2])*s],v=[c[0]-a[0],c[1]-a[1],(c[2]-a[2])*s],nx=u[1]*v[2]-u[2]*v[1],ny=u[2]*v[0]-u[0]*v[2],nz=u[0]*v[1]-u[1]*v[0],light=.35+.65*Math.abs((nx*.3+ny*.5+nz*.8)/(Math.hypot(nx,ny,nz)||1)),shade=Math.round(85+110*light);ctx.fillStyle=`rgb(${shade-28},${shade},${shade+12})`;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.lineTo(c[0],c[1]);ctx.closePath();ctx.fill()}}
function dot(a,j,color){const v=a.subarray((fi*17+j)*3,(fi*17+j+1)*3);if(!Array.from(v).every(Number.isFinite))return;const q=screen(v);ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],3*devicePixelRatio,0,Math.PI*2);ctx.fill()}
for(let j=0;j<17;j++){if(document.getElementById('obs').checked)dot(T,j,d.accepted[fi][j]?'#dd3030':'#df9b00');if(document.getElementById('model').checked)dot(C,j,'#00b5b5')}
const cams=d.cameras.map(screen);ctx.strokeStyle='#444';ctx.beginPath();ctx.moveTo(cams[0][0],cams[0][1]);ctx.lineTo(cams[1][0],cams[1][1]);ctx.stroke();ctx.font='12px system-ui';cams.forEach((c,i)=>{ctx.fillStyle='#333';ctx.fillRect(c[0]-3,c[1]-3,6,6);ctx.fillText(i?'右相机':'左相机',c[0]+6,c[1])});
document.getElementById('frame').textContent=`${fi+1}/${N} · 原帧 ${d.original_pair_ids[fi]}`;document.getElementById('status').textContent=`拟合观测 ${d.accepted[fi].filter(Boolean).length}/17`;document.getElementById('slider').value=fi;canvas.dataset.renderedFrame=fi;canvas.dataset.triangles=F.length}
document.getElementById('slider').oninput=e=>{playing=false;fi=+e.target.value;document.getElementById('play').textContent='播放';draw()};document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'暂停':'播放'};
canvas.onpointerdown=e=>{drag=true;canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=()=>drag=false;canvas.onpointermove=e=>{if(drag){yaw+=e.movementX*.008;pitch+=e.movementY*.008;draw()}};canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.2,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw()};['mesh','obs','model'].forEach(x=>document.getElementById(x).onchange=draw);onresize=draw;
function loop(t){if(playing&&t-lastTime>=1000/30){fi=(fi+1)%N;lastTime=t;draw()}requestAnimationFrame(loop)}draw();requestAnimationFrame(loop);
}catch(e){document.getElementById('error').textContent=e.stack;throw e}
</script></html>'''


def encode(array):
    return base64.b64encode(np.ascontiguousarray(array, dtype='<f4').tobytes()).decode('ascii')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    output = run/'body.html'
    if output.exists():
        raise FileExistsError(output)
    options = fit_arguments(run/'pmpose/fit_command.json')
    observed = observation_snapshot(options)
    cal = load_stereo_fisheye(options.calibration_dir)
    with np.load(options.output_dir/'result.npz') as data:
        z = {key: data[key] for key in data.files}
    n = len(z['pair_id'])
    assert np.array_equal(z['pair_id'], observed['pair_id'])
    assert z['vertices'].shape == (n, 6890, 3)
    assert z['faces'].shape == (13776, 3) and z['faces'].min() >= 0 and z['faces'].max() < 6890
    assert np.array_equal(z['body_accepted'], observed['accepted'])
    assert np.allclose(z['raw_triangulated_points'], observed['triangulated_mm'], equal_nan=True)
    assert np.isfinite(z['vertices']).all()
    frames = json.loads((run/'frames.json').read_text(encoding='utf-8'))
    assert len(frames) == n
    v = z['vertices']
    center = ((v.min((0, 1))+v.max((0, 1)))/2).tolist()
    payload = {'n': n, 'nv': 6890, 'faces': z['faces'].astype(int).tolist(),
               'vertices': encode(v), 'coco': encode(z['predicted_coco']),
               'observed': encode(observed['triangulated_mm']/1000),
               'accepted': observed['accepted'].tolist(), 'center': center,
               'extent': float(np.ptp(v, axis=(0, 1)).max()),
               'original_pair_ids': [f['pair_id'] for f in frames],
               'cameras': [[0, 0, 0], (-cal.R_cam0_to_cam1.T @ (cal.T_cam0_to_cam1_mm/1000)).tolist()]}
    output.write_text(PAGE.replace('MAX', str(n-1)).replace('DATA', json.dumps(payload, separators=(',', ':'))), encoding='utf-8')
    audit = {'source': str(options.output_dir/'result.npz'), 'frames': n, 'vertices': 6890,
             'faces': 13776, 'coordinate_frame': 'camera0_m', 'fixed_display_center': center,
             'fitter_observations_recomputed': True, 'stage_ground_walker': 'not_available_not_rendered',
             'interpolation': False, 'smoothing': False, 'prior_body_result_consumed': False}
    (run/'body_viewer_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
