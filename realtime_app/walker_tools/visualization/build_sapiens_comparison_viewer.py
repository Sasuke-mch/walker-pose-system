"""Visualize all paired detector outputs without fitting or display smoothing."""
from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
from walker_tools._compat import APP_ROOT as _tool_app_root
_tool_prepare_imports()

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

APP = _tool_app_root
from pose_app.rotation import raw_to_model_point
from pose_app.calibration import StereoCalibration
from walker_tools.visualization.visualize_offline_stereo_models import EDGES, load_records


PAGE = r'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>Sapiens2 全448帧检测对照</title><style>
body{margin:18px;background:#f6f7f8;color:#202b30;font:15px system-ui}h1{font-size:23px}button,select,input{margin:5px}button{padding:7px 12px}header{position:sticky;top:0;background:#f6f7f8;z-index:2;padding:6px;border-bottom:1px solid #bbb}#slider{width:45vw}.panels{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.panel{background:white;padding:8px;border:1px solid #bbb}canvas{display:block;width:100%;background:white}h3{font-size:15px;margin:4px}#space{height:520px;touch-action:none}section{display:grid;grid-template-columns:1.3fr 1fr;gap:14px;margin-top:16px}table{border-collapse:collapse;background:white;width:100%;font-size:13px}td,th{border:1px solid black;padding:5px;text-align:left}tr.active{font-weight:bold}#diag{white-space:pre-wrap;font-size:13px}small{color:#45525b}@media(max-width:900px){.panels{grid-template-columns:repeat(2,1fr)}section{display:block}}
</style><h1>Sapiens2 / PMPose：全部448帧同源检测与三角化</h1>
<p>黑色实线＝原始COCO-17骨架；绿色实心点＝双目几何门通过；橙色叉＝双目门拒绝，不能单独认定二维定位错误。三维仅显示有限且双侧正深度点。本页为观测对照，无SMPL拟合表面。</p>
<header><button id="play">播放</button><button id="prev">上一帧</button><button id="next">下一帧</button><input id="slider" type="range" min="0" max="447" value="0"><input id="frame" type="number" min="0" max="447" value="0" style="width:65px"><select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×（30 FPS）</option><option value="2">2×</option></select><br><label><input type="checkbox" id="dense">显示Sapiens原生308点（小蓝点，包含低分点）</label><label><input type="checkbox" id="labels" checked>关节编号</label><label>二维放大 <input id="zoom2" type="range" min="1" max="5" step="0.1" value="1"></label><small>点击下表选择关节，放大围绕该关节；编号9/10腕、13/14膝、15/16踝。</small></header>
<div class="panels"><div class="panel"><h3>Sapiens2 · 左相机</h3><canvas id="s0" width="432" height="768"></canvas></div><div class="panel"><h3>Sapiens2 · 右相机</h3><canvas id="s1" width="432" height="768"></canvas></div><div class="panel"><h3>PMPose · 左相机</h3><canvas id="p0" width="432" height="768"></canvas></div><div class="panel"><h3>PMPose · 右相机</h3><canvas id="p1" width="432" height="768"></canvas></div></div>
<section><div class="panel"><h3>本次重放的固定地面坐标 · 三维原始骨架</h3><select id="model"><option value="sapiens2">Sapiens2</option><option value="pmpose">PMPose</option></select><button id="front">正视</button><button id="side">侧视</button><button id="reset">复位</button><small>左拖旋转，右拖/Shift拖平移，滚轮缩放。无跨帧插值。</small><canvas id="space"></canvas><div id="diag"></div></div><div><h3>当前帧逐点质量（左右平均重投影，px）</h3><table><thead><tr><th>关节</th><th>Sapiens2</th><th>PMPose</th></tr></thead><tbody id="quality"></tbody></table><p>两模型分别使用自己的本次Stage/地面轨迹。地面、助步器和脚踝为工程候选，不能据图认定真实触地、支撑或定位精度。</p></div></section>
<script>const D=DATA,E=EDGES;let i=0,joint=9,playing=false,last=0,az=-.5,el=.18,zoom=1,pan=[0,0],loaded=-1;
const $=id=>document.getElementById(id),images=[new Image(),new Image()];let loadToken=0;
function marker(c,p,valid,r=5){c.strokeStyle=valid?'#087f35':'#d66b00';c.fillStyle=c.strokeStyle;c.lineWidth=2;if(valid){c.beginPath();c.arc(...p,r,0,7);c.fill()}else{c.beginPath();c.moveTo(p[0]-r,p[1]-r);c.lineTo(p[0]+r,p[1]+r);c.moveTo(p[0]-r,p[1]+r);c.lineTo(p[0]+r,p[1]-r);c.stroke()}}
function paint2(){if(loaded!==i)return;for(const [m,prefix] of [['sapiens2','s'],['pmpose','p']])for(let s=0;s<2;s++){const cv=$(prefix+s),c=cv.getContext('2d'),f=D.models[m][i],pts=f.xy[s],z=Number($('zoom2').value),focus=pts[joint]||[540,960];const cx=z>1?focus[0]*.4:216,cy=z>1?focus[1]*.4:384, map=p=>[(p[0]*.4-cx)*z+216,(p[1]*.4-cy)*z+384];c.clearRect(0,0,432,768);c.drawImage(images[s],216-cx*z,384-cy*z,432*z,768*z);if(m==='sapiens2'&&$('dense').checked){c.fillStyle='#0088ba';for(const p of D.dense[s][i]){const q=map(p);c.globalAlpha=.25+.65*Math.min(1,Math.max(0,p[2]));c.beginPath();c.arc(...q,1.8,0,7);c.fill()}c.globalAlpha=1}for(const [a,b]of E)if(pts[a]&&pts[b]){const x=map(pts[a]),y=map(pts[b]);c.strokeStyle='white';c.lineWidth=4;c.beginPath();c.moveTo(...x);c.lineTo(...y);c.stroke();c.strokeStyle='black';c.lineWidth=2;c.stroke()}pts.forEach((p,k)=>{if(!p)return;const q=map(p);marker(c,q,f.valid[k],k===joint?7:4);if($('labels').checked){c.font='bold 12px sans-serif';c.lineWidth=3;c.strokeStyle='white';c.strokeText(k,q[0]+5,q[1]-5);c.fillStyle='black';c.fillText(k,q[0]+5,q[1]-5)}})}}
const center=D.center;function project(p){const x=p[0]-center[0],y=-p[1]-center[1],z=p[2]-.75,cy=Math.cos(az),sy=Math.sin(az),ce=Math.cos(el),se=Math.sin(el),a=x*cy-y*sy,b=x*sy+y*cy;const cv=$('space'),scale=Math.min(cv.width,cv.height)*.30*zoom;return[cv.width/2+a*scale+pan[0],cv.height/2-(z*ce+b*se)*scale+pan[1]]}
function draw3(){const cv=$('space');cv.width=cv.clientWidth;cv.height=520;const c=cv.getContext('2d'),m=$('model').value,f=D.models[m][i];function line(a,b,col,w){a=project(a);b=project(b);c.beginPath();c.moveTo(...a);c.lineTo(...b);c.strokeStyle=col;c.lineWidth=w;c.stroke()}c.fillStyle='#f7f8f9';c.fillRect(0,0,cv.width,cv.height);const lim=D.limits,corner=[[lim[0],lim[2],0],[lim[1],lim[2],0],[lim[1],lim[3],0],[lim[0],lim[3],0]].map(project);c.beginPath();corner.forEach((p,k)=>k?c.lineTo(...p):c.moveTo(...p));c.closePath();c.fillStyle='#718279';c.fill();for(let x=Math.ceil(lim[0]*2)/2;x<=lim[1];x+=.5)line([x,lim[2],0],[x,lim[3],0],'#acb9b1',1);for(let y=Math.ceil(lim[2]*2)/2;y<=lim[3];y+=.5)line([lim[0],y,0],[lim[1],y,0],'#acb9b1',1);
for(const [axis,col]of [[0,'#b52b31'],[1,'#146837'],[2,'#2855a5']]){let p=[0,0,0];p[axis]=axis===2?2:3;line([0,0,0],p,col,2);const q=project(p);c.fillStyle=col;c.fillText(['X 3m','Y 3m','Z 2m'][axis],q[0],q[1]);if(axis<2)for(let v=.5;v<=3;v+=.5){let t=[0,0,0];t[axis]=v;const a=project(t);c.fillText(v+'m',a[0],a[1])}}
for(let side=0;side<2;side++){let prev=null;for(let k=Math.max(0,i-30);k<=i;k++){const r=D.models[m][k],p=r.xyz[15+side];if(p&&r.valid[15+side]){const a=[p[0],p[1],0];if(prev)line(prev,a,side?'#326a89':'#573d91',2);prev=a}else prev=null}let old=null;for(let k=0;k<=i;k++){const p=D.models[m][k].cameras[side],a=[lim[0],p[1],p[2]];if(old)line(old,a,side?'#326a89':'#573d91',1);old=a}const p=f.cameras[side];line(p,[lim[0],p[1],p[2]],'#758590',1)}
for(const [a,b]of D.we){line(f.walker[a],f.walker[b],'#424c53',10);line(f.walker[a],f.walker[b],'#c6cbd0',6)}for(const p of f.walker){const q=project(p);c.beginPath();c.arc(...q,3,0,7);c.fillStyle='#c6cbd0';c.fill()}line(...f.cameras,'#236191',2);for(const p of f.cameras){const q=project(p);c.fillStyle='#236191';c.fillRect(q[0]-5,q[1]-4,10,8)}for(const [a,b]of E)if(f.xyz[a]&&f.xyz[b])line(f.xyz[a],f.xyz[b],'black',2);f.xyz.forEach((p,k)=>{if(p){const q=project(p);marker(c,q,f.valid[k],k===joint?7:4);c.fillStyle='black';c.fillText(k,q[0]+5,q[1]-5)}});$('diag').textContent=`帧 ${i}/447 · ${m}\nStage: ${f.stage}\n地面变换状态: ${f.pose}\n严格通过: ${f.valid.filter(Boolean).length}/17；未插值、未平滑\n当前关节 ${joint}: ${D.names[joint]} · ${f.reason[joint]||'accepted'}\n${f.source}`}
function table(){let h='';for(let k=0;k<17;k++){const cells=['sapiens2','pmpose'].map(m=>{const f=D.models[m][i],e=f.err[k];return `<td>${f.valid[k]?'✓':'×'} ${e===null?'—':e.toFixed(2)}<br>${f.reason[k]||'accepted'}</td>`});h+=`<tr data-j="${k}" class="${k===joint?'active':''}"><td>${k} ${D.names[k]}</td>${cells.join('')}</tr>`}$('quality').innerHTML=h;for(const tr of $('quality').rows)tr.onclick=()=>{joint=Number(tr.dataset.j);table();paint2();draw3()}}
function setFrame(v){i=Math.max(0,Math.min(447,Number.isFinite(Number(v))?Math.round(Number(v)):0));$('slider').value=$('frame').value=i;table();draw3();for(const id of ['s0','s1','p0','p1']){const c=$(id).getContext('2d');c.clearRect(0,0,432,768);c.fillStyle='black';c.fillText('加载当前帧 '+i,12,24)}const token=++loadToken;let count=0;for(let s=0;s<2;s++){images[s].onload=()=>{if(token!==loadToken)return;if(++count===2){loaded=i;paint2()}};images[s].src=`images/${s?'right':'left'}_${String(i).padStart(4,'0')}.jpg`}}
$('slider').oninput=e=>setFrame(e.target.value);$('frame').oninput=e=>setFrame(e.target.value);$('prev').onclick=()=>setFrame(i-1);$('next').onclick=()=>setFrame(i+1);$('play').onclick=()=>{playing=!playing;last=performance.now();$('play').textContent=playing?'暂停':'播放'};function tick(t){if(playing&&t-last>=1000/(30*Number($('speed').value))){setFrame((i+1)%448);last=t}requestAnimationFrame(tick)}requestAnimationFrame(tick);for(const id of ['dense','labels','zoom2'])$(id).oninput=paint2;$('model').onchange=draw3;$('front').onclick=()=>{az=0;el=0;draw3()};$('side').onclick=()=>{az=Math.PI/2;el=0;draw3()};$('reset').onclick=()=>{az=-.5;el=.18;zoom=1;pan=[0,0];draw3()};let drag=null;const cv=$('space');cv.oncontextmenu=e=>e.preventDefault();cv.onpointerdown=e=>{drag=[e.clientX,e.clientY,e.button===2||e.shiftKey];cv.setPointerCapture(e.pointerId)};cv.onpointermove=e=>{if(!drag)return;let dx=e.clientX-drag[0],dy=e.clientY-drag[1];if(drag[2]){pan[0]+=dx;pan[1]+=dy}else{az+=dx*.008;el-=dy*.008}drag[0]=e.clientX;drag[1]=e.clientY;draw3()};cv.onpointerup=()=>drag=null;cv.onpointercancel=()=>drag=null;cv.onwheel=e=>{e.preventDefault();zoom=Math.max(.15,Math.min(8,zoom*Math.exp(-e.deltaY*.001)));draw3()};window.onresize=draw3;setFrame(0);
</script></html>'''


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    protocol = read_json(run/'protocol.json')
    records = {m:load_records(run/m/'stereo_results.jsonl') for m in ('sapiens2','pmpose')}
    if any(len(r) != 448 for r in records.values()):
        raise ValueError('expected the complete audited 448-pair run')
    cal = StereoCalibration.load(protocol['calibration'])
    right_center = -cal.R.T @ cal.T.reshape(3)
    walker = read_json(Path(protocol['walker_model']))
    names = sorted(walker['nodes_walker_mm'])
    nodes = np.asarray([walker['nodes_walker_mm'][k] for k in names])
    static = nodes @ np.asarray(walker['rotation_left_camera_from_walker']).T + walker['translation_left_camera_from_walker_mm']
    data = {'models':{},'dense':[], 'we':[[names.index(a),names.index(b)] for a,b in walker['edges']]}
    all_xyz = []
    for model, rows in records.items():
        scene = np.load(run/model/'scene/scene_transforms.npz',allow_pickle=True)
        ground = [json.loads(x) for x in (run/model/'scene/dynamic_ground_pose.jsonl').read_text(encoding='utf-8').splitlines()]
        frames = []
        for i, row in enumerate(rows):
            if ground[i]['pair_id'] != row['pair_id'] or row['file_name'] != records['sapiens2'][i]['file_name']:
                raise ValueError('same-run frame alignment mismatch')
            R,t = scene['rotation_ground_from_left'][i],scene['translation_ground_from_left_mm'][i]
            xf = lambda p: ((np.asarray(p) @ R.T + t)/1000).tolist()
            pts = row['persons_3d'][0]['keypoints_3d'] if row['persons_3d'] else []
            if pts and len(pts) != 17:
                raise ValueError('COCO17 topology mismatch')
            xyz, valid, reasons, errors = [],[],[],[]
            for j in range(17):
                p = pts[j] if pts else {}
                v = p.get('xyz')
                visible = v is not None and np.isfinite(v).all() and (p.get('depth_left') or 0)>0 and (p.get('depth_right') or 0)>0
                xyz.append(xf(v) if visible else None)
                valid.append(bool(p.get('valid',False)))
                reasons.append(p.get('reason') or (None if valid[-1] else 'no_stereo_person'))
                errors.append(p.get('reprojection_error_mean_px'))
            xy = []
            for side in ('left','right'):
                people = row[side]['persons']
                points = people[0]['keypoints'][:17] if people else []
                xy.append([list(raw_to_model_point(*p[:2],1920,1080,protocol['rotations'][side]))+[p[2]] for p in points])
            frames.append(dict(xyz=xyz,valid=valid,reason=reasons,err=errors,xy=xy,walker=xf(static),
                cameras=xf([np.zeros(3),right_center]),stage=str(scene['stage'][i]),pose=ground[i].get('pose_audit',{}).get('status','unknown'),
                source='严格点门：双侧0.25 / 10px；地面Stage使用既有场景规则。'))
            all_xyz.extend(p for p in xyz if p is not None)
        data['models'][model] = frames
    data['names'] = [p['name'] for p in records['sapiens2'][0]['persons_3d'][0]['keypoints_3d']]
    for side in ('left','right'):
        raw = [json.loads(x) for x in (run/f'{side}_sapiens_raw.jsonl').read_text().splitlines()]
        if [r['pair_id'] for r in raw] != list(range(448)):
            raise ValueError('native 308 frame mismatch')
        data['dense'].append([np.column_stack((r['instances'][0]['keypoints308'],r['instances'][0]['keypoint_scores'])).tolist() for r in raw])
    cloud = np.asarray(all_xyz)
    data['center'] = [float(np.median(cloud[:,0])),float(-np.median(cloud[:,1]))]
    data['limits'] = [float(min(-1,cloud[:,0].min()-.5)),float(max(1,cloud[:,0].max()+.5)),float(min(-1,cloud[:,1].min()-.5)),float(max(1,cloud[:,1].max()+.5))]
    out.mkdir();(out/'images').mkdir()
    video = cv2.VideoWriter(str(out/'comparison_448.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1728,824))
    if not video.isOpened():
        raise RuntimeError('MP4 writer unavailable')
    try:
        for i,row in enumerate(records['sapiens2']):
            images = []
            for side,folder in [('left','left_ccw90'),('right','right_cw90')]:
                image = cv2.imread(str(Path(protocol['input_dir'])/folder/row['file_name']))
                if image is None or image.shape[:2] != (1920,1080):
                    raise ValueError('missing or mismatched upright source image')
                image = cv2.resize(image,(432,768),interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out/'images'/f'{side}_{i:04d}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,88])
                images.append(image)
            panels = []
            for m in ('sapiens2','pmpose'):
                f = data['models'][m][i]
                for s in range(2):
                    im = images[s].copy();pts = f['xy'][s]
                    xy = [tuple(np.rint(np.asarray(p[:2])*.4).astype(int)) for p in pts]
                    for a,b in EDGES:
                        if a<len(xy) and b<len(xy):
                            cv2.line(im,xy[a],xy[b],(255,255,255),4,cv2.LINE_AA)
                            cv2.line(im,xy[a],xy[b],(0,0,0),2,cv2.LINE_AA)
                    for j,p in enumerate(xy):
                        if f['valid'][j]:cv2.circle(im,p,4,(40,175,30),-1,cv2.LINE_AA)
                        else:cv2.drawMarker(im,p,(0,130,240),cv2.MARKER_TILTED_CROSS,10,2,cv2.LINE_AA)
                        cv2.putText(im,str(j),(p[0]+5,p[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),3,cv2.LINE_AA)
                        cv2.putText(im,str(j),(p[0]+5,p[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1,cv2.LINE_AA)
                    panels.append(im)
            frame = np.full((824,1728,3),255,np.uint8);frame[56:]=np.concatenate(panels,axis=1)
            for k,label in enumerate(['Sapiens2 LEFT','Sapiens2 RIGHT','PMPose LEFT','PMPose RIGHT']):
                cv2.putText(frame,f'{label} | pair {i}',(k*432+8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1,cv2.LINE_AA)
                cv2.putText(frame,'green=accepted orange X=rejected',(k*432+8,45),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1,cv2.LINE_AA)
            video.write(frame)
            if i in (0,60,270):cv2.imwrite(str(out/f'preview_{i:04d}.jpg'),frame)
            if i%100==0:print(f'render {i+1}/448',flush=True)
    finally:
        video.release()
    page = PAGE.replace('const D=DATA,E=EDGES;', 'const D='+json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))+',E='+json.dumps(EDGES)+';')
    (out/'index.html').write_text(page,encoding='utf-8')
    manifest = {'input_run':str(run),'frame_ids':list(range(448)),'source_images':protocol['input_dir'],
        'output':str(out),'render_only':True,'no_new_inference_or_fit':True,'display_transform':'[X,-Y,Z]',
        'strict_mask_source':'same-run strict stereo JSONL, not soft fitting mask','finite_rejected_retained':True,
        'video_fps':30,'video_frames':448,'mesh':'unavailable: no Sapiens SMPL fit exists',
        'finite_per_joint':{m:[sum(f['xyz'][j] is not None for f in frames) for j in range(17)] for m,frames in data['models'].items()},
        'accepted_per_joint':{m:[sum(f['valid'][j] for f in frames) for j in range(17)] for m,frames in data['models'].items()}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
