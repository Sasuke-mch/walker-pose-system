"""Replay the same raw scene and display each saved PHC/SMPL-H result directly."""

import argparse
import base64
import csv
import gzip
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from pose_app.calibration import StereoCalibration
from pose_app.fisheye_camera import load_stereo_fisheye, fisheye_project_numpy
from pose_app.triangulation import triangulate_matches
from tools.run_stereo_pva_baseline import _load_predictions
from walker_tools.stereo.evaluate_offline_stereo_predictions import _records, _result
from walker_tools.visualization.build_grasp_body_canvas_viewer import (
    build_canvas_page, encode, recorded_strict_gate,
)

ROOT = Path(__file__).resolve().parents[2]
LABELS = {'source_smplh': '输入 SMPL-H', 'replay448_physx_recovery': 'PHC+ 无腕支撑',
          'replay448_physx_support': 'PHC+ 假设腕支撑'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def finite(values):
    values = np.asarray(values)
    return np.where(np.isfinite(values), values, None).tolist()


def replay(destination, parameters, original_scene):
    """Fresh strict geometry, time mapping and Stage; no old dynamic inputs."""
    destination.mkdir(exist_ok=False)
    inputs = parameters['inputs']
    pairs = Path(inputs['input_pairs'])
    calibration = StereoCalibration.load(Path(inputs['calibration_dir'])/'stereo_fisheye.json')
    records = {side: _records(Path(inputs[side+'_predictions']), 'pmpose', 'raw_keypoint')
               for side in ('left', 'right')}
    with (pairs/'selection_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        selection = list(csv.DictReader(stream))
    with Path(parameters['capture_pairs']).open(encoding='utf-8-sig', newline='') as stream:
        capture = list(csv.DictReader(stream))
    strict, timing = [], []
    if len(selection) != 448 or any(len(records[s]) != 448 for s in records):
        raise ValueError('expected complete448 raw inputs')
    for i, selected in enumerate(selection):
        frame = capture[int(selected['source_pair_index'])]
        time = {'index': i}
        views = {}
        for side, size, rotation, folder in (
                ('left', calibration.left_image_size, 'ccw90', 'left_ccw90'),
                ('right', calibration.right_image_size, 'cw90', 'right_cw90')):
            if selected[side+'_frame_id'] != frame[side+'_frame_id']:
                raise ValueError('capture frame identity differs')
            record = records[side][i]
            name = f'pair_{i:04d}.png'
            if record['id'] != i or record['name'] != name:
                raise ValueError('prediction/image order differs')
            time[side+'_image'] = str((pairs/folder/name).resolve())
            time[side+'_host_return_timestamp_ns'] = int(frame[side+'_host_return_timestamp_ns'])
            views[side] = _result(record, 'pmpose', size, rotation, i/30)
        time['signed_host_delta_ms_right_minus_left'] = (time['right_host_return_timestamp_ns']-time['left_host_return_timestamp_ns'])/1e6
        timing.append(time)
        people = triangulate_matches(views['left'].persons[:1], views['right'].persons[:1],
                                     calibration, .25, .05, 10., max_matches=1)
        strict.append(dict(index=i, original_pair_id=int(frame['pair_id']),
                           persons_3d=[p.to_dict() for p in people]))
    strict_path = destination/'first_instance_strict.jsonl'
    strict_path.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in strict), encoding='utf-8')
    (destination/'frame_times.json').write_text(json.dumps(timing), encoding='utf-8')
    static = read(original_scene/'scene_sources.json')
    capture_dir = Path(parameters['capture_pairs']).parent
    command = [sys.executable, str(ROOT/'realtime_app/walker_tools/scene/replay_fitted_body_scene.py')]
    options = dict(left_video=capture_dir/'left_capture.avi', right_video=capture_dir/'right_capture.avi',
        input_pair_dir=pairs, left_pmpose=inputs['left_predictions'], right_pmpose=inputs['right_predictions'],
        calibration=Path(inputs['calibration_dir'])/'stereo_fisheye.json',
        ground_reference=static['static_ground_reference'], walker_model=static['static_walker_model'],
        output=destination/'scene', strict_jsonl=strict_path, scene_profile='mainline',
        frames_metadata=destination/'frame_times.json')
    for key, value in options.items():
        command += ['--'+key.replace('_','-'), str(value)]
    (destination/'scene_command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
    print('Replaying fresh448 scene', flush=True)
    with (destination/'scene.log').open('w', encoding='utf-8') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def page_for(data, vertices, coco, label):
    page = build_canvas_page(data, vertices, coco, {'cold_start': True, 'detector': 'pmpose'})
    page = page.replace('PMPose', label)
    page = page.replace('腕偏差（左/右）', '仿真状态').replace('观测残差', '参考跟踪均值')
    page = page.replace('参考扶手几何', '腕部外力假设').replace('手脚采样点', '脚底候选点')
    page = page.replace('拟合点：', '当前二维直接监督：')
    page = page.replace("document.getElementById('wrist').textContent=d.wrist_mm[fi].map(x=>x===null?'—':x.toFixed(1)+' mm').join(' / ');",
                        "document.getElementById('wrist').textContent=d.status[fi];")
    page = page.replace('id="handles" type="checkbox" checked', 'id="handles" type="checkbox"')
    page = page.replace('<input id="slider"', '<input id="frame-number" type="number" aria-label="帧号" min="0" max="447" value="0" style="flex:0 0 62px;width:62px"><span id="capture-time"></span><input id="slider"',1)
    page = page.replace('let fi=0,yaw=', "let fi=Math.max(0,Math.min(447,Number(location.hash.slice(1))||0)),yaw=",1)
    page = page.replace('</style>', '.metrics dd{overflow-wrap:anywhere;min-width:0}.side{overflow-x:hidden}</style>',1)
    page = page.replace('<div><dt>地面状态</dt>', '<div><dt>脚跟段内位移 L/R</dt><dd id="heel">—</dd></div><div><dt>腕力最大值</dt><dd id="force">—</dd></div><div><dt>地面状态</dt>')
    note = ('本轮官方冻结权重的三维参考模仿实验；未训练二维反馈。腕支撑是显式外力假设，不能解释为测得的抓握力。'
            '地面、相机、观测骨架和助步器已从同源448图像重新计算；与控制器条件场景核对一致。'
            'Stage含脚踝锚与约0.5秒发布延迟。固定体型来自全片预校准。当前配置：'+data['configuration']+'。脚跟10毫米要求尚未全片通过，当前结果未晋升主线。')
    page = page.replace('<section><h2>参考来源</h2><p>', '<section><h2>实验说明</h2><p>'+note+'</p></section><section><h2>参考来源</h2><p>')
    navigation = ''.join(f'<a href="{name.replace("/","__")}.html">{title}</a>　' for name,title in LABELS.items())
    page = page.replace('<header class="top">', '<header class="top">')
    page = page.replace('<div class="main">', '<nav style="background:white;padding:8px">'+navigation+
        '<a href="'+data['comparisonPlot']+'">全片对比曲线</a>　播放按30帧/秒逐帧，采集时间跨度16.035秒</nav><div class="main">',1)
    page = page.replace('.app{height:100%;', '.app{height:auto;min-height:100%;')
    page = page.replace('grid-template-rows:auto 1fr auto', 'grid-template-rows:auto auto 65vh auto')
    page = page.replace('270px}', '340px}')
    inputs = ('<section style="background:white;padding:12px"><p>左 / 右同源图像：圆圈为检测点，蓝叉为模型投影；图像位置均来自原始鱼眼坐标转换。</p>'
        '<div style="display:flex;justify-content:center;gap:24px"><canvas id="left-img" width="540" height="960" style="width:270px;height:480px"></canvas>'
        '<canvas id="right-img" width="540" height="960" style="width:270px;height:480px"></canvas></div></section>')
    page = page.replace('<script id="data"', inputs+'<script id="data"',1)
    extra = r"""
const images={};let inputVersion=0;
document.getElementById('frame-number').oninput=e=>{fi=Math.max(0,Math.min(447,Math.round(Number(e.target.value)||0)));playing=false;document.getElementById('play').textContent='播放';document.getElementById('slider').value=fi;draw();};
document.querySelectorAll('nav a[href$=".html"]').forEach(a=>a.onclick=()=>{a.href=a.getAttribute('href').split('#')[0]+'#'+fi;});
function drawInputs(){const version=++inputVersion;for(let side=0;side<2;side++){
const name=side?'right':'left',cv=document.getElementById(name+'-img'),ct=cv.getContext('2d');
const path=d.imageRoot+'/'+(side?'right_cw90':'left_ccw90')+'/pair_'+String(fi).padStart(4,'0')+'.png';
if(!images[name]||images[name].path!==path){const image=new Image();images[name]={path,image};image.src=path;}const im=images[name].image,index=fi;
const render=()=>{if(version!==inputVersion)return;ct.clearRect(0,0,540,960);ct.drawImage(im,0,0,540,960);
for(let j=0;j<17;j++){for(let kind=0;kind<2;kind++){const p=kind?d.projected[index][side][j]:d.observed[index][side][j];
if(p[0]===null||p[1]===null)continue;const x=(side?im.naturalWidth-1-p[1]:p[1])*540/im.naturalWidth,y=(side?p[0]:im.naturalHeight-1-p[0])*960/im.naturalHeight;ct.strokeStyle=kind?'#0066cc':'#b51b1b';ct.lineWidth=2;
ct.beginPath();if(kind){ct.moveTo(x-5,y-5);ct.lineTo(x+5,y+5);ct.moveTo(x-5,y+5);ct.lineTo(x+5,y-5);}else{ct.arc(x,y,5,0,2*Math.PI);}ct.stroke();}}};
if(im.complete&&im.naturalWidth)render();else{im.onload=render;im.onerror=()=>{ct.fillText('图像不可用：'+path,8,20);};}}}
"""
    page = page.replace("document.getElementById('slider').oninput=e=>",extra+"document.getElementById('slider').oninput=e=>",1)
    page = page.replace('}function wn(i)', "document.getElementById('frame-number').value=fi;document.getElementById('slider').value=fi;document.getElementById('capture-time').textContent=d.time[fi].toFixed(3)+'s';document.getElementById('heel').textContent=d.heel[fi].map(x=>x===null?'—':x.toFixed(1)+'mm').join(' / ');document.getElementById('force').textContent=d.force[fi]===null?'—':d.force[fi].toFixed(1)+'N';drawInputs();}function wn(i)",1)
    page = page.replace("window.viewerAudit={vertices:", "window.viewerAudit={method:"+json.dumps(label)+",vertices:",1)
    # Compress the numeric payload without changing it; keep a single-file export.
    compact = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',',':'))
    compressed = base64.b64encode(gzip.compress(compact.encode('utf-8'))).decode('ascii')
    page = re.sub(r'(<script id="data" type="application/json">).*?(</script>)',r'\1{}\2',page,count=1,flags=re.S)
    page = page.replace("const d=JSON.parse(document.getElementById('data').textContent),canvas=", 'const startViewer=d=>{const canvas=',1)
    position = page.rfind('</script>')
    loader = ("};new Response(new Blob([Uint8Array.from(atob('"+compressed+"'),c=>c.charCodeAt(0))]).stream()"
              ".pipeThrough(new DecompressionStream('gzip'))).text().then(JSON.parse).then(startViewer)"
              ".catch(e=>document.body.insertAdjacentHTML('beforeend','<pre>数据加载失败：'+e+'</pre>'));\n")
    return page[:position]+loader+page[position:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--view-dir', default='view', help='new output directory name within this experiment')
    parser.add_argument('--candidate',action='append',default=[],help='extra saved run, rendered independently with unchanged states')
    parser.add_argument('--skip-replay', action='store_true', help='resume pages using this experiment own completed scene')
    args = parser.parse_args()
    root = args.root.resolve()
    for candidate in args.candidate:
        LABELS[candidate]={'matched_body':'PHC+ 修正骨架（保留）',
            'stage_heel_v2':'脚跟约束诊断（未通过）'}.get(Path(candidate).name,candidate)
    destination = root/args.view_dir
    metadata = read(root/'reference448.json')
    source = Path(metadata['source'])
    parameters = read(source.parent/'run_parameters.json')
    if not args.skip_replay:
        replay(destination, parameters, Path(metadata['scene']).parent)
    scene_dir = destination/'scene'
    provenance = read(scene_dir/'scene_sources.json')
    if any(provenance[k] for k in ('old_stage_jsonl_read','old_dynamic_ground_read','old_dynamic_walker_pose_read')):
        raise ValueError('old dynamic display inputs forbidden')
    for side in ('left','right'):
        if Path(provenance['raw_'+side+'_pmpose']).resolve()!=Path(parameters['inputs'][side+'_predictions']).resolve():
            raise ValueError('scene/body source mismatch')
    with np.load(scene_dir/'scene_transforms.npz', allow_pickle=True) as stored:
        scene = {key: stored[key] for key in stored.files}
    with np.load(root/'reference448.npz', allow_pickle=False) as reference:
        # Old scene is comparison-only: all displayed geometry uses the replay.
        np.testing.assert_allclose(scene['rotation_ground_from_left'],reference['camera_rotation'],atol=1e-9,rtol=0)
        np.testing.assert_allclose(scene['translation_ground_from_left_mm']/1000,reference['camera_translation_m'],atol=1e-9,rtol=0)
        np.testing.assert_allclose(scene['pair_timestamp_sec'],reference['timestamp_s'],atol=1e-7,rtol=0)
        np.testing.assert_array_equal(scene['stage'].astype(str),reference['stage'])
    rotation = scene['rotation_ground_from_left']
    translation = scene['translation_ground_from_left_mm']/1000
    def xf(points):
        return np.einsum('nij,nvj->nvi', rotation, points)+translation[:,None]
    walker = read(scene_dir/'static_walker_model.json')
    names = sorted(set(name for edge in walker['edges'] for name in edge))
    wr = np.asarray(walker['rotation_left_camera_from_walker'])
    wt = np.asarray(walker['translation_left_camera_from_walker_mm'])/1000
    nodes = np.asarray([walker['nodes_walker_mm'][name] for name in names])/1000@wr.T+wt
    wg = xf(np.broadcast_to(nodes,(448,len(nodes),3)))
    handles = np.asarray([[walker['nodes_walker_mm'][name] for name in walker['handle_segments'][side]] for side in ('left','right')])/1000
    he = xf(np.broadcast_to(handles.reshape(4,3)@wr.T+wt,(448,4,3))).reshape(448,2,2,3)
    cal = load_stereo_fisheye(Path(parameters['inputs']['calibration_dir']))
    centers = np.stack([np.zeros(3),-cal.R_cam0_to_cam1.T@(cal.T_cam0_to_cam1_mm/1000)])
    cameras = xf(np.broadcast_to(centers,(448,2,3)))
    strict,reasons = recorded_strict_gate(destination/'first_instance_strict.jsonl',np.arange(448))
    np.testing.assert_array_equal(strict,scene['accepted'])
    ground = [json.loads(line) for line in (scene_dir/'dynamic_ground_pose.jsonl').read_text(encoding='utf-8').splitlines()]
    obs = np.stack([_load_predictions(Path(parameters['inputs'][side+'_predictions']),side) for side in ('left','right')],1)
    sets = read(ROOT/'research_records/engineering_validation/G20260927_wilor_smplh_full448_v1/contact_surface_sets_smplh_v1/contact_vertex_sets.json')['sets']
    soles = np.unique(sum([sum(sets[side+'_sole_surface_candidate'].values(),[]) for side in ('left','right')],[]))
    for method,label in LABELS.items():
        used = np.zeros((448,17),bool)
        if method=='source_smplh':
            with np.load(source,allow_pickle=False) as z:
                vertices,coco,faces = xf(z['vertices']),xf(z['predicted_coco']),z['faces'].copy()
                status = np.where(z['pose_available'],'source_estimate','unavailable').tolist()
            reference_error = [None]*448
            heel, force = [[None,None]]*448, [None]*448
            support=False
            configuration='原始二维反馈估计，无物理仿真'
            records = [json.loads(line) for line in (source.parent/'trajectory.jsonl').read_text(encoding='utf-8').splitlines()]
            used[:,5:17] = np.any(np.asarray([row['used_views'] for row in records]),axis=1)
        else:
            summary=read(root/method/'summary.json')
            support=summary['assumed_wrist_support']
            configuration=('原始中性骨架' if not method.startswith('corrections_v1') else '固定主线体型骨架')+('；Stage2脚跟软约束' if summary.get('stage_heel_anchor') else '；未加脚跟约束')
            with np.load(root/method/'decoded_smplh.npz',allow_pickle=False) as z:
                vertices,coco,faces = z['world_vertices'],z['world_predicted_coco'],z['faces']
                heel = finite(z['stage2_heel_episode_displacement_m']*1000)
            with np.load(root/method/'rollout.npz',allow_pickle=False) as z:
                status = z['status'].tolist()
                reference_error = finite(z['error_m'].mean(1)*1000)
            with np.load(root/method/'support_diagnostic.npz',allow_pickle=False) as z:
                force = finite(z['maximum_force_n'])
        if vertices.shape!=(448,6890,3) or faces.shape!=(13776,3):
            raise ValueError('full same-parameter SMPL-H mesh required')
        camera_coco = np.einsum('nji,nvj->nvi',rotation,coco-translation[:,None])
        right_coco = camera_coco@cal.R_cam0_to_cam1.T+cal.T_cam0_to_cam1_mm/1000
        projected = np.stack([fisheye_project_numpy(camera_coco,cal.K0,cal.D0),
            fisheye_project_numpy(right_coco,cal.K1,cal.D1)],1)
        for side, points in enumerate((camera_coco,right_coco)):
            projected[:,side][points[...,2]<=0] = np.nan
        errors = np.linalg.norm(projected[:,:,5:17]-obs[:,:,5:17,:2],axis=-1)
        data = dict(v=encode(vertices),j=encode(coco),faces=faces.tolist(),n=448,nv=6890,ids=list(range(448)),
            t=encode(xf(scene['detector_only_triangulated_points_mm']/1000)),w=encode(wg),he=encode(he),c=encode(cameras),
            a=encode(xf(scene['raw_triangulated_points_mm'][:,[15,16]]/1000)),av=strict[:,[15,16]].ravel().tolist(),
            wn=names,we=[[names.index(a),names.index(b)] for a,b in walker['edges']],si=soles.tolist(),pi=[],
            acc=strict.ravel().tolist(),fit_acc=used.tolist(),stage=scene['stage'].tolist(),
            pose=[row['pose_audit']['status'] for row in ground],ground_reasons=[row['pose_audit'].get('reason') for row in ground],
            rotation_missing=[row['pose_audit'].get('missing_rotation_edges',0) for row in ground],
            e3=reference_error,hm=[None]*448,fm=finite(vertices[:,soles,2].min(1)),
            hand_state=[['假设K900 D45, 300N上限' if support else '无外部支撑'] for _ in range(448)],
            wrist_mm=[[None,None]]*448,reprojection=finite(np.nanmedian(errors,axis=2)),reject_reasons=reasons.tolist(),
            status=status,heel=heel,force=force,configuration=configuration,
            comparisonPlot='../corrections_v1/comparison_all/comparison.png' if args.candidate else '../comparison.png',
            time=scene['pair_timestamp_sec'].tolist(),observed=finite(obs),projected=finite(projected),
            imageRoot=os.path.relpath(parameters['inputs']['input_pairs'],destination).replace('\\','/'))
        (destination/(method.replace('/','__')+'.html')).write_text(page_for(data,vertices,coco,label),encoding='utf-8')
        print('Exported',method,flush=True)
    (destination/'metadata.json').write_text(json.dumps(dict(frames=448,vertices=6890,faces=13776,
        scene=str(scene_dir),source=str(source),methods=LABELS,display_map='X,-Y,Z',smoothing=False,
        interpolation=False,scene_replayed_matches_controller_conditioning=True,
        browser_status='pending'),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
