"""Export saved causal SMPL-H results with a freshly replayed display scene."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(APP), str(APP / 'tools')]

from pose_app.fisheye_camera import load_stereo_fisheye, fisheye_project_numpy  # noqa: E402
from pose_app.smplh_first_frame_labels import apply_first_frame_labels  # noqa: E402
from run_stereo_pva_baseline import _load_predictions  # noqa: E402
from walker_tools.visualization.build_grasp_body_canvas_viewer import (  # noqa: E402
    build_canvas_page, encode, recorded_strict_gate, validate_frame_ids,
)

NAMES = ['左肩','右肩','左肘','右肘','左腕','右腕','左髋','右髋','左膝','右膝','左踝','右踝']
TITLE = 'SMPL-H 二维运动跟踪 · 1.2候选'
PANEL = '''<section class="motion-panel"><div class="motion-controls">
<label>关节 <select id="joint" aria-label="关节">JOINT_OPTIONS</select></label>
<label><input id="raw" type="checkbox">查看未通过发布门的原始估计</label>
<span id="motion-status"></span></div>
<p>速度与加速度：当前左相机系 XYZ（含相机运动），真实非均匀时间；黑实线 X、灰虚线 Y、浅灰点线 Z。空缺不补点，重启不连线。</p>
<div class="motion-charts"><canvas id="velocity-chart"></canvas><canvas id="acceleration-chart"></canvas></div>
<div class="input-images"><figure><figcaption>左目同源图像：圆圈观测，蓝叉模型投影</figcaption><canvas id="left-image"></canvas></figure>
<figure><figcaption>右目同源图像：圆圈观测，蓝叉模型投影</figcaption><canvas id="right-image"></canvas></figure></div></section>'''
STYLE = '''<style>.app{height:auto;min-height:0;grid-template-rows:auto 68vh auto}
.main{grid-template-columns:1fr 320px}.side{font-size:13px}.side p{line-height:1.5}
.bottom{position:sticky;bottom:0;z-index:3;flex-wrap:wrap}.bottom #frame-number{flex:0 0 65px;width:65px}
.motion-panel{background:#fff;padding:14px;border-top:1px solid #000}.motion-controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap}
.motion-panel p{font-size:13px}.motion-charts{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.motion-charts canvas{width:100%;height:220px;display:block;border:1px solid #000;background:#fff}
.input-images{display:flex;justify-content:center;gap:24px;flex-wrap:wrap}.input-images figure{margin:12px 0}.input-images canvas{width:270px;height:480px;border:1px solid #000}
select,input{background:#fff;color:#000;border:1px solid #000}.side pre{overflow-wrap:anywhere}
@media(max-width:800px){.main{grid-template-columns:1fr 250px}.motion-charts{grid-template-columns:1fr}}</style>'''


def json_array(values):
    array = np.asarray(values)
    return np.where(np.isfinite(array), array, None).tolist()


def scene_observations(scene, ids):
    """Keep finite raw points for display and strict ankles for accepted trails.

    The scene's raw_triangulated_points_mm field is actually accepted-only.
    Detector-only points remain diagnostic geometry, never restored Stage input.
    """
    raw = np.asarray(scene['detector_only_triangulated_points_mm'])[ids]
    strict = np.asarray(scene['raw_triangulated_points_mm'])[ids]
    if raw.shape != (len(ids), 17, 3) or strict.shape != raw.shape:
        raise ValueError('expected aligned raw and strict COCO-17 scene points')
    return raw, strict[:, [15, 16]]


def motion_page(page, data):
    """Add motion diagnostics without changing the supplied scene or mesh arrays."""
    page = page.replace('<title>PMPose</title>', '<title>'+TITLE+'</title>')
    page = page.replace('<h1>PMPose</h1>', '<h1>'+TITLE+'</h1>')
    page = page.replace('SMPL-H · 448 帧', '448帧 · 完整网格 · 无平滑/插值')
    for old, new in [('腕偏差（左/右）','跟踪状态'),('观测残差','二维回投 左/右'),
                     ('参考扶手几何','可发布导数'),('手脚采样点','脚底候选点'),('脚底高度','脚底 L/R')]:
        page = page.replace(old, new)
    page = page.replace('<div><dt>地面状态</dt>', '<div><dt>助步器四脚</dt><dd id="walker-height">—</dd></div><div><dt>地面状态</dt>')
    shape_label = '原主线全片校准体型（仅beta）' if data['motion']['shapeCalibrated'] else '首帧体型'
    beta = ', '.join(f'{value:.4f}' for value in data['motion']['beta'][0])
    shape_section = '<section><h2>固定体型</h2><p>'+shape_label+'；跟踪期间冻结。</p><pre style="white-space:pre-wrap;font-size:12px">['+beta+']</pre></section>'
    if data['motion']['shapeCalibrated']:
        page = page.replace(TITLE,'SMPL-H 二维运动跟踪 · 固定主线体型')
    world = data['motion'].get('world', False)
    if world:
        page = page.replace('SMPL-H 二维运动跟踪 · 固定主线体型','SMPL-H 世界运动 · Stage条件约束候选')
    page = re.sub(r'document.getElementById\(\'wrist\'\)\.textContent=d\.wrist_mm\[fi\].*?;', '', page)
    # The existing renderer updates its metrics after the frame label. Run the
    # motion diagnostics at the end of draw, after those legacy assignments.
    page = page.replace("}function wn(i)", "drawMotion();renderInputs();document.getElementById('frame-number').value=fi;}function wn(i)")
    if 'drawMotion();renderInputs();' not in page:
        raise ValueError('existing Canvas drawing hook changed')
    source_text = ('本候选在固定地面系联合估计q/v/a，使用本轮Stage相机变换回投双目像素；Stage本身含脚踝锚和约0.5秒固定延迟。脚底约束和世界运动均条件于该采集假设，不是独立世界轨迹或物理接触认证。' if world else
                   '人体来自保存的因果跟踪结果；地面、相机和助步器仅用本轮Stage重放作显示参考。三角化只用于显示，未进入人体跟踪损失。速度、加速度仍是相机系估计。')
    page = page.replace('<section><h2>参考来源</h2><p>', '<section><h2>坐标与来源</h2><p>'+source_text)
    page = page.replace('<section><h2>操作</h2>', '<section><h2>视角</h2><div><button data-motion-view="recommended">推荐</button><button data-motion-view="front">正视</button><button data-motion-view="side">侧视</button><button data-motion-view="top">俯视</button></div></section><section><h2>操作</h2>')
    page = page.replace('<section><h2>视角</h2>',shape_section+'<section><h2>视角</h2>',1)
    page = page.replace('<input id="slider"', '<span id="time"></span><input id="frame-number" type="number" min="0" max="447" value="0" aria-label="帧号"><input id="slider"')
    page = page.replace('</style>', '</style>'+STYLE, 1)
    options = ''.join(f'<option value="{j}">{name}</option>' for j,name in enumerate(NAMES))
    panel = PANEL.replace('JOINT_OPTIONS',options)
    if world:
        panel = panel.replace('当前左相机系 XYZ（含相机运动）', 'Stage条件固定地面系 XYZ（未经独立世界验证）')
    page = page.replace('<script id="data"', panel+'<script id="data"',1)
    additions = (APP / 'walker_tools/visualization/smplh_motion_viewer.js').read_text(encoding='utf-8')
    page = page.replace("document.getElementById('slider').oninput=e=>", additions+"\ndocument.getElementById('slider').oninput=e=>",1)
    flags = 'worldDerivatives:true,worldStateAvailable:true,worldPoseCertified:false,displaySceneOnly:false,' if world else 'worldDerivatives:false,worldStateAvailable:false,displaySceneOnly:true,'
    page = page.replace('window.viewerAudit={vertices:', 'window.viewerAudit={'+flags+'vertices:')
    # Hide optional hand/handle proxy overlays which the tracker never estimated.
    page = page.replace('id="handles" type="checkbox" checked', 'id="handles" type="checkbox"')
    compact = json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    compressed = gzip.compress(compact.encode('utf-8'))
    page = re.sub(r'(<script id="data" type="application/json">).*?(</script>)', r'\1{}\2', page, count=1, flags=re.S)
    old = "const d=JSON.parse(document.getElementById('data').textContent),canvas="
    init = "const canvas="
    if old not in page:
        raise ValueError('existing Canvas initialization hook changed')
    page = page.replace(old,init,1)
    page = page.replace("const canvas=document.getElementById('cv')", "const startViewer=d=>{const canvas=document.getElementById('cv')",1)
    loader = "const loadGzip=b=>new Response(new Blob([b]).stream().pipeThrough(new DecompressionStream('gzip'))).text().then(JSON.parse).then(startViewer).catch(e=>document.body.insertAdjacentHTML('beforeend','<pre>数据加载失败：'+e+'</pre>'));"
    closing = '</script>'
    at = page.rfind(closing)
    page = page[:at]+'};\n'+loader+'LOAD_PAYLOAD\n'+page[at:]
    embedded = "loadGzip(Uint8Array.from(atob('"+base64.b64encode(compressed).decode('ascii')+"'),c=>c.charCodeAt(0)));"
    browser = "fetch('body_data.json.gz').then(r=>{if(!r.ok)throw new Error(r.status);return r.arrayBuffer()}).then(b=>loadGzip(new Uint8Array(b))).catch(e=>document.body.insertAdjacentHTML('beforeend','<pre>请求失败：'+e+'</pre>'));"
    return page.replace('LOAD_PAYLOAD',embedded), page.replace('LOAD_PAYLOAD',browser), compressed


def run(args):
    output = args.output.resolve()
    if (output.exists() or (output.parent/'body_browser.html').exists()) and not args.replace_view:
        raise FileExistsError('refusing existing viewer')
    run_dir = args.run.resolve()
    if args.replace_view:
        prior=json.loads((output.parent/'metadata.json').read_text(encoding='utf-8'))
        if Path(prior['result']).resolve()!=run_dir/'result.npz':
            raise ValueError('refusing to replace a different result viewer')
    p = json.loads((run_dir/'run_parameters.json').read_text(encoding='utf-8'))
    records = [json.loads(line) for line in (run_dir/'trajectory.jsonl').read_text(encoding='utf-8').splitlines()]
    scene = args.scene.resolve()
    source = json.loads((scene/'scene_sources.json').read_text(encoding='utf-8'))
    if any(source[k] for k in ('old_stage_jsonl_read','old_dynamic_ground_read','old_dynamic_walker_pose_read')):
        raise ValueError('old dynamic scene inputs are forbidden')
    for side in ('left','right'):
        if Path(source[f'raw_{side}_pmpose']).resolve()!=Path(p['inputs'][side+'_predictions']).resolve():
            raise ValueError('scene and tracker observations differ')
    if Path(source['input_pair_dir']).resolve()!=Path(p['inputs']['input_pairs']).resolve():
        raise ValueError('scene and tracker image identities differ')
    cal_dir = Path(p['inputs']['calibration_dir'])
    if Path(source['calibration']).resolve()!=(cal_dir/'stereo_fisheye.json').resolve():
        raise ValueError('scene and tracker calibration differ')
    with np.load(run_dir/'result.npz',allow_pickle=False) as stored:
        r = {key:stored[key] for key in ('pair_id','vertices','faces','predicted_coco','betas','timestamp_s','pose_available',
             'joint_velocity_mm_s','joint_acceleration_mm_s2','raw_joint_velocity_mm_s','raw_joint_acceleration_mm_s2','derivative_valid','segment_id')}
    with np.load(scene/'scene_transforms.npz',allow_pickle=False) as stored:
        s = {key:stored[key] for key in ('rotation_ground_from_left','translation_ground_from_left_mm','accepted','raw_triangulated_points_mm','detector_only_triangulated_points_mm','pair_timestamp_sec')}
    ids = validate_frame_ids(r['pair_id'],len(s['rotation_ground_from_left']))
    n = len(ids)
    if n!=448 or [row['frame_index'] for row in records]!=ids.tolist():
        raise ValueError('expected complete ordered448 model/record frames')
    if r['vertices'].shape!=(n,6890,3) or r['faces'].shape!=(13776,3) or r['faces'].min()<0 or r['faces'].max()>=6890:
        raise ValueError('real same-run mesh topology required')
    if not np.array_equal(r['derivative_valid'],np.array([row['derivative_valid'] for row in records])):
        raise ValueError('derivative masks disagree')
    if not np.allclose(r['timestamp_s'],[row['timestamp_s'] for row in records],atol=1e-10,rtol=0):
        raise ValueError('tracker timestamps disagree')
    if not np.allclose(r['timestamp_s'],s['pair_timestamp_sec'],atol=1e-7,rtol=0):
        raise ValueError('scene and tracker pair times disagree')
    strict,reasons = recorded_strict_gate(args.strict_jsonl,ids)
    if not np.array_equal(strict,s['accepted']):
        raise ValueError('display masks and scene masks differ')
    R=s['rotation_ground_from_left'][ids]; tr=s['translation_ground_from_left_mm'][ids]/1000
    def xf(values):
        return np.einsum('nij,nvj->nvi',R,values)+tr[:,None]
    raw_points, strict_ankles = scene_observations(s, ids)
    vg,cg,tg=xf(r['vertices']),xf(r['predicted_coco']),xf(raw_points/1000)
    ankle_ground = xf(strict_ankles/1000)
    world = p.get('world_state_available', False)
    if world:
        if Path(p['world_scene']).resolve() != scene:
            raise ValueError('world tracker and displayed scene differ')
        with np.load(run_dir/'result.npz',allow_pickle=False) as stored:
            if not np.allclose(vg,stored['world_vertices'],atol=2e-6,rtol=0,equal_nan=True):
                raise ValueError('camera/world mesh export transform disagrees')
    w=json.loads((scene/'static_walker_model.json').read_text(encoding='utf-8'))
    names=sorted(set(name for edge in w['edges'] for name in edge))
    Rc=np.asarray(w['rotation_left_camera_from_walker']); tc=np.asarray(w['translation_left_camera_from_walker_mm'])/1000
    static_nodes=np.asarray([w['nodes_walker_mm'][name] for name in names])/1000@Rc.T+tc
    wg=xf(np.broadcast_to(static_nodes,(n,len(names),3)))
    handles=np.asarray([[w['nodes_walker_mm'][name] for name in w['handle_segments'][side]] for side in ('left','right')])/1000
    he=xf(np.broadcast_to(handles.reshape(4,3)@Rc.T+tc,(n,4,3))).reshape(n,2,2,3)
    cal=load_stereo_fisheye(cal_dir)
    c1=-cal.R_cam0_to_cam1.T@(cal.T_cam0_to_cam1_mm/1000)
    cameras=xf(np.broadcast_to(np.stack([np.zeros(3),c1]),(n,2,3)))
    stage_rows=[json.loads(line) for line in (scene/'stage1_stage2.jsonl').read_text(encoding='utf-8').splitlines()]
    ground_rows=[json.loads(line) for line in (scene/'dynamic_ground_pose.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(stage_rows)!=n or len(ground_rows)!=n:
        raise ValueError('published scene frame count differs')
    sets=json.loads(args.surface_sets.read_text(encoding='utf-8'))['sets']
    soles=np.unique(sum([sum(sets[side+'_sole_surface_candidate'].values(),[]) for side in ('left','right')],[]))
    side_soles=[np.unique(sum(sets[side+'_sole_surface_candidate'].values(),[])) for side in ('left','right')]
    foot_nodes=[i for i,name in enumerate(names) if w['nodes_walker_mm'][name][2]==min(value[2] for value in w['nodes_walker_mm'].values())]
    if len(foot_nodes)!=4:
        raise ValueError('static walker topology must identify four bottom foot nodes')
    obs=np.stack([_load_predictions(Path(p['inputs'][side+'_predictions']),side) for side in ('left','right')],1)
    if p.get('first_frame_labels'):
        apply_first_frame_labels(Path(p['first_frame_labels']),Path(p['inputs']['input_pairs']),obs[:,0],obs[:,1])
    model=r['predicted_coco'].astype(float)
    projected=np.stack([fisheye_project_numpy(model,cal.K0,cal.D0),fisheye_project_numpy(model@cal.R_cam0_to_cam1.T+cal.T_cam0_to_cam1_mm/1000,cal.K1,cal.D1)],1)
    reprojection=np.linalg.norm(projected[:,:,5:17]-obs[:,:,5:17,:2],axis=-1)
    recorded=np.array([row['reprojection_px'] for row in records],float)
    difference=float(np.nanmax(abs(reprojection-recorded)))
    if difference>1e-3:
        raise ValueError('2D diagnostics differ from tracker observations')
    fit_acc=np.zeros((n,17),bool)
    fit_acc[:,5:17]=np.any(np.array([row['used_views'] for row in records]),axis=1)
    data={'v':encode(vg),'j':encode(cg),'t':encode(tg),'w':encode(wg),'he':encode(he),'c':encode(cameras),
          'a':encode(ankle_ground),'av':strict[:,[15,16]].ravel().tolist(),
          'n':n,'nv':6890,'faces':r['faces'].tolist(),'ids':ids.tolist(),'wn':names,
          'we':[[names.index(a),names.index(b)] for a,b in w['edges']], 'si':soles.tolist(),'pi':[],
          'acc':strict.ravel().tolist(),'fit_acc':fit_acc.tolist(),'stage':[row['stage']['operational'] for row in stage_rows],
          'pose':[(row.get('pose_audit') or {}).get('status','unknown') for row in ground_rows],
          'ground_reasons':[(row.get('pose_audit') or {}).get('reason') for row in ground_rows],
          'rotation_missing':[(row.get('pose_audit') or {}).get('missing_rotation_edges',0) for row in ground_rows],
          'e3':[None]*n,'hm':[None]*n,'fm':json_array(np.min(vg[:,soles,2],axis=1)),
          'hand_state':[['冻结手姿','冻结手姿'] for _ in range(n)],'wrist_mm':[[None,None] for _ in range(n)],
          'reprojection':json_array(np.nanmedian(reprojection,axis=2)),'reject_reasons':reasons.tolist()}
    data['motion']={'names':NAMES,'time':r['timestamp_s'].tolist(),'segment':r['segment_id'].tolist(),
        'world':world,'groundConstraint':[row.get('ground_constraint') for row in records],
        'footLimitMm':p.get('ground_configuration',{}).get('max_heel_displacement_m',.01)*1000,
        'beta':r['betas'].tolist(),'shapeCalibrated':p.get('shape_fit') is not None,
        'soleHeight':json_array(np.stack([vg[:,indices,2].min(1)*1000 for indices in side_soles],axis=1)),
        'walkerFootHeight':json_array(wg[:,foot_nodes,2]*1000),
        'velocity':json_array(r['joint_velocity_mm_s']),'acceleration':json_array(r['joint_acceleration_mm_s2']),
        'rawVelocity':json_array(r['raw_joint_velocity_mm_s']),'rawAcceleration':json_array(r['raw_joint_acceleration_mm_s2']),
        'valid':r['derivative_valid'].tolist(),'reason':[row['derivative_reason'] for row in records],
        'status':[row['model_status'] for row in records],'used':[row['used_views'] for row in records],
        'viewReason':[row['view_reason'] for row in records],'observed':json_array(obs),'projected':json_array(projected),
        'imageRoot':os.path.relpath(p['inputs']['input_pairs'],output.parent).replace('\\','/')}
    page=build_canvas_page(data,vg,cg,{'inputs':{},'cold_start':True,'detector':'pmpose'})
    embedded,browser,compressed=motion_page(page,data)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(embedded,encoding='utf-8')
    (output.parent/'body_browser.html').write_text(browser,encoding='utf-8')
    (output.parent/'body_data.json.gz').write_bytes(compressed)
    metadata={'result':str(run_dir/'result.npz'),'records':str(run_dir/'trajectory.jsonl'),'scene':str(scene),
        'strict_input':str(args.strict_jsonl.resolve()),'surface_sets':str(args.surface_sets.resolve()),'frames':n,
        'vertices':6890,'faces':13776,'display_map':'X,-Y,Z','smoothing':False,'interpolation':False,
        'observation_display_source':'same_scene_detector_only_triangulated_points_mm',
        'strict_points_role':'unchanged_stage_inputs_and_accepted_ankle_trails',
        'beta':data['motion']['beta'],'shape_fit':p.get('shape_fit'),
        'world_state_available':world,'world_pose_certified':False,
        'derivative_coordinate_frame':p.get('coordinate_frame','current_left_camera'),
        'scene_role':'stage_conditioned_world_estimation' if world else 'display_reference_only',
        'stage_counts':dict(Counter(data['stage'])),'derivative_available_samples':int(r['derivative_valid'].sum()),
        'reprojection_vs_record_max_difference_px':difference,
        'joints':{str(j):{'finite':int(np.isfinite(tg[:,j]).all(1).sum()),'accepted':int(strict[:,j].sum()),
                  'finite_rejected':int((np.isfinite(tg[:,j]).all(1)&~strict[:,j]).sum()),
                  'reject_reasons':dict(Counter(str(x) for x in reasons[:,j] if x is not None))} for j in range(17)},
        'pages':[str(output),str(output.parent/'body_browser.html')],'browser_status':'pending'}
    (output.parent/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:metadata[key] for key in ('frames','stage_counts','derivative_available_samples','reprojection_vs_record_max_difference_px','pages')},ensure_ascii=False))
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('run','scene','strict-jsonl','surface-sets','output'):
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--replace-view',action='store_true',help='Replace an existing export of the same result; model outputs are read-only.')
    run(parser.parse_args())
