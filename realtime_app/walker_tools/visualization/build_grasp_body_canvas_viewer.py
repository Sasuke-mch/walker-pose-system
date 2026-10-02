"""Build an offline, same-run full-mesh viewer using the existing Canvas renderer.

No old dynamic trajectories, substituted hand meshes or interpolated frames.
The old Canvas page is a rendering template only; all data are generated here.
"""
from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()

import argparse
import base64
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "realtime_app"))
from pose_app.fisheye_camera import load_stereo_fisheye, fisheye_project_numpy
from walker_tools.visualization.build_surface_contact_canvas_viewer import PAGE


def simplify_cold_viewer_text(page, label, frames):
    """Edit labels around the embedded payload; preserve numerical data exactly."""
    match = re.search(r'(<script id="data" type="application/json">)(.*?)(</script>)', page, re.S)
    if match is None:
        raise ValueError('missing viewer payload')
    def labels(text):
        text = text.replace('<section><h2>状态</h2><p>腕参考未验证；接触未通过。</p></section>', '')
        text = re.sub(r'<title>.*?</title>', '<title>'+label+'</title>', text)
        text = re.sub(r'<h1>.*?</h1>', '<h1>'+label+'</h1>', text)
        text = re.sub(r'本地离线版 · 第\d+\.\.\d+帧 · .*?本次重放', f'SMPL-H · {frames} 帧', text)
        text = re.sub(r'<section><h2>本次拟合</h2><p>.*?</p></section>',
                      '', text)
        replacements = {'人工腕参考偏差':'腕偏差（左/右）','身体观测残差':'观测残差',
            '双手几何状态':'手部检查','脚底最低高度':'脚底高度','地面变换':'地面状态',
            '当前质量与拒绝':'质量检查','SMPL 表面':'人体','助步器实体杆件':'助步器',
            '扶手 capsule':'扶手范围','脚底/手掌接触点':'手脚采样点',
            '相机侧面投影轨迹':'相机轨迹','踝关节拖尾':'踝轨迹','长坐标系':'坐标轴',
            '回投median L/R: ':'回投中位数 左/右：','严格accepted: ':'严格通过：',
            '；拟合候选: ':'；拟合点：','拒绝原因: ':'拒绝原因：'}
        for a,b in replacements.items(): text=text.replace(a,b)
        text = re.sub(r'<section><h2>操作</h2><p>.*?</p></section>',
            '<section><h2>操作</h2><p>拖动旋转，右键平移，滚轮缩放。黑线：观测骨架；蓝点：模型关节；橙叉：拒绝点；红色脚点：低于地面。</p></section>',text)
        text=text.replace('d.stage[fi];', "({warming_up:'准备',transition:'过渡',stage1_walker_static_human_moving:'Stage 1',stage2_feet_static_walker_moving:'Stage 2'}[d.stage[fi]]||d.stage[fi]);")
        text=text.replace('d.pose[fi];', "({held:'保持',accepted:'更新',accepted_rotation_held:'更新（旋转保持）'}[d.pose[fi]]||d.pose[fi]);")
        text=text.replace("d.hand_state[fi].join(' / ')", "d.hand_state[fi].map(x=>x.replaceAll('扶手穿透','穿透').replaceAll('代理未通过','未通过')).join(' / ')")
        return text
    return labels(page[:match.start()])+match.group(0)+labels(page[match.end():])


def encode(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode("ascii")


def validate_frame_ids(ids, scene_frames):
    """Keep original frame identity when selecting a fitted window."""
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids)==0 or np.any(np.diff(ids)!=1) or ids.min()<0 or ids.max()>=scene_frames:
        raise ValueError('expected contiguous original frame IDs within fresh scene')
    return ids


def main():
    p = argparse.ArgumentParser()
    for name in ("result", "scene", "surface-sets", "geometry-audit", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("refuse existing page")
    r = np.load(a.result, allow_pickle=False)
    s = np.load(a.scene / "scene_transforms.npz", allow_pickle=False)
    metadata = json.loads((a.result.parent / "run_metadata.json").read_text(encoding="utf-8"))
    source = json.loads((a.scene / "scene_sources.json").read_text(encoding="utf-8"))
    if source["old_dynamic_ground_read"] or source["old_stage_jsonl_read"] or "selected original stereo pair" not in source["image_source"]:
        raise ValueError("viewer requires fresh ordered-image Stage/ground replay")
    ids = validate_frame_ids(r["pair_id"],len(s['rotation_ground_from_left']))
    n = len(ids)
    if r["vertices"].shape != (n, 6890, 3) or r["faces"].shape != (13776, 3):
        raise ValueError("not a contiguous same-run male mesh with explicit frame IDs")
    for side in ('left','right'):
        if Path(source[f'raw_{side}_pmpose']).resolve()!=Path(metadata['inputs'][f'{side}_raw']).resolve():
            raise ValueError('replay/fit observation source mismatch')
    # Render only the freshly replayed scene. The fit's stored grounded mesh
    # below is an audit of equivalence, never a source of viewer coordinates.
    R = s["rotation_ground_from_left"][ids]
    tr = s["translation_ground_from_left_mm"][ids] / 1000
    def xf(x):
        return np.einsum("nij,nvj->nvi", R, x) + tr[:, None]
    vg, cg, tg = xf(r["vertices"]), xf(r["predicted_coco"]), xf(r["raw_triangulated_points"] / 1000)
    if not np.allclose(vg, r["vertices_ground_m"], atol=1e-5):
        raise ValueError("fit/viewer ground vertex mismatch")
    # Recompute strict observation audit from original same-run rows, separate
    # from the engineering fitting mask. Preserve all finite rejected points.
    spec = importlib.util.spec_from_file_location("viewer_raw", ROOT / "research_records/engineering_validation/G20260923_smpl_clean_full_sequence_v1/run_clean_full_sequence.py")
    raw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(raw)
    import cv2
    raw.cv2 = cv2
    left = raw.raw_side(Path(metadata["inputs"]["left_raw"]), "left")
    right = raw.raw_side(Path(metadata["inputs"]["right_raw"]), "right")
    cal = load_stereo_fisheye(ROOT / "realtime_app/calibration/results")
    left, right = np.stack([left[i] for i in ids]), np.stack([right[i] for i in ids])
    tri, el, er, _, dl, dr, candidate, reasons, quality, components = raw.raw_triangulate(left, right, cal)
    if not np.allclose(tri, r["raw_triangulated_points"], equal_nan=True, atol=.001):
        raise ValueError("triangulation is not same-source")
    strict = candidate & (components[0] >= .25) & ((el + er) / 2 <= 10)
    reasons = np.asarray(reasons, dtype=object)
    reasons[candidate & ~strict & (components[0] < .25)] = "q2d_below_0.25"
    reasons[candidate & ~strict & ((el + er) / 2 > 10)] = "reprojection_above_10px"
    sets = json.loads(a.surface_sets.read_text(encoding="utf-8"))["sets"]
    soles = np.unique(sum([sum(sets[f"{side}_sole_surface_candidate"].values(), []) for side in ("left", "right")], []))
    palms = np.unique(sum([sum(sets[f"{side}_palm_surface_candidate"].values(), []) for side in ("left", "right")], []))
    w = json.loads((a.scene / "static_walker_model.json").read_text(encoding="utf-8"))
    names = sorted(set(x for edge in w["edges"] for x in edge))
    Rc = np.asarray(w["rotation_left_camera_from_walker"])
    tc = np.asarray(w["translation_left_camera_from_walker_mm"]) / 1000
    static_nodes = np.asarray([w["nodes_walker_mm"][name] for name in names]) / 1000 @ Rc.T + tc
    wg = xf(np.broadcast_to(static_nodes, (n, len(names), 3)))
    handles = np.asarray([[np.asarray(w["nodes_walker_mm"][name]) / 1000 for name in w["handle_segments"][side]] for side in ("left", "right")])
    he = xf(np.broadcast_to(handles.reshape(4, 3) @ Rc.T + tc, (n, 4, 3))).reshape(n, 2, 2, 3)
    c1 = -cal.R_cam0_to_cam1.T @ (cal.T_cam0_to_cam1_mm / 1000)
    cameras = xf(np.broadcast_to(np.stack([np.zeros(3), c1]), (n, 2, 3)))
    stages = [json.loads(line)["stage"]["operational"] for line in (a.scene / "stage1_stage2.jsonl").read_text(encoding="utf-8").splitlines()]
    poses = [(json.loads(line).get("pose_audit") or {}).get("status", "unknown") for line in (a.scene / "dynamic_ground_pose.jsonl").read_text(encoding="utf-8").splitlines()]
    stages = [stages[i] for i in ids]
    poses = [poses[i] for i in ids]
    audit = json.loads(a.geometry_audit.read_text(encoding="utf-8"))
    if Path(audit['source']).resolve()!=a.result.resolve():
        raise ValueError('hand geometry audit/result mismatch')
    hand_state = [[None, None] for _ in range(n)]
    reason_labels = {'screened_triangle_crossings':'表面相交', 'handle_penetration_exceeds_3mm':'扶手穿透>3mm',
                     'finger_region_gap_exceeds_5mm':'指区间隙>5mm', 'palm_gap_exceeds_5mm':'掌区间隙>5mm',
                     'thumb_opposition_proxy_failed':'对握代理未通过'}
    index_by_id = {int(value): i for i,value in enumerate(ids)}
    for row in audit["records"]:
        hand_state[index_by_id[row["pair_id"]]][0 if row["hand"] == "left" else 1] = "代理通过" if row["passed_geometry_proxy"] else "、".join(reason_labels.get(x,x) for x in row["reject_reasons"])
    if any(None in row for row in hand_state):
        raise ValueError("incomplete per-frame hand audit")
    residual = np.linalg.norm(cg - tg, axis=-1) * 1000
    e3 = [float(np.median(residual[i, candidate[i]])) if candidate[i].any() else None for i in range(n)]
    projection = []
    for i in range(n):
        p0 = fisheye_project_numpy(r["predicted_coco"][i] * 1000, cal.K0, cal.D0)
        p1 = fisheye_project_numpy(r["predicted_coco"][i] @ cal.R_cam0_to_cam1.T * 1000 + cal.T_cam0_to_cam1_mm, cal.K1, cal.D1)
        projection.append([float(np.median(np.linalg.norm(p0 - left[i, :, :2], axis=1))), float(np.median(np.linalg.norm(p1 - right[i, :, :2], axis=1)))])
    data = {"v": encode(vg), "j": encode(cg), "w": encode(wg), "t": encode(tg), "he": encode(he),
        "c": encode(cameras), "a": encode(tg[:, [15, 16]]), "av": strict[:, [15, 16]].ravel().tolist(),
        "faces": r["faces"].tolist(), "n": n, "nv": 6890, "ids": ids.tolist(), "wn": names,
        "we": [[names.index(x), names.index(y)] for x, y in w["edges"]], "si": soles.tolist(), "pi": palms.tolist(),
        "acc": strict.ravel().tolist(), "fit_acc": candidate.tolist(), "stage": stages, "pose": poses, "e3": e3,
        "hm": [None] * n, "fm": [float(np.min(vg[i, soles, 2])) for i in range(n)],
        "hand_state": hand_state, "reprojection": projection, "quality": quality.tolist(), "reject_reasons": reasons.tolist()}
    if metadata['inputs'].get('wrist_reference'):
        from pose_app.independent_wrist_reference import load_reference
        wrist_target, _ = load_reference(metadata['inputs']['wrist_reference'],
            metadata['inputs']['calibration_dir'], metadata['allow_diagnostic_wrist_reference'])
        data['wrist_mm'] = (np.linalg.norm(r['smplh_joints'][:, [20, 21]]-wrist_target,axis=-1)*1000).tolist()
    else:
        grasp = json.loads(Path(metadata['inputs']['grasp']).read_text(encoding='utf8'))
        wrist_target = np.asarray([grasp['hands'][side]['wrist_walker_m'] for side in ('left','right')])
        data['wrist_mm'] = (np.linalg.norm(r['wrist_walker_m']-wrist_target,axis=-1)*1000).tolist()
    page = PAGE.replace('DATA', json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    page = page.replace('<title>SMPL surface contact Canvas viewer</title>', '<title>SMPL-H 固定手部约束拟合</title>')
    page = page.replace('.metrics div{display:flex;', '.metrics div{display:flex;gap:8px;')
    page = page.replace('.metrics dt{color:#65737d}', '.metrics dt{color:#65737d;flex:0 0 90px}')
    page = page.replace('.metrics dd{margin:0;text-align:right}', '.metrics dd{margin:0;text-align:right;min-width:0;overflow-wrap:anywhere}')
    page = page.replace('zoom=1.0', 'zoom=.65').replace('zoom=1;', 'zoom=.65;')
    # One fixed viewing center for the entire sequence. Camera framing only:
    # never recenter per frame, or apparent world motion would be hidden.
    center = np.median(cg[:, [11, 12], :2].reshape(-1, 2), axis=0)
    center[1] *= -1  # project() receives the common [X,-Y,Z] display map.
    page = page.replace('let x=p[0]*cy-p[1]*sy,y=p[0]*sy+p[1]*cy,z=p[2];',
        f'const px=p[0]-({center[0]:.9g}),py=p[1]-({center[1]:.9g});let x=px*cy-py*sy,y=px*sy+py*cy,z=p[2];')
    page = page.replace('const sideX=-3.8;', 'const sideX=-1.8;')
    page = page.replace('窗口 60..90 · 31 帧 · 无外部库', f'第{ids[0]}..{ids[-1]}帧 · {n}帧 · 6890顶点 / 13776真实三角面 · 本次重放')
    page = page.replace('male SMPL 表面手脚接触拟合', 'SMPL-H 固定手部约束拟合')
    page = page.replace('min="0" max="30"', f'min="0" max="{n-1}"')
    page = page.replace('手部 surface 中位', '双手几何状态').replace('脚部 surface 中位', '脚底最低高度')
    page = page.replace('COCO 3D 误差', '身体观测残差')
    page = page.replace('<dl class="metrics">', '<dl class="metrics"><div><dt>腕偏差 左/右</dt><dd id="wrist">—</dd></div>')
    page = page.replace("(d.hm[fi]===null?'—':(d.hm[fi]*1000).toFixed(1)+' mm')", "d.hand_state[fi].join(' / ')")
    page = page.replace("unpack(d.a)]).then(([v,w,t,he,c,ankle])", "unpack(d.a),unpack(d.j)]).then(([v,w,t,he,c,ankle,model])")
    page = page.replace('const N=d.n,NV=d.nv,faces=d.faces,si=d.si,pi=d.pi,WN=d.wn.length,av=d.av;', 'const N=d.n,NV=d.nv,faces=d.faces,si=d.si,pi=d.pi,WN=d.wn.length,av=d.av;const E=[[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];window.viewerAudit={vertices:NV,faces:faces.length,indexEntries:faces.length*3,frames:N,displayMap:"X,-Y,Z",sceneFresh:true};')
    marker = "if(document.getElementById('walker').checked){"
    overlay = """for(const e of E){const[a,b]=e,A=[t[fi*51+a*3],-t[fi*51+a*3+1],t[fi*51+a*3+2]],B=[t[fi*51+b*3],-t[fi*51+b*3+1],t[fi*51+b*3+2]];if(A.every(Number.isFinite)&&B.every(Number.isFinite)){ctx.setLineDash([]);seg(A,B,'#000000',1.5);ctx.setLineDash([])}}for(let k=0;k<17;k++)dot([model[fi*51+k*3],-model[fi*51+k*3+1],model[fi*51+k*3+2]],'#1678bc',2);seg([c[fi*6],-c[fi*6+1],c[fi*6+2]],[c[fi*6+3],-c[fi*6+4],c[fi*6+5]],'#4b555d',1.5);"""
    # Canvas follows the project's render order: solid walker, raw points,
    # translucent real mesh, observation bones, model COCO points. Keep the
    # reusable template but relocate its mesh block rather than redraw a proxy.
    mesh_start = page.index("if(document.getElementById('mesh').checked){")
    mesh_end = page.index("if(document.getElementById('contact').checked){", mesh_start)
    mesh_block = page[mesh_start:mesh_end]
    page = page[:mesh_start]+page[mesh_end:]
    contact_start = page.index("if(document.getElementById('contact').checked){")
    contact_end = page.index(marker,contact_start)
    contact_block = page[contact_start:contact_end]
    page = page[:contact_start]+page[contact_end:]
    page = page.replace("document.getElementById('stage').textContent=d.stage[fi];", mesh_block+contact_block+overlay+"document.getElementById('stage').textContent=d.stage[fi];")
    page = page.replace("p[2]<0?4.5:3.0", "p[2]<0?1.2:1.0")
    ticks = """for(let u=0;u<=4.5;u+=.5){for(const P of [[u,0,0],[0,u,0]]){const q=project(P);ctx.fillStyle='#27332d';ctx.font='12px sans-serif';ctx.fillText(u.toFixed(1)+'m',q[0]+3,q[1]-3)}}"""
    page = page.replace("if(document.getElementById('trails').checked){", ticks + "if(document.getElementById('trails').checked){")
    page = page.replace("pitch=Math.max(-1.48,Math.min(1.48,pitch+dy*.007))", "pitch+=dy*.007")
    page = page.replace("document.getElementById('pose').textContent=d.pose[fi];", "document.getElementById('pose').textContent=d.pose[fi];document.getElementById('qa').textContent='回投median L/R: '+d.reprojection[fi].map(x=>x.toFixed(1)).join('/')+' px\\n严格accepted: '+d.acc.slice(fi*17,fi*17+17).filter(Boolean).length+'/17；拟合候选: '+d.fit_acc[fi].filter(Boolean).length+'/17\\n拒绝原因: '+d.reject_reasons[fi].map((x,i)=>x===null?'':i+':'+x).filter(Boolean).join(';');window.viewerAudit.frame=fi;")
    page = page.replace("window.viewerAudit.frame=fi;", "window.viewerAudit.frame=fi;document.getElementById('wrist').textContent=d.wrist_mm[fi].map(x=>x.toFixed(1)).join(' / ')+' mm';")
    page = page.replace('<section><h2>操作</h2>', '<section><h2>当前质量与拒绝</h2><pre id="qa" style="white-space:pre-wrap;font-size:12px"></pre></section><section><h2>操作</h2>')
    if metadata.get('cold_start'):
        detector_label = 'PMPose' if metadata.get('detector') == 'pmpose' else 'Sapiens2'
        page = page.replace('SMPL-H 固定手部约束拟合', detector_label)
        page = page.replace('腕偏差 左/右', '腕偏差（左/右）')
        page = page.replace('project([-6,-6,0]),project([6,-6,0]),project([6,6,0]),project([-6,6,0])',
                            'project([-2.5,-2.5,0]),project([2.5,-2.5,0]),project([2.5,2.5,0]),project([-2.5,2.5,0])')
        page = page.replace("for(let k=-6;k<=6;k++){seg([k,-6,0],[k,6,0],'#b9c8c0',1);seg([-6,k,0],[6,k,0],'#b9c8c0',1)}",
                            "for(let k=-2.5;k<=2.5;k+=.5){seg([k,-2.5,0],[k,2.5,0],'#b9c8c0',1);seg([-2.5,k,0],[2.5,k,0],'#b9c8c0',1)}")
        page = simplify_cold_viewer_text(page, detector_label, n)
    a.output.write_text(page, encoding="utf-8")
    joint_audit = {str(j): {"finite": int(np.isfinite(tri[:, j]).all(1).sum()), "strict_accepted": int(strict[:, j].sum()), "finite_rejected": int((np.isfinite(tri[:, j]).all(1) & ~strict[:, j]).sum()), 'rejection_reasons':{str(value):int((reasons[:,j]==value).sum()) for value in set(reasons[:,j]) if value is not None}} for j in range(17)}
    (a.output.parent / "viewer_validation.json").write_text(json.dumps({"status": "file_checks_passed_browser_pending", "vertices": 6890, "faces": 13776, "index_entries": 41328, "frames": n, "joints": joint_audit, "source": str(a.result.resolve()), "scene": str(a.scene.resolve()), "stage_counts": {x: stages.count(x) for x in set(stages)}, "no_display_smoothing": True}, indent=2), encoding="utf-8")
    print(a.output.resolve())


if __name__ == "__main__":
    main()
