"""Audit fixed PCA hand articulation and camera-rigid handle geometry.

No dynamic ground or old posed mesh is consumed. The fixed static walker
installation is an engineering assumption. Plots show actual frames 0,N//2,N-1.
"""
from __future__ import annotations

# Shared bootstrap for direct scripts, the dispatcher and legacy imports.
import sys as _tool_sys
from pathlib import Path as _ToolPath
_tool_sys.path.insert(0, str(_ToolPath(__file__).resolve().parents[2]))
from walker_tools._compat import prepare_imports as _tool_prepare_imports
_tool_prepare_imports()
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pose_app.smpl_coco_observation import _install_legacy_smpl_pickle_compatibility


def stats(x):
    return {"median": float(np.median(x)), "p95": float(np.percentile(x,95)),
            "min": float(np.min(x)), "max": float(np.max(x))}


def capsule(p, a, b, radius):
    d=b-a
    u=np.clip(((p-a)*d).sum(-1)/np.dot(d,d),0,1)
    return np.linalg.norm(p-a-u[...,None]*d,axis=-1)-radius


def crossing_pairs(vertices, faces):
    """Noncoplanar triangle intersections, excluding shared mesh vertices.

    A diagnostic, not a complete collision solver: coplanar overlap/tangency
    and truncated wrist boundaries are explicitly outside this screen.
    """
    tri=vertices[faces]; lo=tri.min(1);hi=tri.max(1);hits=[]
    for i in range(len(tri)):
        js=np.flatnonzero((np.arange(len(tri))>i)&(lo<=hi[i]).all(1)&(hi>=lo[i]).all(1))
        for j in js:
            if np.intersect1d(faces[i],faces[j]).size:
                continue
            found=False
            for source,target in ((tri[i],tri[j]),(tri[j],tri[i])):
                e1=target[1]-target[0];e2=target[2]-target[0]
                for k in range(3):
                    origin=source[k];direction=source[(k+1)%3]-origin
                    h=np.cross(direction,e2);det=np.dot(e1,h)
                    if abs(det)<1e-12:continue
                    inv=1/det;s=origin-target[0];u=inv*np.dot(s,h)
                    q=np.cross(s,e1);v=inv*np.dot(direction,q);t=inv*np.dot(e2,q)
                    if 1e-7<u and 1e-7<v and u+v<1-1e-7 and 1e-7<t<1-1e-7:
                        found=True;break
                if found:break
            if found:hits.append([int(i),int(j)])
    return hits


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--result',type=Path,required=True)
    ap.add_argument('--walker-model',type=Path,required=True)
    ap.add_argument('--smplh-model',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    if args.output_dir.exists():raise RuntimeError('refuse_existing_output')
    args.output_dir.mkdir(parents=True)
    _install_legacy_smpl_pickle_compatibility()
    import pickle
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    with args.smplh_model.open('rb') as f:asset=pickle.load(f,encoding='latin1')
    z=np.load(args.result);walker=json.loads(args.walker_model.read_text(encoding='utf-8'))
    v=z['vertices'];faces=z['faces'];joints=z['smplh_joints'];n=len(v)
    R=np.asarray(walker['rotation_left_camera_from_walker']);t=np.asarray(walker['translation_left_camera_from_walker_mm'])/1000
    if not np.allclose(R.T@R,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(R),1):raise ValueError('invalid_walker_rotation')
    nodes={k:np.asarray(p)/1000 for k,p in walker['nodes_walker_mm'].items()}
    frames=sorted(set([0,n//2,n-1]));report={'status':'engineering_geometry_audit','frames':n,'diagnostic_frames':frames,'hands':{},'source':str(args.result),'walker_source':str(args.walker_model),'limits':'static camera installation assumption; no true contact labels; self-intersection screen excludes coplanar/tangent cases'}
    dom=np.asarray(asset['weights']).argmax(1);template=np.asarray(asset['v_template']);restj=np.asarray(asset['J'])
    for side,start,wrist,color in [('left',22,20,'#cc6666'),('right',37,21,'#6688cc')]:
        # ALL finger chains, not just wrist + first index joint candidate set.
        mask=((dom>=start)&(dom<start+15))|((dom==wrist)&(np.linalg.norm(template-restj[wrist],axis=1)<.08))
        idx=np.flatnonzero(mask);handfaces=faces[mask[faces].all(1)]
        hand=z[f'hand_points_{side}'];pw=(v[:,idx]-t)@R;hw=(hand-t)@R
        ends=np.array([nodes[k] for k in walker['handle_segments'][side]])
        a,b=ends;res=capsule(pw,a,b,.016);tips=[4,8,12,16,20]
        target=z[f'mano_{side}_target_rotations'];w=z[f'mano_{side}_view_weights'];covered=w.sum(1)>0
        pose=z[f'{side}_hand_pose'];pca=z[f'{side}_hand_pca']
        if np.ptp(pca,axis=0).max()!=0:raise ValueError('pose_is_not_shared')
        bend={}
        for fi,name in enumerate(['thumb','index','middle','ring','pinky']):
            points=hand[:,1+4*fi:5+4*fi];bones=np.diff(points,axis=1)
            cos=(bones[:,:-1]*bones[:,1:]).sum(-1)/(np.linalg.norm(bones[:,:-1],axis=-1)*np.linalg.norm(bones[:,1:],axis=-1))
            angles=np.degrees(np.arccos(np.clip(cos,-1,1)))
            bend[name]={'intersegment_turn_deg':np.median(angles,axis=0).tolist(),
                        'tip_to_base_mm':stats(np.linalg.norm(points[:,-1]-points[:,0],axis=-1)*1000)}
        collisions={str(f):crossing_pairs(v[f],handfaces) for f in frames}
        from collections import Counter
        region_names=['index','middle','pinky','ring','thumb']
        labels=[]
        for face in handfaces:
            region=[region_names[(int(dom[k])-start)//3] if start<=dom[k]<start+15 else 'wrist_palm' for k in face]
            labels.append(Counter(region).most_common(1)[0][0])
        collision_regions={f:dict(Counter(' / '.join(sorted([labels[i],labels[j]])) for i,j in pairs)) for f,pairs in collisions.items()}
        item={'mesh_vertices':len(idx),'mesh_faces':len(handfaces),'pca_rms':float(np.sqrt(np.mean(pca[0]**2))),
              'pca_abs_max':float(np.abs(pca[0]).max()),'fingers':bend,
              'wrist_height_above_handle_mid_mm':stats((hw[covered,0,2]-(a[2]+b[2])/2)*1000),
              'all_hand_surface_capsule_gap_mm':stats(res[covered]*1000),
              'within_3mm_surface_fraction':float((np.abs(res[covered])<=.003).mean()),
              'penetrating_more_than_3mm_fraction':float((res[covered]<-.003).mean()),
              'fingertip_capsule_gap_mm':stats(capsule(hw[covered][:,tips],a,b,.016)*1000),
              'noncoplanar_triangle_intersections_at_selected_frames':collisions}
        item['intersection_skinning_region_pair_counts']=collision_regions
        # Identical beta/root/body/transl, straight fingers control: distinguish
        # pose-induced intersection from an inherent model/shape defect.
        import torch, smplx
        from smplx.utils import Struct
        asset_data=dict(asset)
        asset_data.update(hands_componentsl=np.eye(45),hands_componentsr=np.eye(45),
                          hands_meanl=np.zeros(45),hands_meanr=np.zeros(45))
        model=smplx.SMPLH(str(args.smplh_model),data_struct=Struct(**asset_data),
                         use_pca=False,flat_hand_mean=True,batch_size=len(frames))
        with torch.no_grad():
            control=model(betas=torch.tensor(z['betas']).expand(len(frames),-1),
                global_orient=torch.tensor(z['global_orient'][frames]),
                body_pose=torch.tensor(z['body_pose'][frames]),
                transl=torch.tensor(z['transl'][frames]),
                left_hand_pose=torch.zeros(len(frames),45),right_hand_pose=torch.zeros(len(frames),45),return_verts=True)
        item['straight_finger_control_intersection_counts']={str(f):len(crossing_pairs(control.vertices[k].numpy(),handfaces)) for k,f in enumerate(frames)}
        report['hands'][side]=item
        for f in frames:
            points=(v[f]-t)@R
            q=points[handfaces]*[1,-1,1];skeleton=hw[f]*[1,-1,1];he=ends*[1,-1,1]
            fig=plt.figure(figsize=(13,4),dpi=160)
            allp=np.concatenate([q.reshape(-1,3),he]);center=(allp.min(0)+allp.max(0))/2;span=np.ptp(allp,axis=0).max()*1.1
            for col,(elev,azim) in enumerate([(15,-70),(15,20),(80,-70)]):
                ax=fig.add_subplot(1,3,col+1,projection='3d')
                ax.add_collection3d(Poly3DCollection(q,facecolor=color,edgecolor='#555555',linewidth=.08,alpha=.8))
                ax.plot(*he.T,c='#111111',lw=8,label='handle axis (radius 16 mm)')
                # Physical radial extent at 8 axial positions, without moving hand.
                axis=(he[1]-he[0]);axis/=np.linalg.norm(axis);ref=np.array([0.,0.,1.]);u=np.cross(axis,ref);u/=np.linalg.norm(u);vv=np.cross(axis,u)
                angle=np.linspace(0,2*np.pi,24)
                for alpha in np.linspace(0,1,8):
                    circle=(1-alpha)*he[0]+alpha*he[1]+.016*(np.cos(angle)[:,None]*u+np.sin(angle)[:,None]*vv)
                    ax.plot(*circle.T,c='#111111',lw=.35)
                for fi in range(5):
                    ids=[0]+list(range(1+4*fi,5+4*fi));ax.plot(*skeleton[ids].T,c='#7b1fa2',lw=1)
                ax.set_xlim(center[0]-span/2,center[0]+span/2);ax.set_ylim(center[1]-span/2,center[1]+span/2);ax.set_zlim(center[2]-span/2,center[2]+span/2)
                ax.set_box_aspect([1,1,1]);ax.view_init(elev,azim);ax.set_xlabel('X walker (m)');ax.set_ylabel('-Y walker (m)');ax.set_zlabel('Z walker (m)')
            fig.suptitle(f'{side} fixed PCA pose | actual frame {f} | static camera-rigid geometry assumption\nlocal articulation fixed; wrist and palm transform NOT fixed to handle')
            fig.tight_layout();fig.savefig(args.output_dir/f'{side}_frame{f:04d}.png');plt.close(fig)
    (args.output_dir/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
