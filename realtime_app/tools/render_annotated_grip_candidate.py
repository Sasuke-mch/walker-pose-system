"""Render actual local grip triangles in walker coordinates, without smoothing."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--result',type=Path,required=True)
    ap.add_argument('--walker-model',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--source-result',type=Path)
    args=ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    z=np.load(args.result,allow_pickle=False)
    w=json.loads(args.walker_model.read_text(encoding='utf-8'))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.source_result:
        from scipy.spatial.transform import Rotation
        source=np.load(args.source_result,allow_pickle=False)
        report=json.loads((args.result.parent/'report.json').read_text(encoding='utf-8'))
        summary={'scope':'independent_local_grip_candidate_not_whole_body_fit','hands':{}}
        if report.get('construction_prior'):
            summary['scope']='actively_constructed_bilateral_grasp_not_observation_reconstruction'
        checked={k:z[k] for k in z.files}
        checked['accepted_for_main_fit']=np.asarray(False)
        checked['scope']=np.asarray('shared_finger_PCA_and_wrist_SE3_in_walker_frame')
        for side in ('left','right'):
            pose=z[f'{side}_hand_pose']
            pred=Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix()
            target=source[f'mano_{side}_target_rotations']
            weights=source[f'mano_{side}_view_weights']
            relative=np.swapaxes(target,-1,-2)@pred
            angles=np.degrees(Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude().reshape(target.shape[:-2]))
            aa=angles[weights>0].reshape(-1)
            delta=Rotation.from_matrix(np.swapaxes(Rotation.from_rotvec(source[f'{side}_hand_pose'][0].reshape(-1,3)).as_matrix(),-1,-2)@pred).magnitude()
            item=report['hands'][side]
            summary['hands'][side]={'geometry_status':item['status'],'feasible_starts':sum(c['accepted_geometry'] for c in item['all_starts']),'total_starts':len(item['all_starts']),'native_rotation_residual_median_deg':float(np.median(aa)),'native_rotation_residual_p95_deg':float(np.percentile(aa,95)),'rotation_change_from_previous_shared_pose_mean_deg':float(np.degrees(delta).mean()),'native_covered_frames':int((weights.sum(1)>0).sum()),'wrist_walker_m':z[f'{side}_wrist_walker_m'].tolist(),'hand_PCA_shape':list(z[f'{side}_hand_pca'].shape),'approx_manual_mask_iou':{c:item['metrics'][c]['approx_visible_mask_iou'] for c in ('left','right')},'max_handle_penetration_mm':max(0,-item['selected']['min_gap_mm']),'triangle_intersections':item['selected']['triangle_intersections'],'palm_nearest_gap_abs_mm':float(np.sqrt(item['selected']['terms'].get('palm_contact',0))*5),'wrap_prior_cost':item['selected']['terms'].get('wrap')}
            checked[f'{side}_native_observation_mask']=weights.sum(1)>0
            checked[f'{side}_accepted_geometry']=np.asarray(item['status']=='geometry_gated_candidate')
        (args.output_dir/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(args.output_dir/'fixed_grip_checked.npz',**checked)
    for side in ('left','right'):
        points=z[f'{side}_vertices_walker_m']*[1,-1,1]
        faces=z[f'{side}_faces']
        ends=np.array([w['nodes_walker_mm'][k] for k in w['handle_segments'][side]])/1000*[1,-1,1]
        axis=ends[1]-ends[0];axis/=np.linalg.norm(axis)
        u=np.cross(axis,[0,0,1]);u/=np.linalg.norm(u);v=np.cross(axis,u)
        phi=np.linspace(0,2*np.pi,48)
        centers=np.linspace(ends[0],ends[1],16)
        cylinder=centers[:,None]+.016*(np.cos(phi)[None,:,None]*u+np.sin(phi)[None,:,None]*v)
        center=(np.minimum(points.min(0),ends.min(0))+np.maximum(points.max(0),ends.max(0)))/2
        span=np.max(np.ptp(np.concatenate([points,ends]),axis=0))*1.1
        fig=plt.figure(figsize=(14,4.8),dpi=160)
        for i,(elev,azim) in enumerate([(15,-70),(15,20),(80,-70)]):
            ax=fig.add_subplot(1,3,i+1,projection='3d')
            ax.add_collection3d(Poly3DCollection(points[faces],facecolor='#d8a182',edgecolor='#333333',linewidth=.08))
            ax.plot_surface(cylinder[:,:,0],cylinder[:,:,1],cylinder[:,:,2],color='#333333',alpha=.95)
            ax.set_xlim(center[0]-span/2,center[0]+span/2)
            ax.set_ylim(center[1]-span/2,center[1]+span/2)
            ax.set_zlim(center[2]-span/2,center[2]+span/2)
            ax.set_box_aspect((1,1,1));ax.view_init(elev,azim)
            ax.set_xlabel('X walker (m)');ax.set_ylabel('-Y walker (m)');ax.set_zlabel('Z walker (m)')
        status=report['hands'][side]['status'] if args.source_result else 'candidate_not_validated'
        provenance='actively constructed grasp' if args.source_result and report.get('construction_prior') else 'shared PCA + walker-relative wrist SE3'
        fig.suptitle(f'{side} {provenance} | {status}\nActual SMPL-H hand triangles; radius 16 mm assumption; upper-limb IK not solved')
        fig.tight_layout();fig.savefig(args.output_dir/f'{side}_three_views.png');plt.close(fig)


if __name__=='__main__':main()
