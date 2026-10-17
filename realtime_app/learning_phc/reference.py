"""Causal, explicit SMPL-H -> author neutral SMPL reference retargeting.

Only q_i, the frozen shape convention, scene_i and q_{i-1} enter reference_i.
No full-clip height correction, time resampling, smoothing or future derivative.
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .contract import BASE_ROTATION, NAMES, SMPL_INDICES

BODY_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def reference_pd_action(action, reference_rotation, parents, scale, weight):
    """Move PD targets toward current reference local rotations on SO(3).

    This changes actuator commands only. It neither overwrites simulation state
    nor reinterprets a frozen absolute-action checkpoint as a residual policy.
    Weight 1 is a non-neural reference PD baseline; 0 is exactly original PHC.
    """
    if (np.shape(action) != (69,) or np.shape(reference_rotation) != (24, 3, 3)
            or np.shape(scale) != (69,) or len(parents) != 24
            or not 0 <= weight <= 1
            or not all(np.isfinite(x).all() for x in (action, reference_rotation, scale))
            or np.any(np.asarray(scale) <= 0)):
        raise ValueError('finite PHC action, reference rotations, positive scale and weight in [0,1] required')
    if weight == 0:
        return action.copy()
    local = np.stack([reference_rotation[parents[j]].T @ reference_rotation[j]
                      for j in range(1, 24)])
    policy_target = Rotation.from_rotvec((np.clip(action, -1, 1)*scale).reshape(23, 3))
    difference = policy_target.inv()*Rotation.from_matrix(local)
    target = policy_target*Rotation.from_rotvec(weight*difference.as_rotvec())
    return np.clip(target.as_rotvec().ravel()/scale, -1, 1)


def retarget(time_s, root_position, root_rotation, body_rotation, hand_rotation,
             available, segment, xml):
    time_s = np.asarray(time_s)
    if time_s.ndim != 1 or not np.isfinite(time_s).all() or np.any(np.diff(time_s) <= 0):
        raise ValueError('strictly increasing finite capture time required')
    count = len(time_s)
    if (np.shape(root_position) != (count, 3) or np.shape(root_rotation) != (count, 3, 3)
            or np.shape(body_rotation) != (count, 21, 3, 3)
            or np.shape(hand_rotation) != (2, 3, 3)
            or np.shape(available) != (count,) or np.shape(segment) != (count,)):
        raise ValueError('SMPL-H reference shapes do not agree')
    tree = ET.parse(xml)
    offsets = [np.zeros(3)]
    parents = [-1]
    parent_lookup = {child.get('name'): parent.get('name') for parent in tree.findall('.//body')
                     for child in parent.findall('body')}
    for name in NAMES[1:]:
        offsets.append(np.fromstring(tree.find(f'.//body[@name="{name}"]').get('pos'), sep=' '))
        parents.append(NAMES.index(parent_lookup[name]))
    offsets = np.asarray(offsets)
    positions = np.full((count, 24, 3), np.nan)
    rotations = np.full((count, 24, 3, 3), np.nan)
    velocity = np.full_like(positions, np.nan)
    for i in range(count):
        if not available[i]:
            continue
        if not all(np.isfinite(a).all() for a in (root_position[i], root_rotation[i], body_rotation[i])):
            raise ValueError('available source frame contains nonfinite model parameters')
        global_smpl = [root_rotation[i]]
        for j in range(1, 22):
            global_smpl.append(global_smpl[BODY_PARENTS[j]] @ body_rotation[i, j-1])
        # The released policy has two coarse terminal hand links. Use the first
        # native SMPL-H finger-base orientation only to initialize those links;
        # this does not infer or control all finger articulation.
        global_smpl += [global_smpl[20] @ hand_rotation[0], global_smpl[21] @ hand_rotation[1]]
        rotations[i] = np.asarray(global_smpl)[SMPL_INDICES] @ BASE_ROTATION.T
        positions[i, 0] = root_position[i]
        for j in range(1, 24):
            positions[i, j] = positions[i, parents[j]] + rotations[i, parents[j]] @ offsets[j]
        velocity[i] = 0.
        if i and available[i-1] and segment[i] == segment[i-1]:
            velocity[i] = (positions[i]-positions[i-1])/(time_s[i]-time_s[i-1])
    return dict(timestamp_s=time_s, reference_position_m=positions,
                reference_rotation=rotations, reference_velocity_m_s=velocity,
                available=np.asarray(available, bool), segment_id=np.asarray(segment))


def matched_body_xml(source, xml, output):
    """Match source rest joints; preserve the author's collision shapes and PD.

    This corrects bone offsets only. It does not certify collision skin or mass.
    """
    import torch
    from .analyze import load_model
    if output.exists():
        raise FileExistsError(output)
    model = load_model()
    with np.load(source,allow_pickle=False) as z:
        beta=torch.tensor(z['betas'],dtype=torch.float32)
    rest=model.v_template+torch.einsum('bl,vcl->bvc',beta,model.shapedirs)[0]
    joints=(model.J_regressor@rest).detach().numpy()
    # SMPL-H first finger bases replace the two coarse SMPL hand links.
    indices=SMPL_INDICES.copy()
    indices[indices==23]=37
    tree=ET.parse(xml)
    for parent in tree.findall('.//body'):
        for child in parent.findall('body'):
            a,b=NAMES.index(parent.get('name')),NAMES.index(child.get('name'))
            offset=BASE_ROTATION@(joints[indices[b]]-joints[indices[a]])
            child.set('pos',' '.join(f'{x:.10g}' for x in offset))
    output.parent.mkdir(parents=True,exist_ok=True)
    tree.write(output,encoding='unicode')


def prepare(source, scene, xml, output, match_shape=False):
    if output.exists():
        raise FileExistsError(output)
    if match_shape:
        matched=output.with_suffix('.xml')
        matched_body_xml(source,xml,matched)
        xml=matched
    # The trusted local scene export stores Stage labels as an object array.
    with np.load(source, allow_pickle=False) as fitted, np.load(scene, allow_pickle=True) as world:
        ids = fitted['pair_id']
        if ids.tolist() != list(range(448)):
            raise ValueError('expected all448 source IDs in order, including unavailable states')
        time_s = fitted['timestamp_s']
        np.testing.assert_allclose(time_s, world['pair_timestamp_sec'], atol=1e-7, rtol=0)
        rotation, translation = world['rotation_ground_from_left'], world['translation_ground_from_left_mm']/1000
        # Source stores native joints in camera coordinates, but model root
        # orientations in both camera and world. Use the actual pelvis joint,
        # not SMPL transl, because SMPL rotates about shaped rest pelvis J0.
        pelvis = np.einsum('nij,nj->ni', rotation, fitted['smplh_joints'][:, 0])+translation
        root_rotation = fitted['world_root_rotation']
        np.testing.assert_allclose(root_rotation, rotation @ fitted['root_rotation'], atol=1e-6, rtol=0)
        hand = np.stack([Rotation.from_rotvec(fitted[key][0].reshape(15, 3)[0]).as_matrix()
                         for key in ('left_hand_pose', 'right_hand_pose')])
        result = retarget(time_s, pelvis, root_rotation, fitted['body_rotations'], hand,
                          fitted['pose_available'], fitted['segment_id'], xml)
        result.update(pair_id=ids, betas=fitted['betas'], source_pelvis_world_m=pelvis,
                      camera_rotation=rotation, camera_translation_m=translation,
                      stage=world['stage'].astype(str))
        parameters=json.loads(source.with_name('run_parameters.json').read_text(encoding='utf-8'))
        from pose_app.body_observations import load_ordered_predictions as _load_predictions
        from pose_app.fisheye_camera import load_stereo_fisheye
        observations=np.stack([_load_predictions(Path(parameters['inputs'][side+'_predictions']),side)
                               for side in ('left','right')],1)[:,:,5:17]
        mapping=np.array([NAMES.index(name) for name in ('L_Shoulder','R_Shoulder','L_Elbow','R_Elbow',
            'L_Wrist','R_Wrist','L_Hip','R_Hip','L_Knee','R_Knee','L_Ankle','R_Ankle')])
        world_coco=np.einsum('nij,nvj->nvi',rotation,fitted['predicted_coco'][:,5:17])+translation[:,None]
        offsets=np.einsum('nkji,nkj->nki',result['reference_rotation'][:,mapping],
                           world_coco-result['reference_position_m'][:,mapping])
        calibration=load_stereo_fisheye(Path(parameters['inputs']['calibration_dir']))
        result.update(feedback_pixels=observations,feedback_coco_body=mapping,feedback_coco_local_m=offsets,
                      K0=calibration.K0,D0=calibration.D0,K1=calibration.K1,D1=calibration.D1,
                      R01=calibration.R_cam0_to_cam1,T01_m=calibration.T_cam0_to_cam1_mm/1000)
        heels=np.einsum('nij,nvj->nvi',rotation,fitted['vertices'][:,[3387,6787]])+translation[:,None]
        result['heel_local_m']=np.einsum('nkji,nkj->nki',result['reference_rotation'][:,[3,7]],
                                          heels-result['reference_position_m'][:,[3,7]])
        import torch
        from .analyze import load_model
        model=load_model()
        beta=torch.tensor(fitted['betas'],dtype=torch.float32)
        rest=model.v_template+torch.einsum('bl,vcl->bvc',beta,model.shapedirs)[0]
        result.update(heel_rest_joints_m=(model.J_regressor@rest).detach().numpy(),
            heel_rest_vertices_m=rest[[3387,6787]].detach().numpy(),
            heel_skin_weights=model.lbs_weights[[3387,6787]].detach().numpy(),
            heel_pose_dirs=model.posedirs.reshape(-1,6890,3)[:,[3387,6787]].detach().numpy(),
            heel_smpl_parents=model.parents.detach().numpy(),
            heel_hand_rotation=np.concatenate([Rotation.from_rotvec(fitted[k][0].reshape(15,3)).as_matrix()
                                               for k in ('left_hand_pose','right_hand_pose')]))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    metadata = dict(source=str(source.resolve()), scene=str(scene.resolve()), xml=str(xml.resolve()),
        output=str(output.resolve()), frames=448, duration_s=float(time_s[-1]-time_s[0]),
        source_model='complete causal SMPL-H state estimator; frozen full-clip pre-calibrated beta',
        scene_semantics='frozen same-source Stage conditional ground, nominal0.5s publication lag; not independent world truth',
        retarget='source-beta rest joint offsets;author collision shapes retained' if match_shape else 'preserve current world root/pelvis and joint rotations; author neutral body lengths',
        terminal_hands='coarse2 links initialized from first finger-base rotations, not full hands',
        reference_velocity='backward difference within segment, zero on initialization/reinitialization',
        future_reference_frames=0, height_fix=False, smoothing=False,
        accuracy='retargeted reference is not independent3D truth; beta calibration and conditional scene disclosed')
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata), flush=True)


class SMPLHHeelSurface:
    """Exact two-vertex SMPL-H LBS using frozen shape/hands, no mesh surrogate."""
    def __init__(self,reference):
        self.rest=reference['heel_rest_joints_m']
        self.vertices=reference['heel_rest_vertices_m']
        self.weights=reference['heel_skin_weights']
        self.pose_dirs=reference['heel_pose_dirs']
        self.parents=reference['heel_smpl_parents']
        self.hands=reference['heel_hand_rotation']

    def points(self,pelvis,native_rotation):
        global_body=native_rotation[np.argsort(SMPL_INDICES)[:22]]@BASE_ROTATION
        local=np.empty((52,3,3))
        local[0]=global_body[0]
        for j in range(1,22):
            local[j]=global_body[self.parents[j]].T@global_body[j]
        local[22:]=self.hands
        global_rotation=np.empty_like(local)
        joints=np.empty((52,3))
        global_rotation[0],joints[0]=local[0],pelvis
        for j in range(1,52):
            parent=self.parents[j]
            global_rotation[j]=global_rotation[parent]@local[j]
            joints[j]=joints[parent]+global_rotation[parent]@(self.rest[j]-self.rest[parent])
        posed=self.vertices+np.einsum('p,pvc->vc',(local[1:]-np.eye(3)).ravel(),self.pose_dirs)
        translation=joints-np.einsum('nij,nj->ni',global_rotation,self.rest)
        return np.einsum('vj,jcd,vd->vc',self.weights,global_rotation,posed)+self.weights@translation


class PixelReferenceFeedback:
    """One bounded raw-pixel GN step on the controller reference, never on state.

    Uses source current-frame COCO-to-body offsets as a local approximation.
    Linearization, line search and returned target share the source reference
    base. A correction around simulated state cannot be applied to another base.
    Final evaluation still decodes the real SMPL-H, independently of this proxy.
    """
    def __init__(self,reference,xml,gain=1.):
        import torch
        self.torch=torch
        self.reference=reference
        self.gain=gain
        tree=ET.parse(xml)
        lookup={child.get('name'):parent.get('name') for parent in tree.findall('.//body')
                for child in parent.findall('body')}
        self.parents=[-1]+[NAMES.index(lookup[n]) for n in NAMES[1:]]
        self.offset=torch.tensor(np.array([np.zeros(3)]+[np.fromstring(tree.find(f'.//body[@name="{n}"]').get('pos'),sep=' ') for n in NAMES[1:]]),dtype=torch.float64)
        self.scale=torch.tensor([.03]*3+[.12]*72,dtype=torch.float64)

    def fk(self,delta,root,local):
        torch=self.torch
        vectors=(delta[3:]*self.scale[3:]).reshape(24,3)
        x,y,z=vectors.unbind(-1)
        zero=torch.zeros_like(x)
        skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(24,3,3)
        angle=torch.sqrt((vectors*vectors).sum(-1).clamp(min=1e-24))
        eye=torch.eye(3,dtype=torch.float64).expand(24,-1,-1)
        exp=eye+torch.sinc(angle[:,None,None]/torch.pi)*skew+.5*torch.sinc(angle[:,None,None]/(2*torch.pi))**2*(skew@skew)
        q=local@exp
        p=[root+delta[:3]*self.scale[:3]]
        r=[q[0]]
        for j in range(1,24):
            a=self.parents[j]
            p.append(p[a]+r[a]@self.offset[j])
            r.append(r[a]@q[j])
        return torch.stack(p),torch.stack(r)

    def correct(self,i):
        torch=self.torch
        ref=self.reference
        from pose_app.fisheye_camera import fisheye_project_torch
        mapping=ref['feedback_coco_body']
        position=ref['reference_position_m'][i]
        rotation=ref['reference_rotation'][i]
        local=np.array([rotation[0]]+[rotation[self.parents[j]].T@rotation[j] for j in range(1,24)])
        root=torch.tensor(position[0],dtype=torch.float64)
        local=torch.tensor(local,dtype=torch.float64)
        offsets=torch.tensor(ref['feedback_coco_local_m'][i],dtype=torch.float64)
        camera_r=torch.tensor(ref['camera_rotation'][i],dtype=torch.float64)
        camera_t=torch.tensor(ref['camera_translation_m'][i],dtype=torch.float64)
        r01=torch.tensor(ref['R01'],dtype=torch.float64)
        t01=torch.tensor(ref['T01_m'],dtype=torch.float64)
        observed=ref['feedback_pixels'][i]
        valid=np.isfinite(observed).all(-1)&(observed[...,2]>=.2)&(observed[...,2]<=1)
        valid&=(observed[...,0]>=0)&(observed[...,0]<1920)&(observed[...,1]>=0)&(observed[...,1]<1080)
        valid[:,9]=False # COCO14 right knee remains evaluation-only.
        def project(delta):
            p,r=self.fk(delta,root,local)
            coco=p[mapping]+torch.einsum('nij,nj->ni',r[mapping],offsets)
            left=(coco-camera_t)@camera_r
            right=left@r01.T+t01
            return torch.stack([fisheye_project_torch(left,ref['K0'],ref['D0']),
                fisheye_project_torch(right,ref['K1'],ref['D1'])])
        zero=torch.zeros(75,dtype=torch.float64,requires_grad=True)
        predicted=project(zero).detach().numpy()
        valid&=np.isfinite(predicted).all(-1)
        if valid.sum()<8:
            return ref['reference_position_m'][i],ref['reference_rotation'][i],dict(status='insufficient_pixels',used=int(valid.sum()))
        residual=(observed[...,:2]-predicted)[valid]
        weight=np.minimum(1,25/np.maximum(np.linalg.norm(residual,axis=-1),1e-12))*observed[...,2][valid]
        jac=torch.autograd.functional.jacobian(project,zero,vectorize=True).detach().numpy()[valid].reshape(-1,75)
        w=np.repeat(weight,2)/25**2
        delta=np.linalg.solve(jac.T@(w[:,None]*jac)+np.eye(75),jac.T@(w*residual.ravel()))
        delta/=max(1,np.linalg.norm(delta[:3]),np.linalg.norm(delta[3:].reshape(24,3),axis=1).max())
        def cost(pred):
            e=np.linalg.norm((observed[...,:2]-pred)[valid],axis=-1)
            return float(np.sum(np.where(e<=25,.5*e**2,25*(e-12.5))*observed[...,2][valid]))
        before=cost(predicted)
        accepted=False
        for factor in (1.,.5,.25):
            trial=delta*factor*self.gain
            after=cost(project(torch.tensor(trial,dtype=torch.float64)).detach().numpy())
            if np.isfinite(after) and after<before:
                delta=trial;accepted=True;break
        if not accepted:
            return ref['reference_position_m'][i],ref['reference_rotation'][i],dict(status='no_pixel_descent',used=int(valid.sum()))
        p,r=self.fk(torch.tensor(delta,dtype=torch.float64),root,local)
        return p.detach().numpy(),r.detach().numpy(),dict(status='bounded_pixel_feedback',used=int(valid.sum()),proxy_cost_before=before,proxy_cost_after=after)


def dynamic_probe(xml, motion, output):
    """Known small arm motion on the author body; interface test, no human GT."""
    import joblib
    if output.exists():
        raise FileExistsError(output)
    initial = next(iter(joblib.load(motion).values()))
    rotations = Rotation.from_quat(initial['pose_quat_global'][0]).as_matrix()
    tree = ET.parse(xml)
    parent_lookup = {child.get('name'): parent.get('name') for parent in tree.findall('.//body')
                     for child in parent.findall('body')}
    parents = [-1]+[NAMES.index(parent_lookup[name]) for name in NAMES[1:]]
    offsets = np.array([np.zeros(3)]+[np.fromstring(tree.find(f'.//body[@name="{name}"]').get('pos'),sep=' ')
                                     for name in NAMES[1:]])
    local = np.array([rotations[parents[j]].T@rotations[j] for j in range(1,24)])
    time_s = np.arange(301)/30
    positions = np.zeros((301,24,3))
    global_rotation = np.empty((301,24,3,3))
    for i,t in enumerate(time_s):
        q = local.copy()
        angle = np.deg2rad(15)*(1-np.cos(2*np.pi*t/4))/2
        for name,sign in (('L_Shoulder',1),('R_Shoulder',-1)):
            j=NAMES.index(name)
            q[j-1] = q[j-1]@Rotation.from_rotvec([sign*angle,0,0]).as_matrix()
        positions[i,0]=np.asarray(initial['root_trans_offset'][0])
        global_rotation[i,0]=rotations[0]
        for j in range(1,24):
            global_rotation[i,j]=global_rotation[i,parents[j]]@q[j-1]
            positions[i,j]=positions[i,parents[j]]+global_rotation[i,parents[j]]@offsets[j]
    velocity=np.zeros_like(positions)
    velocity[1:]=np.diff(positions,axis=0)*30
    np.savez_compressed(output,pair_id=np.arange(301),timestamp_s=time_s,
        reference_position_m=positions,reference_rotation=global_rotation,
        reference_velocity_m_s=velocity,available=np.ones(301,bool),segment_id=np.zeros(301,int),
        camera_rotation=np.broadcast_to(np.eye(3),(301,3,3)),camera_translation_m=np.zeros((301,3)))
    output.with_suffix('.json').write_text(json.dumps(dict(scope='synthetic author-body dynamic interface probe; not paper reproduction or human accuracy',
        seconds=10,fps=30,shoulder_excursion_deg=15,source_motion=str(motion),xml=str(xml)),indent=2),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--dynamic-probe',type=Path,help='official standing motion used to initialize known arm motion')
    parser.add_argument('--match-shape',action='store_true',help='match source rest bone offsets, retain collision shapes; isolated geometry contrast')
    parser.add_argument('--xml', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.dynamic_probe:
        dynamic_probe(args.xml,args.dynamic_probe,args.output)
    else:
        if args.source is None or args.scene is None:
            parser.error('source and scene required for human reference')
        prepare(args.source, args.scene, args.xml, args.output,args.match_shape)


if __name__ == '__main__':
    main()
