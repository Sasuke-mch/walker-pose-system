"""Fit one PCA hand and wrist SE3 per side using annotated stereo silhouettes.

Independent local grip candidate, NOT a new whole-body fit. No virtual WiLoR
camera or old dynamic ground enters this solver. Retain all starts and gates.
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_app.smpl_coco_observation import _install_legacy_smpl_pickle_compatibility
from pose_app.fisheye_camera import load_stereo_fisheye, fisheye_project_numpy, fisheye_project_torch
from pose_app.smplh_hand_observation import hand21
from pose_app.wilor_mano_prior import rotation_pose_loss
from audit_fixed_mano_grip import crossing_pairs


def load_masks(path, remap_upper_right=False):
    label = json.loads(path.read_text(encoding='utf-8'))
    h, w = label['imageHeight'], label['imageWidth']
    if (w, h) != (1080, 1920):
        raise ValueError('unexpected_annotation_size')
    masks = {s: np.zeros((h, w), np.uint8) for s in ('left', 'right')}
    audit = []
    for i, shape in enumerate(label['shapes']):
        name = shape['label']
        if remap_upper_right and i == 1:
            if name != 'right_hand' or np.mean(np.asarray(shape['points'])[:, 0]) <= w/2:
                raise ValueError('remap_precondition_failed')
            name = 'left_hand'
        if name not in ('left_hand', 'right_hand'):
            audit.append({'shape': i, 'reason': 'not_a_hand_label', 'label': name})
            continue
        if shape['shape_type'] != 'polygon':
            raise ValueError('polygon_required')
        cv2.fillPoly(masks[name[:-5]], [np.rint(shape['points']).astype(np.int32)], 1)
        audit.append({'shape': i, 'original_label': shape['label'], 'used_label': name})
    if any(not m.any() for m in masks.values()):
        raise ValueError('missing_hand_mask')
    return masks, audit


def upright_projection(uv, camera):
    # Original raw images are 1920x1080; label images are rotated 1080x1920.
    import torch
    return torch.stack((uv[..., 1], 1919-uv[..., 0]), -1) if camera == 'left' else torch.stack((1079-uv[..., 1], uv[..., 0]), -1)


def capsule_gap(p, a, b, radius=.016):
    import torch
    axis = b-a
    u = ((p-a)*axis).sum(-1)/(axis*axis).sum()
    near = a+u.clamp(0, 1)[..., None]*axis
    return torch.linalg.vector_norm(p-near, dim=-1)-radius


def handle_occluded(p, a, b, radius=.016):
    """Ray/infinite-cylinder first hit, restricted to finite axial segment.

    Camera origin is zero. Caps are not included; endpoint silhouettes remain
    conservative. This is static model occlusion, not measured visibility.
    """
    import torch
    axis = (b-a)/torch.linalg.vector_norm(b-a)
    perp = p-(p*axis).sum(-1, keepdim=True)*axis
    base = -a+torch.dot(a, axis)*axis
    A = (perp*perp).sum(-1).clamp_min(1e-12)
    B = 2*(perp*base).sum(-1)
    C = (base*base).sum()-radius*radius
    disc = B*B-4*A*C
    ray_t = (-B-torch.sqrt(disc.clamp_min(0)))/(2*A)
    hit = ray_t[:, None]*p
    length = torch.linalg.vector_norm(b-a)
    along = ((hit-a)*axis).sum(-1)
    return (disc>0)&(ray_t>0)&(ray_t<.999)&(along>=0)&(along<=length)


def render_mask(uv, faces, size):
    mask = np.zeros(size, np.uint8)
    finite = np.isfinite(uv).all(-1)
    for f in faces:
        if finite[f].all():
            cv2.fillConvexPoly(mask, np.clip(np.rint(uv[f]), -5000, 5000).astype(np.int32), 1)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', type=Path, required=True)
    ap.add_argument('--walker-model', type=Path, required=True)
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--remap-left-shape1', action='store_true')
    ap.add_argument('--starts', type=int, default=6)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--seed', type=int, default=20261001)
    ap.add_argument('--initial-grip', type=Path)
    ap.add_argument('--visible-skin-target', action='store_true', help='weak RGB skin cue within manual polygons; never a human per-pixel truth')
    ap.add_argument('--wrap-prior', action='store_true', help='known enclosing-grip hypothesis; palm surface and opposing thumb/finger regions')
    ap.add_argument('--orientation-noise',type=float,default=.28)
    ap.add_argument('--image-weight',type=float,default=1.)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise ValueError('refuse_existing_output')
    args.output_dir.mkdir(parents=True)
    _install_legacy_smpl_pickle_compatibility()
    import torch
    import torch.nn.functional as F
    import smplx
    from smplx.lbs import batch_rodrigues
    from smplx.utils import Struct
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)
    root = Path(__file__).resolve().parents[2]
    z = np.load(args.result, allow_pickle=False)
    initial_grip = np.load(args.initial_grip,allow_pickle=False) if args.initial_grip else None
    walker = json.loads(args.walker_model.read_text(encoding='utf-8'))
    Rc = tensor(walker['rotation_left_camera_from_walker'])
    tc = tensor(np.asarray(walker['translation_left_camera_from_walker_mm'])/1000)
    if not torch.allclose(Rc.T@Rc, torch.eye(3, device=device), atol=1e-5):
        raise ValueError('walker_rotation_invalid')
    cal = load_stereo_fisheye(root/'realtime_app/calibration/results')
    R01 = tensor(cal.R_cam0_to_cam1)
    t01 = tensor(cal.T_cam0_to_cam1_mm/1000)
    masks, annotation_audit = {}, {}
    for camera in ('left', 'right'):
        masks[camera], annotation_audit[camera] = load_masks(args.annotations/'labels'/f'{camera}_pair_0000.json', args.remap_left_shape1 and camera=='left')
    with (root/'third_party/WiLoR/mano_data/models/SMPLH_male.pkl').open('rb') as f:
        asset = pickle.load(f, encoding='latin1')
    components, means, scales = {}, {}, {}
    for side in ('left', 'right'):
        with (root/f'third_party/WiLoR/mano_data/models/MANO_{side.upper()}.pkl').open('rb') as f:
            mano = pickle.load(f, encoding='latin1')
        components[side] = tensor(mano['hands_components'][:12])
        means[side] = tensor(mano['hands_mean'])
        scales[side] = torch.linalg.vector_norm(components[side], dim=1)
    data = dict(asset)
    data.update(hands_componentsl=np.eye(45), hands_componentsr=np.eye(45), hands_meanl=np.zeros(45), hands_meanr=np.zeros(45))
    model = smplx.SMPLH(str(root/'third_party/WiLoR/mano_data/models/SMPLH_male.pkl'), data_struct=Struct(**data), use_pca=False, flat_hand_mean=True).to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    faces = np.asarray(asset['f'])
    dom = np.asarray(asset['weights']).argmax(1)
    rest = np.asarray(asset['v_template'])
    restj = np.asarray(asset['J'])
    zero = torch.zeros((1, 45), device=device)
    bodyzero = torch.zeros((1, 63), device=device)
    beta = tensor(z['betas'])
    all_report = {'status': 'local_grip_candidate_only', 'source_result': str(args.result.resolve()), 'walker_source': str(args.walker_model.resolve()), 'args': {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, 'annotation_audit': annotation_audit, 'assumption': 'shared_finger_PCA_and_wrist_SE3_in_walker_frame_all_video', 'body_optimized': False, 'shape_optimized': False, 'radius_m': .016, 'limits': 'two annotated views of one frame; static coarse installation; no external 3D truth; local neutral-body hand surface; upper-limb IK is not solved', 'hands': {}}
    export = {}
    for side, start, wrist in [('left',22,20), ('right',37,21)]:
        use = ((dom>=start)&(dom<start+15))|((dom==wrist)&(np.linalg.norm(rest-restj[wrist],axis=1)<.08))
        idx = np.flatnonzero(use)
        hf = faces[use[faces].all(1)]
        local_index = np.full(6890, -1)
        local_index[idx] = np.arange(len(idx))
        local_faces = local_index[hf]
        groups = [np.flatnonzero((dom[idx]>=start+3*i)&(dom[idx]<start+3*i+3)) for i in range(5)]
        palm_center = .5*(restj[wrist]+restj[[start,start+3,start+6,start+9]].mean(0))
        palm_group = np.flatnonzero((dom[idx]==wrist)&(np.linalg.norm(rest[idx]-palm_center,axis=1)<.03))
        if len(palm_group)<10:raise ValueError('palm_patch_unavailable')
        # Distal finger regions; template skinning identity is not a pad label.
        targets = tensor(z[f'mano_{side}_target_rotations'])
        weights = tensor(z[f'mano_{side}_view_weights'])
        z0 = tensor(z[f'{side}_hand_pca'][0:1])
        ends = tensor([np.asarray(walker['nodes_walker_mm'][k])/1000 for k in walker['handle_segments'][side]])
        a,b = ends
        # Reproduce the source wrist coordinate frame through its FK tree.
        rotations = Rotation.from_rotvec(np.concatenate([z['global_orient'][0:1],z['body_pose'][0].reshape(21,3)])).as_matrix()
        cumulative = []
        for j, rot in enumerate(rotations):
            parent = int(model.parents[j])
            cumulative.append(rot if parent<0 else cumulative[parent]@rot)
        R0 = Rc.T@tensor(cumulative[wrist])
        t0 = (tensor(z['smplh_joints'][0,wrist])-tc)@Rc
        image_data = []
        for camera in ('left','right'):
            mask = masks[camera][side]
            target_mask = mask.copy()
            if args.visible_skin_target:
                source_image = cv2.imread(str(args.annotations/'images'/f'{camera}_pair_0000.png'))
                hsv = cv2.cvtColor(source_image,cv2.COLOR_BGR2HSV)
                skin = ((hsv[:,:,0]<25)|(hsv[:,:,0]>160))&(hsv[:,:,1]>35)&(hsv[:,:,2]>40)
                target_mask = mask*skin.astype(np.uint8)
                if target_mask.sum()<50:raise ValueError('insufficient_visible_skin_cue')
                cv2.imwrite(str(args.output_dir/f'{side}_{camera}_weak_skin_target.png'),target_mask*255)
            ys,xs = np.nonzero(target_mask)
            samples = np.stack([xs,ys],-1)[::max(1,len(xs)//350)][:350]
            dist = cv2.distanceTransform(1-mask, cv2.DIST_L2,5)
            image_data.append((camera,tensor(samples),tensor(dist)[None,None],mask))
        def forward(coeff, rotvec, translation):
            pose = means[side][None]+(coeff/scales[side][None])@components[side]
            out = model(betas=beta,body_pose=bodyzero,global_orient=torch.zeros((1,3),device=device),transl=torch.zeros((1,3),device=device),left_hand_pose=pose if side=='left' else zero,right_hand_pose=pose if side=='right' else zero,return_verts=True)
            local = out.vertices[0,idx]-out.joints[0,wrist]
            localhand = hand21(out.joints,out.vertices,side)[0]-out.joints[0,wrist]
            R = batch_rodrigues(rotvec[None])[0]
            return pose, local, local@R.T+translation, localhand@R.T+translation
        def project(q, camera):
            pc = q@Rc.T+tc
            ec = ends@Rc.T+tc
            if camera=='right':
                pc = pc@R01.T+t01
                ec = ec@R01.T+t01
            uv = upright_projection(fisheye_project_torch(pc, cal.K0 if camera=='left' else cal.K1,cal.D0 if camera=='left' else cal.D1),camera)
            return uv,handle_occluded(pc,ec[0],ec[1]).detach()
        active_pairs = torch.empty((0,2),dtype=torch.long,device=device)
        face_tensor = torch.as_tensor(local_faces,dtype=torch.long,device=device)
        def objective(coeff,rv,tr):
            pose,local,q,h = forward(coeff,rv,tr)
            image_loss = q.sum()*0
            for camera,samples,dist,mask in image_data:
                uv,occ = project(q,camera)
                # One-way annotated visible coverage tolerates fingers hidden by rail.
                if args.visible_skin_target:
                    visible_support = (~occ)&(uv[:,0]>=0)&(uv[:,0]<=1079)&(uv[:,1]>=0)&(uv[:,1]<=1919)
                    distances = torch.cdist(samples,uv[visible_support] if visible_support.any() else uv).amin(1)
                else:
                    distances = torch.cdist(samples,uv).amin(1)
                coverage = F.smooth_l1_loss(distances/15,torch.zeros_like(distances))
                grid = torch.stack([uv[:,0]/1079*2-1,uv[:,1]/1919*2-1],-1)[None,None]
                outside = F.grid_sample(dist,grid,align_corners=True,padding_mode='border').reshape(-1)
                bounds = (uv[:,0]>=0)&(uv[:,0]<=1079)&(uv[:,1]>=0)&(uv[:,1]<=1919)
                visible = bounds&(~occ)
                outside_loss = F.smooth_l1_loss(outside[visible]/15,torch.zeros_like(outside[visible])) if visible.any() else q.sum()*0+10
                image_loss = image_loss+coverage+.5*outside_loss
            gaps = capsule_gap(q,a,b)
            pen = (F.relu(-gaps-.0005)/.003).square()
            penetration = pen.mean()+pen.amax()
            contact = q.sum()*0
            for g in groups:
                # Minimum over region, not all surface vertices driven onto rod.
                contact = contact+(gaps[g].square().amin()/.005**2)
            contact = contact/5
            wrap = q.sum()*0
            palm_contact = q.sum()*0
            if args.wrap_prior:
                palm_contact = gaps[palm_group].square().amin()/.005**2
                contact_points = torch.stack([q[g[torch.argmin(gaps[g].square())]] for g in groups])
                axis = F.normalize(b-a,dim=0)
                radial = contact_points-a-((contact_points-a)*axis).sum(-1,keepdim=True)*axis
                radial = F.normalize(radial,dim=-1)
                other = F.normalize(radial[:4].mean(0),dim=0)
                wrap = F.relu(torch.dot(radial[4],other)-.2).square()
            # Inter-finger bone capsule surrogate; exact triangles are gated later.
            self_loss = q.sum()*0
            chains = [h[1+4*i:5+4*i] for i in range(5)]
            cloud = []
            for chain in chains:
                seg = []
                for j in range(3):
                    alphas = torch.linspace(.35 if j==0 else 0,1,7,device=device)
                    seg.append(chain[j][None]*(1-alphas[:,None])+chain[j+1][None]*alphas[:,None])
                cloud.append(torch.cat(seg))
            for i in range(5):
                for j in range(i+1,5):
                    d = torch.cdist(cloud[i],cloud[j])
                    interfere=(F.relu(.009-d)/.003).square()
                    self_loss = self_loss+interfere.mean()+interfere.amax()
            prior = rotation_pose_loss(pose,targets,weights)
            mesh_collision = q.sum()*0
            if len(active_pairs):
                tri = local[face_tensor]
                first, second = tri[active_pairs[:,0]],tri[active_pairs[:,1]]
                def plane_separation(source, target):
                    normal = torch.cross(target[:,1]-target[:,0],target[:,2]-target[:,0],dim=-1)
                    normal = F.normalize(normal,dim=-1,eps=1e-10)
                    signed = ((source-target[:,0,None])*normal[:,None]).sum(-1)
                    # Move the source triangle entirely to either side of the
                    # other plane. Only exact screened intersections are active.
                    positive = (F.relu(.0007-signed)/.002).square().mean(1)
                    negative = (F.relu(.0007+signed)/.002).square().mean(1)
                    return torch.minimum(positive,negative)
                mesh_collision = (plane_separation(first,second)+plane_separation(second,first)).mean()
            reg = (coeff-z0).square().mean()
            # Weak wrist regularizer only; source absolute wrist is not truth.
            anchor = ((tr-t0)/.15).square().mean()
            left_camera=q@Rc.T+tc
            right_camera=left_camera@R01.T+t01
            depth=(F.relu(.01-left_camera[:,2])/.01).square().mean()+(F.relu(.01-right_camera[:,2])/.01).square().mean()
            total = args.image_weight*image_loss+20*penetration+.3*contact+2*palm_contact+2*wrap+8*self_loss+30*mesh_collision+.3*prior+.01*reg+.01*anchor+20*depth
            return total,{'image':image_loss,'penetration':penetration,'contact':contact,'palm_contact':palm_contact,'wrap':wrap,'self_surrogate':self_loss,'mesh_collision':mesh_collision,'native_prior':prior}
        with torch.no_grad():
            initial_rv = tensor(Rotation.from_matrix(R0.cpu().numpy()).as_rotvec())
            _, _, initial_q, _ = forward(z0, initial_rv, t0)
            initial_metrics = {}
            for camera, samples, dist, mask in image_data:
                iu, io = project(initial_q, camera)
                im = render_mask(iu.cpu().numpy(), local_faces[~io.cpu().numpy()[local_faces].all(1)], mask.shape)
                inter = int((im.astype(bool)&mask.astype(bool)).sum())
                union = int((im.astype(bool)|mask.astype(bool)).sum())
                dd = torch.cdist(samples, iu).amin(1).cpu().numpy()
                initial_metrics[camera] = {'approx_visible_mask_iou':inter/max(union,1),'visible_annotation_coverage_vertex_distance_median_px':float(np.median(dd)),'p95_px':float(np.percentile(dd,95))}
        candidates=[]
        for run in range(args.starts):
            active_pairs = torch.empty((0,2),dtype=torch.long,device=device)
            coeffbase=tensor(initial_grip[f'{side}_hand_pca']) if initial_grip is not None else z0
            coeff=torch.nn.Parameter(coeffbase.clone()+tensor(rng.normal(0,.12,z0.shape))*(run>0))
            rotationbase=initial_grip[f'{side}_rotation_walker_from_wrist'] if initial_grip is not None else R0.cpu().numpy()
            rv0=Rotation.from_matrix(rotationbase).as_rotvec()
            rv=torch.nn.Parameter(tensor(rv0+rng.normal(0,args.orientation_noise,3)*(run>0)))
            translationbase=tensor(initial_grip[f'{side}_wrist_walker_m']) if initial_grip is not None else t0
            tr=torch.nn.Parameter(translationbase.clone()+tensor(rng.normal(0,.015,3))*(run>0))
            opt=torch.optim.Adam([{'params':[coeff,rv],'lr':.015},{'params':[tr],'lr':.0015}])
            trace=[]
            for step in range(args.steps):
                if step%25==0:
                    with torch.no_grad():
                        _,local_check,_,_=forward(coeff,rv,tr)
                        found=crossing_pairs(local_check.cpu().numpy(),local_faces)
                        active_pairs=torch.as_tensor(found,dtype=torch.long,device=device).reshape(-1,2)
                opt.zero_grad()
                loss,terms=objective(coeff,rv,tr)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite_loss')
                loss.backward()
                if step==0 and any(p.grad is None or not torch.isfinite(p.grad).all() for p in (coeff,rv,tr)):
                    raise ValueError('invalid_gradient')
                opt.step()
                if step in (args.steps//2,3*args.steps//4):
                    for group in opt.param_groups:group['lr']*=.4
                if step%100==0 or step==args.steps-1:
                    trace.append({'step':step,'total':float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()}})
            with torch.no_grad():
                final,terms=objective(coeff,rv,tr)
                pose,local,q,h=forward(coeff,rv,tr)
                qq=q.cpu().numpy()
                collisions=crossing_pairs(qq,local_faces)
                gap=capsule_gap(q,a,b).cpu().numpy()
                left_camera=q@Rc.T+tc
                right_camera=left_camera@R01.T+t01
                positive_depth=bool((left_camera[:,2]>0).all() and (right_camera[:,2]>0).all())
                good=len(collisions)==0 and float(gap.min())>=-.003 and positive_depth
                item={'start':run,'objective':float(final),'accepted_geometry':good,'positive_depth_both_cameras':positive_depth,'triangle_intersections':len(collisions),'intersection_pairs':collisions,'min_gap_mm':float(gap.min()*1000),'trace':trace,'pca':coeff.cpu().numpy().tolist(),'pose':pose.cpu().numpy().tolist(),'rotvec_walker':rv.cpu().numpy().tolist(),'wrist_walker_m':tr.cpu().numpy().tolist(),'terms':{k:float(v) for k,v in terms.items()}}
                candidates.append((item,qq,h.cpu().numpy(),local.cpu().numpy()))
                (args.output_dir/f'{side}_starts.json').write_text(json.dumps([c[0] for c in candidates],indent=2),encoding='utf-8')
            print(json.dumps({'hand':side,'start':run,'objective':item['objective'],'intersections':len(collisions),'min_gap_mm':item['min_gap_mm']}),flush=True)
        passing=[c for c in candidates if c[0]['accepted_geometry']]
        best=min(passing or candidates,key=lambda c:c[0]['objective'])
        item,qq,hh,ll=best
        metrics={}
        for camera,samples,dist,mask in image_data:
            with torch.no_grad():uv,occ=project(tensor(qq),camera)
            uv=uv.cpu().numpy()
            predicted=render_mask(uv,local_faces,mask.shape)
            # Remove triangles completely behind model handle (partial visibility remains approximate).
            visible_faces=local_faces[~occ.cpu().numpy()[local_faces].all(1)]
            visible=render_mask(uv,visible_faces,mask.shape)
            inter=int((visible.astype(bool)&mask.astype(bool)).sum())
            union=int((visible.astype(bool)|mask.astype(bool)).sum())
            coverage=np.linalg.norm(samples.cpu().numpy()[:,None]-uv[None],axis=-1).min(1)
            metrics[camera]={'approx_visible_mask_iou':inter/max(union,1),'visible_annotation_coverage_vertex_distance_median_px':float(np.median(coverage)),'p95_px':float(np.percentile(coverage,95))}
            image=cv2.imread(str(args.annotations/'images'/f'{camera}_pair_0000.png'))
            overlay=image.copy()
            overlay[visible>0]=(overlay[visible>0]*.55+np.array([70,210,250])*.45).astype(np.uint8)
            cv2.drawContours(overlay,cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(40,255,40),2)
            cv2.drawContours(overlay,cv2.findContours(visible,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,150,255),1)
            yy,xx=np.nonzero(mask|predicted)
            box=(max(0,int(xx.min())-35),max(0,int(yy.min())-35),min(1080,int(xx.max())+36),min(1920,int(yy.max())+36))
            x0,y0,x1,y1=box
            cv2.imwrite(str(args.output_dir/f'{side}_{camera}_overlay.png'),np.concatenate([image[y0:y1,x0:x1],overlay[y0:y1,x0:x1]],axis=1))
        # Explicit actual local candidate faces and local wrist frame, not a detached full-body mesh.
        export.update({f'{side}_vertices_walker_m':qq,f'{side}_joints21_walker_m':hh,f'{side}_vertices_local_m':ll,f'{side}_faces':local_faces,f'{side}_vertex_indices':idx,f'{side}_hand_pca':np.asarray(item['pca']),f'{side}_hand_pose':np.asarray(item['pose']),f'{side}_rotation_walker_from_wrist':Rotation.from_rotvec(item['rotvec_walker']).as_matrix(),f'{side}_wrist_walker_m':np.asarray(item['wrist_walker_m']),f'{side}_accepted_geometry':np.asarray(bool(passing)),f'{side}_native_observation_mask':weights.sum(1).cpu().numpy()>0})
        all_report['hands'][side]={'status':'geometry_gated_candidate' if passing else 'no_geometry_feasible_candidate','selected_start':item['start'],'initial_metrics':initial_metrics,'metrics':metrics,'selected':item,'all_starts':[c[0] for c in candidates]}
    np.savez_compressed(args.output_dir/'fixed_grip.npz',**export,betas=z['betas'],faces_source=np.asarray(asset['f']))
    (args.output_dir/'report.json').write_text(json.dumps(all_report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
