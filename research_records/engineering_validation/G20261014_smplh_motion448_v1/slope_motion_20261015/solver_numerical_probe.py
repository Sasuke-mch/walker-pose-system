"""Read-only real-asset derivative probes at saved event frames 37 and 59."""
from pathlib import Path
from dataclasses import replace
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
sys.path.insert(0, str(REPO / 'realtime_app'))
from pose_app.fisheye_camera import load_stereo_fisheye, fisheye_project_torch
from pose_app.smplh_motion_tracker import SMPLHObservationMap, SMPLHMotionConfig, MotionPose, retract
from pose_app.smplh_iterated_motion_tracker import SMPLHIteratedMotionTracker, IteratedMAPConfig


def main():
    run = OUT.parent / 'first_frame_sapiens_20261015/replay_selected_init/map_sigma075'
    p = json.loads((run / 'run_parameters.json').read_text())
    inp = p['inputs']
    model = SMPLHObservationMap(Path(p['effective_initial_fit']),
        *[Path(inp[k]) for k in ('smplh_model', 'regressor', 'vposer_dir', 'mano_left', 'mano_right')], device='cuda')
    cfg = SMPLHMotionConfig(**p['configuration'])
    cal = load_stereo_fisheye(inp['calibration_dir'])
    tr = SMPLHIteratedMotionTracker(model, cal, cfg)
    snap = dict(np.load(OUT / 'motion_chain_snapshots.npz'))
    report = {'source_run': str(run), 'purpose': 'Same fixed-prior MAP derivative and line-search numerical diagnosis; no tracking replay or production change',
        'production_dtype': 'float32 VPoser/SMPL-H, float64 projection and MAP algebra', 'events': []}
    preacts = []
    handles = [m.register_forward_pre_hook(lambda module, inputs: preacts.append(inputs[0].detach().cpu().numpy().copy()))
               for m in model.vposer.decoder_net if isinstance(m, torch.nn.LeakyReLU)]
    for frame in (37, 59):
        key = f'{frame}_'
        prior = MotionPose(snap[key+'prior_t'], snap[key+'prior_R'], snap[key+'prior_z'])
        P = snap[key+'P'][:38,:38]
        L = np.linalg.cholesky(P)
        selected = np.repeat(snap[key+'valid'].ravel(), 2)
        target = snap[key+'pixels'][...,:2].reshape(-1)[selected]
        postdelta = np.r_[snap[key+'post_t']-prior.translation_m,
            Rotation.from_matrix(prior.root_rotation.T@snap[key+'post_R']).as_rotvec(), snap[key+'post_z']-prior.latent]

        def numpy_eval(y, eps=None):
            d = L@y
            pose = retract(prior,d)
            proj = tr._pixels(model.evaluate(pose)['coco_m'])[0].reshape(-1)[selected]
            residual = (proj-target).reshape(-1,2)
            norms = np.linalg.norm(residual,axis=1)
            cost = float(np.sum(np.where(norms <= cfg.huber_px, .5*norms**2,
                cfg.huber_px*(norms-.5*cfg.huber_px)))/cfg.pixel_sigma_px**2+.5*y@y+.5*pose.latent@pose.latent/cfg.latent_prior_sigma**2)
            if eps is None:
                return cost
            offsets = np.r_[d[None]+np.eye(38)*eps, d[None]-np.eye(38)*eps]
            preacts.clear()
            data = model.evaluate_batch(prior.translation_m+offsets[:,:3],
                prior.root_rotation @ Rotation.from_rotvec(offsets[:,3:6]).as_matrix(), prior.latent+offsets[:,6:])
            activations = [x.copy() for x in preacts]
            J = ((tr._pixels(data['coco_m'])[:38]-tr._pixels(data['coco_m'])[38:])/(2*eps)).reshape(38,-1).T[selected]
            Jw = J@L
            weights = np.repeat(np.minimum(1.,cfg.huber_px/np.maximum(norms,1e-12)),2)/cfg.pixel_sigma_px**2
            Z=L[6:]
            g=Jw.T@(weights*residual.ravel())+y+Z.T@pose.latent/cfg.latent_prior_sigma**2
            H=(Jw.T*weights)@Jw+np.eye(38)+Z.T@Z/cfg.latent_prior_sigma**2
            flips = [int(np.count_nonzero((a[:38]>0)!=(a[38:]>0))) for a in activations]
            return cost,g,H,flips

        def autograd_eval(y,dtype=torch.float32):
            # Preserve the production FP32 decoder but differentiate its smooth
            # extension, then do calibrated projection/objective in FP64.
            yy = torch.tensor(y, dtype=torch.float64, device=model.device, requires_grad=True)
            LL = torch.tensor(L,dtype=torch.float64,device=model.device)
            dd = LL@yy
            v=dd[3:6]
            zero=v[0]*0
            skew=torch.stack((zero,-v[2],v[1],v[2],zero,-v[0],-v[1],v[0],zero)).reshape(3,3)
            roots=(torch.tensor(prior.root_rotation,dtype=torch.float64,device=model.device)@torch.matrix_exp(skew)).to(dtype)[None]
            z=(torch.tensor(prior.latent,dtype=torch.float64,device=model.device)+dd[6:]).to(dtype)[None]
            trans=(torch.tensor(prior.translation_m,dtype=torch.float64,device=model.device)+dd[:3]).to(dtype)[None]
            body=model.vposer.decode(z)['pose_body_matrot'].reshape(1,21,3,3).clone()
            body[:,model.fixed_body_indices]=model.fixed_body[list(model.fixed_body_indices)]
            result=model.model(betas=model.betas,transl=trans,global_orient=roots,body_pose=body,
                left_hand_pose=model.hand_rotations[0],right_hand_pose=model.hand_rotations[1])
            coco=torch.einsum('jv,bvc->bjc',model.regressor,result.vertices).double()[:,5:17]
            right=coco@torch.tensor(cal.R_cam0_to_cam1.T,dtype=torch.float64,device=model.device)+torch.tensor(cal.T_cam0_to_cam1_mm/1000,dtype=torch.float64,device=model.device)
            proj=torch.stack((fisheye_project_torch(coco,cal.K0,cal.D0),fisheye_project_torch(right,cal.K1,cal.D1)),dim=1).reshape(-1)[torch.tensor(selected,device=model.device)]
            residual=(proj-torch.tensor(target,dtype=torch.float64,device=model.device)).reshape(-1,2)
            norms=torch.linalg.norm(residual,dim=1)
            latent=torch.tensor(prior.latent,dtype=torch.float64,device=model.device)+dd[6:]
            cost=torch.where(norms<=cfg.huber_px,.5*norms**2,cfg.huber_px*(norms-.5*cfg.huber_px)).sum()/cfg.pixel_sigma_px**2+.5*yy@yy+.5*latent@latent/cfg.latent_prior_sigma**2
            grad=torch.autograd.grad(cost,yy)[0]
            return float(cost.detach()),grad.detach().cpu().numpy()

        tr80=SMPLHIteratedMotionTracker(model,cal,cfg,replace(IteratedMAPConfig(**p['iterated_map_configuration']),maximum_iterations=80))
        tr80.pose=prior.copy()
        tr80.covariance=snap[key+'P'].copy()
        tr80.last_accepted_pose=prior.copy()
        tr80._update(snap[key+'pixels'],snap[key+'valid'],np.full((2,12),'available',object))
        d80=np.r_[tr80.pose.translation_m-prior.translation_m,
            Rotation.from_matrix(prior.root_rotation.T@tr80.pose.root_rotation).as_rotvec(),tr80.pose.latent-prior.latent]
        event={'frame_index':frame,'points':[],'Pqq_eigenvalue_min_max':np.linalg.eigvalsh(P)[[0,-1]].tolist(),
               'same_frame_map80_optimizer':tr80.optimizer_diagnostic}
        for name,y in [('prior',np.zeros(38)),('saved_map15_posterior',np.linalg.solve(L,postdelta)),('repeated_map80_posterior',np.linalg.solve(L,d80))]:
            adcost,adg=autograd_eval(y)
            numcost=numpy_eval(y)
            row={'point':name,'numpy_cost':numcost,'autograd_cost':adcost,'objective_difference':adcost-numcost,
                'autograd_gradient_inf':float(np.max(abs(adg))),'finite_differences':[]}
            for eps in (1e-2,1e-3,1e-4,1e-5):
                cost,g,H,flips=numpy_eval(y,eps)
                direction=np.linalg.solve(H+1e-4*np.diag(np.diag(H)),-g)
                norm=float(np.linalg.norm(direction))
                unit=direction/norm
                directional=[]
                for h in (1e-3,1e-4,1e-5):
                    plus=numpy_eval(y+h*unit);minus=numpy_eval(y-h*unit)
                    directional.append({'h_in_whitened_y':h,'forward_slope':(plus-cost)/h,'backward_slope':(cost-minus)/h,'central_slope':(plus-minus)/(2*h)})
                trials=[]
                for power in (0,3,6,9,11,14):
                    alpha=.5**power
                    trial=numpy_eval(y+alpha*direction)
                    trials.append({'alpha':alpha,'cost_difference':trial-cost,'armijo_satisfied':bool(trial<=cost+1e-4*alpha*(g@direction))})
                row['finite_differences'].append({'eps_in_parameter_delta':eps,'gradient_inf':float(np.max(abs(g))),
                    'gradient_relative_L2_error_to_autograd':float(np.linalg.norm(g-adg)/np.linalg.norm(adg)),
                    'gradient_cosine_with_autograd':float(g@adg/(np.linalg.norm(g)*np.linalg.norm(adg))),
                    'leaky_relu_branch_flips_across_coordinate_pairs':flips,
                    'GN_direction_norm':norm,'FD_predicted_directional_slope':float(g@unit),'AD_directional_slope':float(adg@unit),
                    'directional_numerical_slopes':directional,'line_search_trials':trials})
            if name=='repeated_map80_posterior':
                _,g,H,_=numpy_eval(y,1e-3)
                direction=np.linalg.solve(H+1e-4*np.diag(np.diag(H)),-g)
                unit=direction/np.linalg.norm(direction)
                def set_dtype(dtype):
                    model.model.to(dtype=dtype)
                    model.vposer.to(dtype=dtype)
                    for attr in ('betas','fixed_body','regressor'):
                        setattr(model,attr,getattr(model,attr).to(dtype=dtype))
                    model.hand_rotations=[h.to(dtype=dtype) for h in model.hand_rotations]
                set_dtype(torch.float64)
                try:
                    c64,g64=autograd_eval(y,torch.float64)
                    directions={'FD_GN_eps1e-3':unit,'AD_steepest_descent':-g64/np.linalg.norm(g64)}
                    row['float64_model_probe']={'objective':c64,'objective_difference_vs_fp32':c64-numcost,
                        'gradient_inf':float(np.max(abs(g64))),'gradient_rel_L2_difference_vs_fp32_AD':float(np.linalg.norm(g64-adg)/np.linalg.norm(adg)),
                        'directions':[]}
                    for label,u in directions.items():
                        slopes=[]
                        for step in (1e-3,1e-4,1e-5,1e-6):
                            plus,_=autograd_eval(y+step*u,torch.float64)
                            minus,_=autograd_eval(y-step*u,torch.float64)
                            slopes.append({'h':step,'central_slope':(plus-minus)/(2*step),'forward_cost_change':plus-c64})
                        row['float64_model_probe']['directions'].append({'label':label,'predicted_slope':float(g64@u),'numerical':slopes})
                finally:
                    set_dtype(torch.float32)
            event['points'].append(row)
            print(frame,name,'costdiff',adcost-numcost,'ADginf',row['autograd_gradient_inf'],
                  'FD',[(d['eps_in_parameter_delta'],d['gradient_inf'],d['gradient_relative_L2_error_to_autograd'],d['leaky_relu_branch_flips_across_coordinate_pairs']) for d in row['finite_differences']],flush=True)
        report['events'].append(event)
        (OUT/'solver_numerical_probe.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for handle in handles:
        handle.remove()


def numerical_reference():
    """One preregistered exact-AD/double L-BFGS-B call per event, no replay."""
    from scipy.optimize import minimize
    import time

    report_path=OUT/'solver_numerical_probe.json'
    report=json.loads(report_path.read_text())
    reference=report['numerical_reference']
    assert reference['protocol']['maximum_iterations']==80
    assert reference['protocol']['maximum_line_search_steps']==20
    run=OUT.parent/'first_frame_sapiens_20261015/replay_selected_init/map_sigma075'
    p=json.loads((run/'run_parameters.json').read_text())
    inp=p['inputs']
    model=SMPLHObservationMap(Path(p['effective_initial_fit']),
        *[Path(inp[k]) for k in ('smplh_model','regressor','vposer_dir','mano_left','mano_right')],device='cuda')
    model.model.double()
    model.vposer.double()
    for attr in ('betas','fixed_body','regressor'):
        setattr(model,attr,getattr(model,attr).double())
    model.hand_rotations=[h.double() for h in model.hand_rotations]
    cfg=SMPLHMotionConfig(**p['configuration'])
    cal=load_stereo_fisheye(inp['calibration_dir'])
    snap=dict(np.load(OUT/'motion_chain_snapshots.npz'))
    controls=json.loads((OUT/'leave_pair_fixed_prior_diagnostic.json').read_text())['frames']
    scene_path=OUT.parent/'first_frame_sapiens_20261015/visualization/scene/scene_transforms.npz'
    with np.load(scene_path) as scene_file:
        scene={k:scene_file[k].copy() for k in ('rotation_ground_from_left','translation_ground_from_left_mm')}
    reference['ground_transform_source']=str(scene_path)
    reference['status']='running'
    report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

    def tensor(v):
        return torch.tensor(v,dtype=torch.float64,device=model.device)

    for frame in (37,59):
        # A completed or started optimizer call is never silently repeated.
        if any(e['frame_index']==frame for e in reference['events']):
            raise RuntimeError(f'Numerical reference for frame{frame} already declared; refusing duplicate call')
        key=f'{frame}_'
        P=snap[key+'P'][:38,:38]
        L=np.linalg.cholesky(P)
        S=np.sqrt(np.diag(P))
        precision=tensor(np.linalg.inv(P))
        prior_t=tensor(snap[key+'prior_t'])
        prior_R=tensor(snap[key+'prior_R'])
        prior_z=tensor(snap[key+'prior_z'])
        St=tensor(S)
        valid=snap[key+'valid']
        selected=torch.tensor(np.repeat(valid.ravel(),2),device=model.device)
        target=tensor(snap[key+'pixels'][...,:2])
        RR=tensor(cal.R_cam0_to_cam1.T)
        tt=tensor(cal.T_cam0_to_cam1_mm/1000)
        Rg=tensor(scene['rotation_ground_from_left'][frame])
        tg=tensor(scene['translation_ground_from_left_mm'][frame]/1000)
        cache={}
        calls=0

        def evaluate(x):
            nonlocal calls
            if cache and np.array_equal(x,cache['x']):
                return cache['cost'],cache['gradient']
            calls+=1
            xx=tensor(x).requires_grad_()
            delta=St*xx
            v=delta[3:6]
            zero=v[0]*0
            skew=torch.stack((zero,-v[2],v[1],v[2],zero,-v[0],-v[1],v[0],zero)).reshape(3,3)
            root=prior_R@torch.matrix_exp(skew)
            latent=prior_z+delta[6:]
            body=model.vposer.decode(latent[None])['pose_body_matrot'].reshape(1,21,3,3).clone()
            body[:,model.fixed_body_indices]=model.fixed_body[list(model.fixed_body_indices)]
            result=model.model(betas=model.betas,transl=(prior_t+delta[:3])[None],global_orient=root[None],body_pose=body,
                left_hand_pose=model.hand_rotations[0],right_hand_pose=model.hand_rotations[1])
            coco=torch.einsum('jv,bvc->bjc',model.regressor,result.vertices)[0]
            points=coco[5:17]
            right=points@RR+tt
            projected=torch.stack((fisheye_project_torch(points,cal.K0,cal.D0),fisheye_project_torch(right,cal.K1,cal.D1)))
            residual=(projected.reshape(-1)[selected]-target.reshape(-1)[selected]).reshape(-1,2)
            norm=torch.linalg.norm(residual,dim=1)
            pixel=torch.where(norm<=cfg.huber_px,.5*norm**2,cfg.huber_px*(norm-.5*cfg.huber_px)).sum()/cfg.pixel_sigma_px**2
            temporal=.5*delta@precision@delta
            vposer=.5*latent@latent/cfg.latent_prior_sigma**2
            cost=pixel+temporal+vposer
            gradient=torch.autograd.grad(cost,xx)[0]
            if not torch.isfinite(cost) or not torch.isfinite(gradient).all():
                raise FloatingPointError('nonfinite exact-AD objective or gradient')
            err=torch.linalg.norm(projected-target,dim=-1).detach().cpu().numpy()
            ankle=((coco[15:17]@Rg.T+tg)[:,2]*1000).detach().cpu().numpy()
            grad=gradient.detach().cpu().numpy()
            delta_np=delta.detach().cpu().numpy()
            cache.clear()
            cache.update(x=np.array(x,copy=True),cost=float(cost.detach()),gradient=grad,
                description={'objective':float(cost.detach()),'pixel_cost':float(pixel.detach()),'temporal_cost':float(temporal.detach()),
                    'whole_vposer_cost':float(vposer.detach()),'scaled_gradient_inf_in_original_white_y':float(np.max(abs(L.T@(grad/S)))),
                    'scaled_gradient_inf_in_LBFGSB_x':float(np.max(abs(grad))),
                    'all24_error_median_p95_px':[float(np.median(err)),float(np.percentile(err,95))],
                    'all24_error_px':err.tolist(),'ankle_height_mm':ankle.tolist(),'mean_ankle_height_mm':float(np.mean(ankle)),
                    'delta_in_original_prior_tangent':delta_np.tolist(),'latent_max_abs':float(latent.detach().abs().max()),
                    'translation_m':(prior_t+delta[:3]).detach().cpu().numpy().tolist(),
                    'root_rotation':root.detach().cpu().numpy().tolist(),'latent':latent.detach().cpu().numpy().tolist()})
            return cache['cost'],grad

        event={'frame_index':frame,'status':'optimizer_call_started','original_valid_mask':valid.tolist(),'history':[]}
        reference['events'].append(event)
        report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        evaluate(np.zeros(38))
        event['before']=dict(cache['description'])
        def callback(x):
            evaluate(x)
            row=cache['description']
            event['history'].append({'iteration':len(event['history'])+1,'objective':row['objective'],
                'scaled_gradient_inf_in_original_white_y':row['scaled_gradient_inf_in_original_white_y'],
                'mean_ankle_height_mm':row['mean_ankle_height_mm']})
        bounds=[(None,None)]*6+[((-cfg.max_abs_latent-float(prior_z[k]))/S[k+6],
            (cfg.max_abs_latent-float(prior_z[k]))/S[k+6]) for k in range(32)]
        started=time.perf_counter()
        try:
            solution=minimize(evaluate,np.zeros(38),method='L-BFGS-B',jac=True,bounds=bounds,callback=callback,
                options={'maxiter':80,'maxls':20,'maxfun':2000,'ftol':1e-12,'gtol':1e-5})
            evaluate(solution.x)
            event.update(status='completed',optimizer_success=bool(solution.success),optimizer_status=int(solution.status),
                optimizer_message=str(solution.message),iterations=int(solution.nit),function_evaluations=int(solution.nfev),
                actual_forward_backward_calls=calls,seconds=time.perf_counter()-started,after=dict(cache['description']))
            event['mean_ankle_height_update_mm']=event['after']['mean_ankle_height_mm']-event['before']['mean_ankle_height_mm']
            control=next(e for e in controls if e['frame_index']==frame)
            event['controls']={}
            for policy in ('all','all_budget80'):
                c=control['policies'][policy]
                a=c['after']
                event['controls'][policy]={'objective':a['original_objective'],'gradient_inf':c['optimizer']['scaled_gradient_inf'],
                    'all24_error_median_p95_px':a['all24_error_median_p95_px'],'ankle_height_mm':a['ankle_height_mm'],
                    'mean_ankle_height_mm':float(np.mean(a['ankle_height_mm'])),
                    'mean_ankle_height_update_mm':float(np.mean(a['ankle_height_mm'])-event['before']['mean_ankle_height_mm']),
                    'termination':c['optimizer']['termination'],'optimizer_success':c['optimizer']['optimizer_success']}
        except Exception as exc:
            event.update(status='failed_preserved',exception_type=type(exc).__name__,exception=str(exc),seconds=time.perf_counter()-started,
                actual_forward_backward_calls=calls,last_evaluated=dict(cache.get('description',{})))
        report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print('numerical_reference',frame,event['status'],'objective',event.get('after',{}).get('objective'),
            'mean_ankle_update_mm',event.get('mean_ankle_height_update_mm'),'iterations',event.get('iterations'),
            'termination',event.get('optimizer_message'),flush=True)
    reference['status']='completed_one_call_per_event'
    report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    if '--numerical-reference-only' in sys.argv:
        numerical_reference()
    else:
        main()
