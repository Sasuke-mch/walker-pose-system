#!/usr/bin/env python3
"""A0 current v3 initialization vs A1 direct triangulation initialization."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
APP_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP_ROOT))
from tools.diagnose_smplx_loss_gradients import parser
from tools.benchmark_smplx_r3_integration import FitEnv,FitRouteConfig,DEV_WINDOW,build_per_frame_rows
from tools.benchmark_smplx_structured_contact import knot_indices,expand_knots,enforce_coco_pelvis_delta
from tools.fit_smplx_clean_male_coarse_to_fine import balanced_huber_2d

BETA=np.array([.39927265,-.07243486,.68638891,1.28997993],dtype=np.float32)
def run(env, base, label):
 t=env.torch; ids=knot_indices(env.window_size,5); params=[t.nn.Parameter(x.detach()[ids].clone()) for x in base]; opt=t.optim.Adam(params,lr=.005); beta=t.as_tensor(np.r_[BETA,np.zeros(6,dtype=np.float32)],device=env.device); cfg=FitRouteConfig(label,'prefit',False,None,temporal_mode='factorized',root_temporal_weight=220.,local_temporal_weight=220.,temporal_point_space='coco',foot_contact_weight=0.,hand_contact_weight=0.); snaps=[]
 for step in range(401):
  full=tuple(expand_knots(t,x,env.window_size,'linear') for x in params);full=(full[0],enforce_coco_pelvis_delta(env,*full),full[2])
  if step in (0,100,200,400):
   env.betas_tensor=beta[None,:].expand(env.window_size,-1);m=env.snapshot(*full,env.masks_with_holdout(set()),cfg,set());m['iteration']=step;snaps.append(m)
  if step==400:break
  opt.zero_grad();loss=balanced_huber_2d(env,*full,delta_px=100. if step<100 else 20.,betas=beta);_,coco,_,_=env.forward(*full,betas=beta);pel=.5*(coco[:,11]+coco[:,12]);rel=coco[:,5:17]-pel[:,None];loss=loss+220.*t.mean((pel[2:]-2*pel[1:-1]+pel[:-2]).square())+220.*t.mean((rel[2:]-2*rel[1:-1]+rel[:-2]).square())+.02*t.mean(full[2].square());loss.backward();opt.step()
 env.betas_tensor=beta[None,:].expand(env.window_size,-1);m=env.snapshot(*full,env.masks_with_holdout(set()),cfg,set());rows=build_per_frame_rows(env,cfg,*full,set(),label,0,False);pts=np.asarray([r['smplx_coco17_joints_ground_mm'] for r in rows]);p95=np.percentile(np.linalg.norm(np.diff(pts,n=2,axis=0),axis=2),95,axis=0);m['per_joint_accel_p95_mm_per_frame2']=p95.tolist();return {'label':label,'initial':snaps[0],'final':m,'per_joint_accel_p95_mm_per_frame2':p95.tolist()}
def main():
 p=parser();p.add_argument('--v3-parameters',type=Path,required=True);a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);env=FitEnv(a,*DEV_WINDOW,device=a.device);t=env.torch;s=np.load(a.v3_parameters);v3=tuple(t.tensor(s[k],dtype=t.float32,device=env.device) for k in ('root_orient','translation','vposer_latent'));direct=env.make_params(set());res={'schema_version':'smplx_direct_triangulation_init_comparison_v1','design':'same 5-frame temporal fit, same fixed beta, only initialization source changes','A0_v3_initialization':run(env,v3,'A0-v3-init'),'A1_direct_triangulation_initialization':run(env,direct,'A1-direct-triangulation-init')};(out/'metrics.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return 0
if __name__=='__main__':raise SystemExit(main())
