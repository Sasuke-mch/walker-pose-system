#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
APP_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(APP_ROOT))
from tools.diagnose_smplx_loss_gradients import parser
from tools.benchmark_smplx_r3_integration import FitEnv,FitRouteConfig,DEV_WINDOW
from tools.benchmark_smplx_structured_contact import expand_knots
from tools.fit_smplx_clean_male_coarse_to_fine import balanced_huber_2d
from pose_app.fisheye_camera import fisheye_project_numpy

def main():
 p=parser();p.add_argument('--temporal-dir',type=Path,required=True);a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);env=FitEnv(a,*DEV_WINDOW,device=a.device);t=env.torch;s=np.load(a.temporal_dir/'knot_parameters.npz');
 rootk=t.tensor(s['root'],dtype=t.float32,device=env.device);trk=t.tensor(s['translation'],dtype=t.float32,device=env.device);latk=t.tensor(s['latent'],dtype=t.float32,device=env.device);root=expand_knots(t,rootk,env.window_size,'linear');tr=expand_knots(t,trk,env.window_size,'linear');lat=expand_knots(t,latk,env.window_size,'linear');beta=t.nn.Parameter(t.zeros(4,dtype=t.float32,device=env.device))
 def bf():return t.cat([beta,t.zeros(6,device=env.device)])
 def metric():
  with t.no_grad():_,p,_,_=env.forward(root,tr,lat,betas=bf())
  p=p.detach().cpu().numpy();
  if p.ndim==4:p=p[np.arange(env.window_size),np.arange(env.window_size)]
  per={j:[] for j in range(5,17)};allv=[]
  for side in ('left','right'):
   vals=[];K=env.cal.K0 if side=='left' else env.cal.K1;D=env.cal.D0 if side=='left' else env.cal.D1
   for i in range(env.window_size):
    left=(env.per_frame_rotation[i].T@(p[i]*1000-env.per_frame_translation[i]).T).T;cam=left if side=='left' else (env.cal.R_cam0_to_cam1@left.T+env.cal.T_cam0_to_cam1_mm[:,None]).T;pr=fisheye_project_numpy(cam[5:17],K,D);tar=env.obs2d[f'kp_{side}_px'][i];valid=env.obs2d[f'accepted_{side}'][i];e=np.linalg.norm(pr[valid]-tar[valid],axis=1);vals.extend(e.tolist());allv.extend(e.tolist());
    if side=='left':lp=float(np.percentile(vals,95))
    else:rp=float(np.percentile(vals,95))
   for j in range(12): per[j+5].extend([])
  return {'left_p95':lp,'right_p95':rp,'loss':float(balanced_huber_2d(env,root,tr,lat,20.,bf()).detach().cpu())}
 initial=metric();opt=t.optim.Adam([beta],lr=.01)
 for _ in range(300):
  opt.zero_grad();loss=balanced_huber_2d(env,root.detach(),tr.detach(),lat.detach(),20.,bf())+.2*t.mean(beta.square());loss.backward();opt.step();beta.data.clamp_(-1.5,1.5)
 final=metric();payload={'schema_version':'smplx_male_beta_after_temporal_v1','status':'evaluated_no_contact','temporal_source':str(a.temporal_dir),'initial':initial,'final':final,'betas':beta.detach().cpu().numpy().tolist(),'interpretation':'beta is evaluated with temporal knots fixed; no contact fitting and no anatomical-truth claim'}
 (out/'metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return 0
if __name__=='__main__':raise SystemExit(main())
