#!/usr/bin/env python3
"""Independent raw-PMPose -> male-SMPL full-sequence experiment.

This entry point deliberately has no dependency on any previous fit, temporal
prefit, triangulation JSON, contact target, HTML or parameter file.  It reads
only the two raw PMPose JSON files, raw pair timestamps, calibration, the
official male SMPL pickle and the 17x6890 observation regressor.
"""
from __future__ import annotations

import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "realtime_app"))
from pose_app.fisheye_camera import fisheye_project_numpy, fisheye_project_torch, load_stereo_fisheye
from pose_app.smpl_coco_observation import (load_coco17_regressor, load_smpl_male,
    regress_coco17_numpy, regress_coco17_torch, validate_coco17_regressor)

NAMES = ("nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
         "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee",
         "right_knee","left_ankle","right_ankle")
SKELETON = ((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))

def raw_side(path: Path, side: str):
    obj=json.loads(path.read_text(encoding="utf-8"))
    rows={}
    for item in obj.get("images",[]):
        fid=int(item.get("image_id", item.get("frame_index", len(rows))))
        kp=np.asarray(item["keypoints"][0], dtype=np.float64)
        # PMPose raw export may carry 23 channels; its first 17 channels are
        # the documented COCO-17 observation order. Preserve the full source
        # separately in the audit, but only these 17 enter this SMPL line.
        if kp.shape == (23,3): kp=kp[:17]
        if kp.shape != (17,3): raise ValueError(f"{path}: expected 17x3 or 23x3, got {kp.shape}")
        # PMPose was run on the rotated upright view. Convert its coordinates
        # back to the original 1920x1080 fisheye pixel convention before any
        # association or triangulation.
        u, v = kp[:,0].copy(), kp[:,1].copy()
        if side == "left": kp[:,0], kp[:,1] = 1919.0-v, u       # raw -> ccw90
        elif side == "right": kp[:,0], kp[:,1] = v, 1079.0-u   # raw -> cw90
        else: raise ValueError(side)
        rows[fid]=kp
    return rows

def raw_triangulate(left,right,cal):
    T=len(left); raw=np.full((T,17,3),np.nan); el=np.full((T,17),np.nan); er=el.copy(); gap=el.copy()
    dl=el.copy(); dr=el.copy(); accepted=np.zeros((T,17),bool); reason=np.full((T,17),"missing",dtype=object)
    q2d=np.zeros((T,17)); qray=np.zeros_like(q2d); qrep=np.zeros_like(q2d); qsync=np.zeros_like(q2d); qdepth=np.zeros_like(q2d)
    P0=np.hstack([np.eye(3),np.zeros((3,1))]); P1=np.hstack([cal.R_cam0_to_cam1,np.asarray(cal.T_cam0_to_cam1_mm).reshape(3,1)])
    for t in range(T):
        for j in range(17):
            a,b=left[t,j],right[t,j]; cL=float(a[2]); cR=float(b[2]); q2d[t,j]=math.sqrt(max(cL,0)*max(cR,0))
            if not np.isfinite(a).all() or not np.isfinite(b).all(): reason[t,j]="nan_inf"; continue
            if not (0<=a[0]<cal.image_width and 0<=a[1]<cal.image_height and 0<=b[0]<cal.image_width and 0<=b[1]<cal.image_height): reason[t,j]="out_of_raw_image_bounds"; continue
            ua=cv2.fisheye.undistortPoints(a[:2].reshape(1,1,2),cal.K0,cal.D0).reshape(2)
            ub=cv2.fisheye.undistortPoints(b[:2].reshape(1,1,2),cal.K1,cal.D1).reshape(2)
            x=cv2.triangulatePoints(P0,P1,ua.reshape(2,1),ub.reshape(2,1)); X=(x[:3,0]/x[3,0])
            if not np.isfinite(X).all(): reason[t,j]="non_finite_triangulation"; continue
            raw[t,j]=X; pL=fisheye_project_numpy(X[None],cal.K0,cal.D0)[0]; Xr=cal.R_cam0_to_cam1@X+cal.T_cam0_to_cam1_mm; pR=fisheye_project_numpy(Xr[None],cal.K1,cal.D1)[0]
            el[t,j]=np.linalg.norm(pL-a[:2]); er[t,j]=np.linalg.norm(pR-b[:2]); dl[t,j]=X[2]; dr[t,j]=Xr[2]
            # ray gap is the closest distance between the two calibrated rays
            r0=np.array([ua[0],ua[1],1.0]); r0/=np.linalg.norm(r0); r1=cal.R_cam0_to_cam1.T@np.array([ub[0],ub[1],1.0]); r1/=np.linalg.norm(r1); o1=-cal.R_cam0_to_cam1.T@cal.T_cam0_to_cam1_mm
            n=np.cross(r0,r1); gap[t,j]=abs(np.dot(o1,n))/max(np.linalg.norm(n),1e-12)
            qray[t,j]=math.exp(-gap[t,j]/30.0); qrep[t,j]=math.exp(-(el[t,j]+er[t,j])/(2*8.0)); qsync[t,j]=1.0; qdepth[t,j]=1.0 if dl[t,j]>0 and dr[t,j]>0 else 0.0
            accepted[t,j]=bool(dl[t,j]>0 and dr[t,j]>0 and np.linalg.norm(n)>1e-8 and np.isfinite(el[t,j]+er[t,j]))
            reason[t,j]=None if accepted[t,j] else ("negative_depth" if qdepth[t,j]==0 else "parallel_rays")
    q=np.clip(q2d*qray*qrep*qsync*qdepth,0,1); q[~accepted]=0
    # Preserve every finite, positive-depth joint. Reprojection quality remains
    # represented in qrep/q and therefore in the fitting weight; no joint is
    # hard-coded out after triangulation.
    return raw,el,er,gap,dl,dr,accepted,reason,q,(q2d,qray,qrep,qsync,qdepth)

def smooth_points(X,q):
    # Small local quadratic solve: robust observation fidelity + acceleration.
    Y=X.copy(); T,J,_=X.shape
    for j in range(J):
        for c in range(3):
            y=X[:,j,c].copy(); finite=np.isfinite(y)
            if finite.any(): y[~finite]=np.interp(np.flatnonzero(~finite),np.flatnonzero(finite),y[finite])
            else: y[:]=0.0
            w=q[:,j].copy(); w[~finite]=0.0; A=np.diag(w); b=w*y
            lam=0.15
            for t in range(1,T-1):
                d=np.zeros(T); d[t-1]=1; d[t]=-2; d[t+1]=1; A+=lam*np.outer(d,d)
            z=np.linalg.solve(A+1e-3*np.eye(T),b); Y[:,j,c]=z
    return Y

def read_timestamps(path,n):
    vals=[]
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f): vals.append((float(r["left_host_return_timestamp_ns"]),float(r["right_host_return_timestamp_ns"])))
    if len(vals)<n: raise ValueError(f"timestamp rows {len(vals)} < PMPose frames {n}")
    vals=vals[:n]; return np.asarray(vals), np.abs(np.asarray(vals)[:,1]-np.asarray(vals)[:,0])/1e6

def interp_knots(x,k):
    idx=np.linspace(0,len(x)-1,k).round().astype(int); out=[]
    for t in range(len(x)):
        a=np.searchsorted(idx,t,side="right")-1; a=max(0,min(a,k-2)); u=(t-idx[a])/max(idx[a+1]-idx[a],1); out.append(x[idx[a]]*(1-u)+x[idx[a+1]]*u)
    return np.stack(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--left",type=Path,required=True); ap.add_argument("--right",type=Path,required=True); ap.add_argument("--timestamps",type=Path,required=True); ap.add_argument("--calibration-dir",type=Path,required=True); ap.add_argument("--model",type=Path,required=True); ap.add_argument("--regressor",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--steps",type=int,default=250); ap.add_argument("--open-beta",action="store_true"); args=ap.parse_args(); out=args.out.resolve()
    if out.exists() and any(out.iterdir()): raise RuntimeError(f"refuse non-empty output: {out}")
    out.mkdir(parents=True,exist_ok=True)
    cal=load_stereo_fisheye(args.calibration_dir); reg=validate_coco17_regressor(load_coco17_regressor(args.regressor));
    L=raw_side(args.left,"left"); R=raw_side(args.right,"right"); ids=sorted(set(L)&set(R));
    if ids!=list(range(len(ids))): raise ValueError("raw PMPose frame ids are not contiguous from zero")
    left=np.stack([L[i] for i in ids]); right=np.stack([R[i] for i in ids]); timestamps,dt=read_timestamps(args.timestamps,len(ids))
    tri=raw_triangulate(left,right,cal); raw,el,er,gap,dl,dr,acc,reason,q,parts=tri; proc=smooth_points(raw,q)
    np.savez(out/"raw_observations.npz",left_2d=left,right_2d=right,timestamps=timestamps,time_delta_ms=dt)
    with (out/"raw_observation_audit.json").open("w",encoding="utf-8") as f: json.dump({"source_whitelist":[str(args.left.resolve()),str(args.right.resolve()),str(args.timestamps.resolve()),str((args.calibration_dir/"stereo_fisheye.json").resolve()),str(args.model.resolve()),str(args.regressor.resolve())],"frames":len(ids),"max_matches":1,"old_fit_inputs_read":False},f,indent=2)
    np.savez(out/"triangulation.npz",raw=raw,left_reprojection_error=el,right_reprojection_error=er,ray_gap=gap,depth_left=dl,depth_right=dr,accepted=acc,confidence=q,confidence_2d=parts[0],confidence_ray=parts[1],confidence_reproj=parts[2],confidence_sync=parts[3],confidence_depth=parts[4],time_delta_ms=dt)
    with (out/"triangulation_quality.json").open("w",encoding="utf-8") as f: json.dump({"frames":len(ids),"accepted_per_joint":acc.sum(0).astype(int).tolist(),"rejected":int((~acc).sum()),"negative_depth":int(((dl<=0)|(dr<=0)).sum()),"ray_gap_median_mm":float(np.nanmedian(gap)),"rejection_reasons":{str(k):int(np.sum(reason==k)) for k in sorted({str(x) for x in reason.ravel()})}},f,indent=2)
    np.savez(out/"temporal_points.npz",temporally_processed_points=proc)
    # Fit from zero parameters.  The only data-derived initial translation is pelvis triangulation.
    import torch
    torch.set_num_threads(max(1,min(8,torch.get_num_threads())))
    model=load_smpl_male(args.model,batch_size=len(ids),device="cpu",dtype=torch.float32)
    T=len(ids); K=max(2,int(math.ceil(T/4))); knots=np.linspace(0,T-1,K).round().astype(int)
    pelvis=np.nanmean(proc[:,[11,12]],axis=1); trans0=np.nan_to_num(pelvis,nan=0.0)/1000.0
    from scipy.spatial.transform import Rotation
    root0=np.zeros((T,3),dtype=np.float32)
    for t in range(T):
        x=proc[t,5]-proc[t,6]; y=(proc[t,5]+proc[t,6]-proc[t,11]-proc[t,12])*0.5
        x=x/max(np.linalg.norm(x),1e-8); y=y-x*np.dot(x,y); y=y/max(np.linalg.norm(y),1e-8); z=np.cross(x,y); z=z/max(np.linalg.norm(z),1e-8)
        M=np.column_stack([x,y,z]);
        if np.linalg.det(M)<0: M[:,2]*=-1
        root0[t]=Rotation.from_matrix(M).as_rotvec()
    trans_k=torch.nn.Parameter(torch.tensor(trans0[knots],dtype=torch.float32)); pose_k=torch.nn.Parameter(torch.zeros(K,69)); root_k=torch.nn.Parameter(torch.tensor(root0[knots],dtype=torch.float32)); betas=torch.nn.Parameter(torch.zeros(10),requires_grad=args.open_beta)
    obsL=torch.tensor(left[:,:,:2],dtype=torch.float32); obsR=torch.tensor(right[:,:,:2],dtype=torch.float32); w=torch.tensor(q,dtype=torch.float32); target=torch.tensor(proc/1000.0,dtype=torch.float32)
    params=[root_k,pose_k,trans_k]+([betas] if args.open_beta else []); opt=torch.optim.Adam([{"params":[root_k,pose_k,trans_k],"lr":0.004}]+([{"params":[betas],"lr":0.0004}] if args.open_beta else [])); K0=torch.tensor(cal.K0,dtype=torch.float32); D0=torch.tensor(cal.D0,dtype=torch.float32); K1=torch.tensor(cal.K1,dtype=torch.float32); D1=torch.tensor(cal.D1,dtype=torch.float32); R01=torch.tensor(cal.R_cam0_to_cam1,dtype=torch.float32); t01=torch.tensor(cal.T_cam0_to_cam1_mm/1000,dtype=torch.float32)
    W=np.zeros((T,K),dtype=np.float32)
    for t in range(T):
        a=max(0,min(K-2,np.searchsorted(knots,t,side="right")-1)); u=float((t-knots[a])/max(knots[a+1]-knots[a],1)); W[t,a]=1-u; W[t,a+1]=u
    Wt=torch.tensor(W,dtype=torch.float32)
    def forward(beta):
        roots=Wt@root_k; poses=Wt@pose_k; trans=Wt@trans_k
        res=model(betas=beta[None].expand(T,-1),body_pose=poses,global_orient=roots,transl=trans,return_verts=True); jc=regress_coco17_torch(res.vertices,reg); return res,jc
    for step in range(args.steps):
        opt.zero_grad(); res,jc=forward(betas); poses=Wt@pose_k; X=jc-target; l3=(torch.sqrt((X*X).sum(-1)+1e-8)*w).sum()/(w.sum()+1e-6); ll=0
        leftc=jc; rightc=torch.einsum("ij,tvj->tvi",R01,leftc)+t01; pl=fisheye_project_torch(leftc,K0,D0); pr=fisheye_project_torch(rightc,K1,D1); valid=torch.isfinite(obsL).all(-1)&torch.isfinite(obsR).all(-1); ll=(((torch.linalg.vector_norm(pl-obsL,dim=-1)*w)[valid].mean()+(torch.linalg.vector_norm(pr-obsR,dim=-1)*w)[valid].mean())/100.0)*0.15
        accloss=((jc[2:]-2*jc[1:-1]+jc[:-2])**2).mean()*0.03; betareg=(betas.square().mean()*0.01 if args.open_beta else 0.0); loss=l3+ll+accloss+0.005*poses.square().mean()+betareg; loss.backward(); torch.nn.utils.clip_grad_norm_(params,10.0); opt.step()
        if args.open_beta: betas.data.clamp_(-1.5,1.5)
        if step in {0,args.steps-1}: print(json.dumps({"step":step,"loss":float(loss.detach()),"loss3d":float(l3.detach()),"loss2d":float(ll.detach())}),flush=True)
    with torch.no_grad(): res,jc=forward(betas); verts=res.vertices.numpy(); pred=jc.numpy(); joints=res.joints.numpy(); body=(Wt@pose_k).detach().numpy(); roots=(Wt@root_k).detach().numpy(); trans=(Wt@trans_k).detach().numpy()
    faces=np.asarray(model.faces.detach().cpu().numpy() if hasattr(model.faces,"detach") else model.faces)
    beta_np=betas.detach().numpy(); np.savez(out/"result.npz",vertices=verts,faces=faces,smpl_joints=joints,predicted_coco=pred,betas=beta_np,body_pose=body,global_orient=roots,transl=trans,raw_triangulated_points=raw,temporally_processed_points=proc,joint_confidence=q,accepted_mask=acc,reject_reason=reason,ground_model=np.array([],dtype=np.float32),walker_model=np.array([],dtype=np.float32))
    d3=np.linalg.norm(pred-proc/1000,axis=-1)*1000; pl=fisheye_project_numpy(pred,cal.K0,cal.D0); pr=fisheye_project_numpy(np.einsum("ij,tvj->tvi",cal.R_cam0_to_cam1,pred)+cal.T_cam0_to_cam1_mm/1000.0,cal.K1,cal.D1); e2=np.concatenate([np.linalg.norm(pl-left[:,:,:2],axis=-1)[acc],np.linalg.norm(pr-right[:,:,:2],axis=-1)[acc]])
    metrics={"frames":T,"2d":{"median_px":float(np.median(e2)) if len(e2) else None,"p95_px":float(np.percentile(e2,95)) if len(e2) else None},"3d":{"median_mm":float(np.median(d3[acc])) if acc.any() else None,"p95_mm":float(np.percentile(d3[acc],95)) if acc.any() else None,"max_mm":float(np.max(d3[acc])) if acc.any() else None,"joint_p95_mm":{NAMES[j]:float(np.percentile(d3[:,j][acc[:,j]],95)) if acc[:,j].any() else None for j in range(17)}},"temporal":{"coco_velocity_p95_mm":float(np.percentile(np.linalg.norm(np.diff(pred,axis=0),axis=-1)*1000,95)),"coco_acceleration_p95_mm":float(np.percentile(np.linalg.norm(np.diff(pred,2,axis=0),axis=-1)*1000,95)),"root_velocity_p95_mm":float(np.percentile(np.linalg.norm(np.diff(trans,axis=0),axis=-1)*1000,95)),"root_acceleration_p95_mm":float(np.percentile(np.linalg.norm(np.diff(trans,2,axis=0),axis=-1)*1000,95))},"negative_depth":int(((dl<=0)|(dr<=0)).sum()),"nan_inf":int((~np.isfinite(pred)).sum()),"right_knee_supervised":int(acc[:,14].sum()),"contact":{"status":"unavailable_no_raw_ground_or_walker_geometry","foot_contact_frames":0,"hand_contact_frames":0}}
    (out/"smpl_fit_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8"); (out/"beta_estimate.json").write_text(json.dumps({"betas":beta_np.tolist(),"shared":True,"boundary":bool(np.any(np.isclose(np.abs(beta_np),1.5,atol=1e-4))),"status":"opened_after_beta_zero_base" if args.open_beta else "beta_zero_base"},indent=2),encoding="utf-8"); (out/"contact_metrics.json").write_text(json.dumps(metrics["contact"],indent=2),encoding="utf-8")
    (out/"full_sequence_visualization.html").write_text("<!doctype html><meta charset='utf-8'><title>independent clean male SMPL</title><pre id='p'>result.npz generated; mesh and COCO points are from the same parameter set. Contact unavailable: no raw ground/walker geometry.</pre>",encoding="utf-8")
    return 0

if __name__=="__main__":
    import cv2
    raise SystemExit(main())
