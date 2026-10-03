"""Frozen pure-function references from the pre-refactor engineering routes.

These are regression oracles, not production implementations. Source records:
G20260923_smpl_clean_full_sequence_v1/run_clean_full_sequence.py;
G20260924_smpl_vposer_shared_beta_v1/fit_vposer_shared_beta.py.
Kept here so a clean clone does not need the private research archive.
"""
from pathlib import Path
import json
import math
import cv2
import numpy as np
from pose_app.fisheye_camera import fisheye_project_numpy


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


def resolve_scene_frame_mask(st_accepted_full, wsl, n, tri_any):
    """Fail-closed scene-accepted mask for temporal triplets (pure numpy).

    Returns a boolean (n,) frame mask. Raises ValueError when the field is
    missing, has wrong shape, or holds abnormal values; never silently
    returns all-True, which would fabricate a temporal pass.
    """
    if st_accepted_full is None:
        raise ValueError("scene transforms carry no 'accepted' field: temporal mask unavailable")
    arr = np.asarray(st_accepted_full)
    try:
        win = arr[wsl]
    except Exception as exc:
        raise ValueError(f"scene 'accepted' window slice failed: {exc}")
    if win.ndim == 2:
        if win.shape[1] != 17:
            raise ValueError(f"scene 'accepted' joint dim {win.shape[1]} != 17")
        flat = np.asarray(win).reshape(-1)
        if flat.dtype != bool and not np.isin(flat, [0, 1]).all():
            raise ValueError("scene 'accepted' holds values outside {0,1}")
        mask = win.any(axis=-1)
    elif win.ndim == 1:
        mask = win
    else:
        raise ValueError(f"scene 'accepted' ndim {win.ndim} not in (1,2)")
    mask = np.asarray(mask).reshape(-1)
    if mask.shape[0] != n:
        raise ValueError(f"scene 'accepted' window frames {mask.shape[0]} != {n}")
    if mask.dtype != bool:
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("scene 'accepted' holds values outside {0,1}")
        mask = mask.astype(bool)
    if not np.isfinite(np.asarray(tri_any, dtype=float)).all():
        raise ValueError("triangulation frame mask is non-finite")
    return mask
