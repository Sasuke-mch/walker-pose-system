import argparse, csv, json, math
from pathlib import Path
import cv2
import numpy as np


def make_board(square_mm, marker_mm):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    b = cv2.aruco.CharucoBoard((8, 6), square_mm, marker_mm, d)
    return b, cv2.aruco.CharucoDetector(b)


def detect(path, board, detector):
    img = cv2.imread(str(path))
    if img is None:
        return None
    cc, ci, _, _ = detector.detectBoard(img)
    if cc is None or ci is None or len(ci) < 6:
        return None
    obj, pts = board.matchImagePoints(cc, ci)
    return (np.asarray(obj, np.float64).reshape(-1,1,3),
            np.asarray(pts, np.float64).reshape(-1,1,2),
            ci.reshape(-1).astype(int), img.shape[1], img.shape[0])


def errors(obs, pred):
    a=np.asarray(obs).reshape(-1,2); b=np.asarray(pred).reshape(-1,2)
    return np.linalg.norm(a-b, axis=1)


def eval_pinhole(obj, pts, K, D):
    ok, rv, tv = cv2.solvePnP(obj, pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None
    pred,_=cv2.projectPoints(obj, rv, tv, K, D)
    return errors(pts,pred)


def eval_fisheye(obj, pts, K, D):
    und=cv2.fisheye.undistortPoints(pts, K, D)
    ok,rv,tv=cv2.solvePnP(obj,und,np.eye(3),None,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return None
    pred,_=cv2.fisheye.projectPoints(obj,rv,tv,K,D)
    return errors(pts,pred)


def stats(vals):
    a=np.asarray(vals,float)
    if len(a)==0: return None
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.percentile(a,95)), max=float(a.max()))


def fmt(s):
    if s is None: return 'n=0'
    return f"n={s['n']:4d} mean={s['mean']:.4f} median={s['median']:.4f} p95={s['p95']:.4f} max={s['max']:.4f}"


def bin_report(rows,key,bins):
    out=[]
    for lo,hi,label in bins:
        subset=[r for r in rows if r[key]>=lo and (math.isinf(hi) or r[key]<hi)]
        ps=stats([r['pinhole_error_px'] for r in subset]); fs=stats([r['fisheye_error_px'] for r in subset])
        win='N/A' if not ps or not fs else ('pinhole' if ps['median']<fs['median'] else 'fisheye')
        out.append((label,ps,fs,win))
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='calibration/captures/cam0_intrinsic')
    ap.add_argument('--compare-dir',default='calibration/results/cam0_model_compare')
    ap.add_argument('--output',default='calibration/reports/cam0_radial_analysis')
    a=ap.parse_args()
    inp=Path(a.input); comp=Path(a.compare_dir); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    model=json.loads((comp/'model_compare.json').read_text(encoding='utf-8'))
    split=json.loads((comp/'split.json').read_text(encoding='utf-8'))
    cfg=model['board']; board,detector=make_board(float(cfg['square_mm']),float(cfg['marker_mm']))
    pK=np.asarray(model['pinhole_rational']['K'],float); pD=np.asarray(model['pinhole_rational']['D'],float).reshape(-1,1)
    fK=np.asarray(model['fisheye']['K'],float); fD=np.asarray(model['fisheye']['D'],float).reshape(-1,1)
    rows=[]; skipped=[]
    for name in split['validation']:
        d=detect(inp/name,board,detector)
        if d is None: skipped.append((name,'detect')); continue
        obj,pts,ids,w,h=d
        pe=eval_pinhole(obj,pts,pK,pD); fe=eval_fisheye(obj,pts,fK,fD)
        if pe is None or fe is None: skipped.append((name,'pose')); continue
        xy=pts.reshape(-1,2); cx=w/2; cy=h/2; hd=math.hypot(cx,cy)
        for (x,y),cid,pv,fv in zip(xy,ids,pe,fe):
            rows.append(dict(file=name,charuco_id=int(cid),x=float(x),y=float(y),
                radius_norm=float(math.hypot(x-cx,y-cy)/hd),
                x_edge_norm=float(abs(x-cx)/cx), y_edge_norm=float(abs(y-cy)/cy),
                horizontal_side='L' if x<cx else 'R', vertical_side='T' if y<cy else 'B',
                pinhole_error_px=float(pv), fisheye_error_px=float(fv),
                fisheye_minus_pinhole_px=float(fv-pv)))
    if not rows: raise RuntimeError('No validation points evaluated')
    csvp=out/'validation_point_errors.csv'
    with csvp.open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)

    rb=[(0,.4,'r < 0.40'),(.4,.6,'0.40 <= r < 0.60'),(.6,.75,'0.60 <= r < 0.75'),(.75,.9,'0.75 <= r < 0.90'),(.9,math.inf,'r >= 0.90')]
    xb=[(0,.5,'|x| < 0.50'),(.5,.75,'0.50 <= |x| < 0.75'),(.75,.9,'0.75 <= |x| < 0.90'),(.9,math.inf,'|x| >= 0.90')]
    yb=[(0,.5,'|y| < 0.50'),(.5,.75,'0.50 <= |y| < 0.75'),(.75,.9,'0.75 <= |y| < 0.90'),(.9,math.inf,'|y| >= 0.90')]
    lines=['=== Cam0 radial / edge validation analysis ===',f'Validation views requested: {len(split["validation"])}',f'Validation points evaluated: {len(rows)}']
    if skipped: lines.append(f'Skipped: {skipped}')
    lines += ['', 'OVERALL', 'Pinhole : '+fmt(stats([r['pinhole_error_px'] for r in rows])), 'Fisheye : '+fmt(stats([r['fisheye_error_px'] for r in rows]))]
    for title,key,bins in [('RADIAL BINS','radius_norm',rb),('HORIZONTAL EDGE BINS','x_edge_norm',xb),('VERTICAL EDGE BINS','y_edge_norm',yb)]:
        lines += ['',title]
        for label,ps,fs,win in bin_report(rows,key,bins):
            lines += [f'[{label}] winner={win}','  Pinhole : '+fmt(ps),'  Fisheye : '+fmt(fs)]
    lines += ['', 'OUTER HORIZONTAL SIDE CHECK (|x| >= 0.75)']
    for side in ['L','R']:
        s=[r for r in rows if r['horizontal_side']==side and r['x_edge_norm']>=.75]
        lines += [f'{side}: n={len(s)}','  Pinhole : '+fmt(stats([r['pinhole_error_px'] for r in s])),'  Fisheye : '+fmt(stats([r['fisheye_error_px'] for r in s]))]
    nr=sum(r['radius_norm']>=.75 for r in rows); nx=sum(r['x_edge_norm']>=.75 for r in rows); ny=sum(r['y_edge_norm']>=.75 for r in rows)
    lines += ['', 'DATA SUFFICIENCY',f'points with r >= 0.75: {nr}',f'points with |x| >= 0.75: {nx}',f'points with |y| >= 0.75: {ny}']
    if nr<30: lines.append('WARNING: too few outer-radial validation points (<30).')
    if nx<30: lines.append('WARNING: too few far-left/right validation points (<30); HFOV remains weakly constrained.')
    if ny<30: lines.append('WARNING: too few top/bottom validation points (<30); VFOV remains weakly constrained.')
    pf=model['pinhole_rational']['raw_fov']; ff=model['fisheye']['raw_fov']
    lines += ['', 'MODEL-DERIVED RAW FOV',f"Pinhole: HFOV={pf['HFOV_deg']:.3f}, VFOV={pf['VFOV_deg']:.3f}",f"Fisheye: HFOV={ff['HFOV_deg']:.3f}, VFOV={ff['VFOV_deg']:.3f}", '', 'Rule: choose based on outer-bin validation, not tiny overall-RMS differences.']
    summary='\n'.join(lines); (out/'summary.txt').write_text(summary,encoding='utf-8')
    print(summary); print('\nSaved:'); print(' ',csvp); print(' ',out/'summary.txt')

if __name__=='__main__': main()
