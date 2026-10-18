"""Render Phase 1 overlays: observed COCO 2D, predicted model 2D, solid residuals.

Reads phase1_<model>_fit.json, projects final model joints to left/right raw
fisheye, draws obs (green) / pred (red) / solid line between them. Also reports
strict/relaxed/rejected/unavailable and left/right median/P95 pixel errors.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
REPO = APP_ROOT.parent
sys.path.insert(0, str(APP_ROOT))
from pose_app.project_paths import project_path
from realtime_app.pose_app.fisheye_camera import load_stereo_fisheye, fisheye_project_numpy

CAL = project_path("realtime_app", "calibration", "results")
IMG = project_path("research_records", "engineering_validation",
                   "V20260908_people0_1_2_pmpose_c3_chain", "people_1",
                   "input_448pairs")
OUT = project_path("research_records", "engineering_validation",
                   "G20260919_alternative_surface_models_v1")
HELD = {9, 16}
COCO_NAMES = ["nose","leye","reye","lear","rear","lsho","rsho","lelb","relb","lwri","rwri",
              "lhip","rhip","lkne","rkne","lank","rank"]

def main():
    model = sys.argv[1]
    data = json.loads((OUT / f"phase1_{model.lower()}_fit.json").read_text(encoding="utf-8"))
    cal = load_stereo_fisheye(CAL)
    K0,D0,K1,D1 = cal.K0,cal.D0,cal.K1,cal.D1
    Rr,Tr = cal.R_cam0_to_cam1, cal.T_cam0_to_cam1_mm
    H0,W0 = cal.image_height, cal.image_width
    coco_to_model = json.loads((OUT / "phase1_coco_map.json").read_text(encoding="utf-8"))[model]

    def lproj(uv):
        return np.stack([uv[:,1], W0-1-uv[:,0]],1)
    def rproj(uv):
        return np.stack([H0-1-uv[:,1], uv[:,0]],1)

    from PIL import Image, ImageDraw
    stats = {"strict":[], "relaxed":[]}
    for row in data:
        if row["outcome"] != "matched":
            continue
        f = row["frame"]; jc = np.asarray(row["joints_cam_mm"])
        obs_l = np.asarray(row["obs_l"]); obs_r = np.asarray(row["obs_r"]); valid = row["valid"]
        img_l = Image.open(IMG/"left_ccw90"/f"pair_{f:04d}.png").convert("RGB")
        img_r = Image.open(IMG/"right_cw90"/f"pair_{f:04d}.png").convert("RGB")
        dl,dr = ImageDraw.Draw(img_l), ImageDraw.Draw(img_r)
        for ci, mj in coco_to_model.items():
            ci=int(ci)
            if not valid[ci] or ci in HELD:
                continue
            p = jc[mj:mj+1]
            ul = fisheye_project_numpy(p,K0,D0)[0]; ul = lproj(ul[None,:])[0]
            pr = (Rr@p[0]+Tr); ur = fisheye_project_numpy(pr[None,:],K1,D1)[0]; ur = rproj(ur[None,:])[0]
            ol, orr = obs_l[ci], obs_r[ci]
            ol = np.asarray([ol[1], W0-1-ol[0]])
            orr = np.asarray([H0-1-orr[1], orr[0]])
            for d,px,ox,side in ((dl,ul,ol,"L"),(dr,ur,orr,"R")):
                d.line([tuple(px),tuple(ox)],fill=(255,255,0),width=2)
                d.ellipse([px[0]-4,px[1]-4,px[0]+4,px[1]+4],outline=(255,0,0),width=2)
                d.ellipse([ox[0]-4,ox[1]-4,ox[0]+4,ox[1]+4],outline=(0,255,0),width=2)
                d.text((px[0]+6,px[1]-6),COCO_NAMES[ci],fill=(255,0,0))
            el = float(np.hypot(*(ul-ol))); er = float(np.hypot(*(ur-orr)))
            stats["strict"].append((f,ci,"L",el)); stats["relaxed"].append((f,ci,"R",er))
        img_l.save(OUT/f"phase1_{model.lower()}_final_left_{f:03d}.png")
        img_r.save(OUT/f"phase1_{model.lower()}_final_right_{f:03d}.png")
    # summarize
    def pct(a,q): a=np.asarray(a); return float(np.percentile(a,q)) if len(a) else 0.0
    le = [s[3] for s in stats["strict"] if s[2]=="L"]
    re_ = [s[3] for s in stats["relaxed"] if s[2]=="R"]
    summ = {"frames":len(data),
            "matched":sum(1 for r in data if r["outcome"]=="matched"),
            "unavailable":sum(1 for r in data if r["outcome"]=="unavailable"),
            "rejected":sum(1 for r in data if r["outcome"]=="rejected"),
            "left_median_px":round(pct(le,50),1),"left_p95_px":round(pct(le,95),1),
            "right_median_px":round(pct(re_,50),1),"right_p95_px":round(pct(re_,95),1)}
    (OUT/f"phase1_{model.lower()}_summary.json").write_text(json.dumps(summ,indent=2),encoding="utf-8")
    print(json.dumps(summ,indent=2))

if __name__=="__main__":
    main()
