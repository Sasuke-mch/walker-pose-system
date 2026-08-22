#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

EXTS={".jpg",".jpeg",".png",".webp",".bmp"}
NAMES=["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
def key(p): return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)",p.name)]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input-dir",required=True); ap.add_argument("--weights",required=True); ap.add_argument("--raw-output",required=True); ap.add_argument("--common-output",required=True); ap.add_argument("--vis-dir",required=True)
    ap.add_argument("--device",default="0"); ap.add_argument("--imgsz",type=int,default=1280); ap.add_argument("--conf",type=float,default=0.05); ap.add_argument("--iou",type=float,default=0.7); ap.add_argument("--save-vis",action="store_true")
    a=ap.parse_args(); inp=Path(a.input_dir); images=sorted([p for p in inp.iterdir() if p.is_file() and p.suffix.lower() in EXTS],key=key); model=YOLO(a.weights); raw=[]; common=[]; vis=Path(a.vis_dir)
    if a.save_vis: vis.mkdir(parents=True,exist_ok=True)
    for fi,p in enumerate(images):
        im=cv2.imread(str(p));
        if im is None: raise RuntimeError(f"Cannot read image: {p}")
        r=model.predict(source=im,device=a.device,imgsz=a.imgsz,conf=a.conf,iou=a.iou,save=False,verbose=False)[0]
        boxes=r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0,4)); bs=r.boxes.conf.cpu().numpy() if r.boxes is not None else np.empty((0,)); kd=r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.empty((0,17,3))
        ri=[]; ci=[]
        for pid,(box,bscore,kp) in enumerate(zip(boxes,bs,kd)):
            ri.append({"person_id":pid,"bbox_xyxy":[float(v) for v in box],"bbox_score":float(bscore),"keypoints":kp.astype(float).tolist()})
            ci.append({"person_id":pid,"bbox_xyxy":[float(v) for v in box],"bbox_score":float(bscore),"keypoints":[{"name":name,"x":float(v[0]),"y":float(v[1]),"score":float(v[2]) if len(v)>2 else 1.0} for name,v in zip(NAMES,kp)]})
        raw.append({"frame_index":fi,"file_name":p.name,"width":int(im.shape[1]),"height":int(im.shape[0]),"instances":ri}); common.append({"frame_index":fi,"file_name":p.name,"timestamp":None,"camera_id":None,"width":int(im.shape[1]),"height":int(im.shape[0]),"instances":ci})
        if a.save_vis: cv2.imwrite(str(vis/f"{p.stem}.jpg"),r.plot())
        print(f"[{fi+1}/{len(images)}] {p.name}: {len(ri)} persons")
    rp=Path(a.raw_output); rp.parent.mkdir(parents=True,exist_ok=True); rp.write_text(json.dumps({"schema_version":"1.0","model":"YOLO26x-pose","weights":a.weights,"frames":raw},ensure_ascii=False,indent=2),encoding="utf-8")
    cp=Path(a.common_output); cp.parent.mkdir(parents=True,exist_ok=True); cp.write_text(json.dumps({"schema_version":"1.0","model":{"name":"YOLO26x-pose","detector":"YOLO26x-pose","keypoint_format":"COCO17"},"sequence":{"input_dir":str(inp),"num_frames":len(common),"fps":None},"frames":common},ensure_ascii=False,indent=2),encoding="utf-8")
    print("Saved raw:",rp); print("Saved common:",cp)
if __name__=="__main__": main()
