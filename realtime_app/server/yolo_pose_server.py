from __future__ import annotations
import argparse
import time
from typing import Any
import numpy as np
import torch
from ultralytics import YOLO
from common_http import run_server

def sync(device: str) -> None:
    if torch.cuda.is_available() and str(device).lower() not in {"cpu","mps"}:torch.cuda.synchronize()

def pose_score(bbox_score: float, scores: np.ndarray, threshold: float) -> float:
    valid=scores>threshold
    mean=float(np.mean(scores[valid])) if np.any(valid) else 0.0
    return float(bbox_score*mean)

class Adapter:
    model_name="YOLO26x-pose"
    def __init__(self,args):
        self.args=args;self.model=YOLO(args.weights)
        self.model.predict(source=np.zeros((640,640,3),dtype=np.uint8),device=args.device,
            imgsz=args.imgsz,conf=args.conf,iou=args.iou,max_det=args.max_det,verbose=False)
        sync(args.device)
    def health(self)->dict[str,Any]:
        return {"weights":self.args.weights,"device":self.args.device,
                "pose_threshold":self.args.pose_thr}
    def infer(self,image,payload):
        sync(self.args.device);start=time.perf_counter()
        results=self.model.predict(source=image,device=self.args.device,imgsz=self.args.imgsz,
            conf=self.args.conf,iou=self.args.iou,max_det=self.args.max_det,verbose=False)
        sync(self.args.device);model_ms=(time.perf_counter()-start)*1000.0;persons=[]
        if results:
            result=results[0]
            if result.boxes is not None and result.keypoints is not None and len(result.boxes)>0:
                boxes=result.boxes.xyxy.detach().cpu().numpy()
                bbox_scores=result.boxes.conf.detach().cpu().numpy()
                xy=result.keypoints.xy.detach().cpu().numpy()
                scores=(result.keypoints.conf.detach().cpu().numpy() if result.keypoints.conf is not None
                        else np.ones(xy.shape[:2],dtype=np.float32))
                count=min(len(boxes),len(bbox_scores),len(xy),len(scores))
                for index in range(count):
                    points=np.asarray(xy[index][:17],dtype=np.float32)
                    kp_scores=np.asarray(scores[index][:17],dtype=np.float32)
                    if points.shape!=(17,2):continue
                    score=pose_score(float(bbox_scores[index]),kp_scores,self.args.keypoint_score_thr)
                    if score<self.args.pose_thr:continue
                    persons.append({"person_id":len(persons),"bbox":[float(v) for v in boxes[index][:4]],
                        "bbox_score":float(bbox_scores[index]),"pose_score":score,
                        "keypoints":np.concatenate([points,kp_scores[:,None]],axis=1).astype(float).tolist()})
        h,w=image.shape[:2]
        return {"model":self.model_name,"source_frame_id":payload.get("source_frame_id"),
                "image_width":w,"image_height":h,"model_ms":model_ms,"persons":persons}

def arguments():
    p=argparse.ArgumentParser();p.add_argument("--weights",required=True)
    p.add_argument("--device",default="0");p.add_argument("--imgsz",type=int,default=1280)
    p.add_argument("--conf",type=float,default=.01);p.add_argument("--iou",type=float,default=.7)
    p.add_argument("--max-det",type=int,default=300);p.add_argument("--keypoint-score-thr",type=float,default=.2)
    p.add_argument("--pose-thr",type=float,default=.4);p.add_argument("--host",default="0.0.0.0")
    p.add_argument("--port",type=int,default=18080);return p.parse_args()
if __name__=="__main__":
    args=arguments();run_server(Adapter(args),args.host,args.port)
