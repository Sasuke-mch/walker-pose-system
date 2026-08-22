#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import cv2
from ultralytics import YOLO

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
def key(p): return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--output-json", required=True)
    ap.add_argument("--device", default="0"); ap.add_argument("--imgsz", type=int, default=1280); ap.add_argument("--conf", type=float, default=0.05); ap.add_argument("--iou", type=float, default=0.7)
    a = ap.parse_args(); inp = Path(a.input_dir); images = sorted([p for p in inp.iterdir() if p.is_file() and p.suffix.lower() in EXTS], key=key)
    model = YOLO(a.weights); records=[]
    for i,p in enumerate(images):
        im=cv2.imread(str(p));
        if im is None: raise RuntimeError(f"Cannot read image: {p}")
        r=model.predict(source=im, device=a.device, imgsz=a.imgsz, conf=a.conf, iou=a.iou, save=False, verbose=False)[0]
        det=[]
        if r.boxes is not None:
            for box,score,cls in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                if int(cls)==0: det.append({"class_id":0,"class_name":"person","bbox_xyxy":[float(v) for v in box],"score":float(score)})
        records.append({"image_id":i,"frame_index":i,"file_name":p.name,"image_path":f"/workspace/input/{p.name}","width":int(im.shape[1]),"height":int(im.shape[0]),"detections":det})
        print(f"[{i+1}/{len(images)}] {p.name}: {len(det)} persons")
    out=Path(a.output_json); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({"schema_version":"1.0","detector":"YOLO26x","images":records},ensure_ascii=False,indent=2),encoding="utf-8")
    print("Saved:",out)
if __name__=="__main__": main()
