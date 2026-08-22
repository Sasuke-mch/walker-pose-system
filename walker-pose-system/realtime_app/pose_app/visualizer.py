from __future__ import annotations
import cv2
import numpy as np
from .schema import InferenceResult

EDGES = ((0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))
COLORS = ((0,220,255),(255,180,0),(80,255,80),(255,100,180),(180,120,255))


def draw(image: np.ndarray, result: InferenceResult, threshold: float,
         processed: int, dropped: int, mode: str) -> np.ndarray:
    output = image.copy()
    for person in result.persons:
        color = COLORS[person.person_id % len(COLORS)]
        x1,y1,x2,y2 = [int(round(v)) for v in person.bbox]
        cv2.rectangle(output,(x1,y1),(x2,y2),color,2)
        cv2.putText(output,f"id={person.person_id} pose={person.pose_score:.2f}",
                    (x1,max(20,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)
        for a,b in EDGES:
            pa,pb = person.keypoints[a], person.keypoints[b]
            if pa[2] >= threshold and pb[2] >= threshold:
                cv2.line(output,(round(pa[0]),round(pa[1])),(round(pb[0]),round(pb[1])),
                         color,2,cv2.LINE_AA)
        for x,y,score in person.keypoints:
            if score >= threshold:
                cv2.circle(output,(round(x),round(y)),4,color,-1,cv2.LINE_AA)
    stage = result.stage_times_ms
    if "detector_ms" in stage:
        timing = f"det: {stage['detector_ms']:.1f} ms | pose: {stage.get('pose_ms', 0.0):.1f} ms | rt: {result.roundtrip_ms:.1f} ms"
    else:
        timing = f"model: {result.model_ms:.1f} ms | roundtrip: {result.roundtrip_ms:.1f} ms"
    lines = [
        f"{result.model_name} | {mode}",
        f"source frame: {result.source_frame_id} | processed: {processed} | dropped: {dropped}",
        f"persons: {len(result.persons)} | {timing}",
        "Q/Esc: stop  Space: pause preview",
    ]
    overlay = output.copy(); width = min(output.shape[1]-16, 1000); height = 30*len(lines)+10
    cv2.rectangle(overlay,(8,8),(8+width,8+height),(0,0,0),-1)
    cv2.addWeighted(overlay,.55,output,.45,0,output)
    for i,line in enumerate(lines):
        cv2.putText(output,line,(18,34+i*30),cv2.FONT_HERSHEY_SIMPLEX,.65,
                    (255,255,255),1,cv2.LINE_AA)
    return output
