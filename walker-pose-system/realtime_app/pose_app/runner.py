from __future__ import annotations
from dataclasses import dataclass
import logging
import threading
import time
import cv2
from .http_client import PoseServiceClient
from .output import OutputWriter
from .sources import FrameSource, SourceFrame
from .visualizer import draw

LOG = logging.getLogger(__name__)

class LatestFrame:
    def __init__(self) -> None:
        self.condition = threading.Condition(); self.generation=0; self.frame=None; self.closed=False
    def put(self, frame: SourceFrame) -> None:
        with self.condition:
            self.generation += 1; self.frame=frame; self.condition.notify_all()
    def wait_newer(self, generation: int, timeout: float=.25):
        deadline=time.monotonic()+timeout
        with self.condition:
            while not self.closed and (self.frame is None or self.generation<=generation):
                remain=deadline-time.monotonic()
                if remain<=0:return None
                self.condition.wait(remain)
            if self.closed or self.frame is None:return None
            return self.generation,self.frame
    def close(self):
        with self.condition:self.closed=True;self.condition.notify_all()

@dataclass
class RunOptions:
    simulate_realtime: bool
    max_frames: int | None
    preview: bool
    loop: bool
    output_fps: float
    draw_threshold: float

class Runner:
    def __init__(self, source: FrameSource, client: PoseServiceClient,
                 writer: OutputWriter, options: RunOptions) -> None:
        self.source=source; self.client=client; self.writer=writer; self.options=options
        self.stop_event=threading.Event(); self.producer_error=None

    def _preview(self, image) -> bool:
        if not self.options.preview:return True
        cv2.imshow("YOLO26x-pose video test",image)
        key=cv2.waitKey(1)&0xFF
        if key in (27,ord('q'),ord('Q')):return False
        if key==ord(' '):
            while True:
                key=cv2.waitKey(50)&0xFF
                if key in (27,ord('q'),ord('Q')):return False
                if key==ord(' '):break
        return True

    def run(self) -> dict:
        try:
            return self._run_realtime() if self.options.simulate_realtime or self.source.is_live else self._run_sequential()
        finally:
            self.source.close()
            if self.options.preview:cv2.destroyAllWindows()

    def _run_sequential(self) -> dict:
        processed=0; dropped=0
        while not self.stop_event.is_set():
            frame=self.source.read()
            if frame is None:break
            result=self.client.infer(frame.image,frame.frame_id,frame.timestamp_sec,0)
            processed+=1
            annotated=draw(frame.image,result,self.options.draw_threshold,processed,dropped,"sequential")
            self.writer.write(annotated,result,dropped)
            if not self._preview(annotated):break
            if self.options.max_frames is not None and processed>=self.options.max_frames:break
        return self.writer.close()

    def _producer(self, slot: LatestFrame) -> None:
        try:
            start=time.monotonic(); first_timestamp=None
            while not self.stop_event.is_set():
                frame=self.source.read()
                if frame is None:break
                if not self.source.is_live:
                    if first_timestamp is None:first_timestamp=frame.timestamp_sec
                    target=start+(frame.timestamp_sec-first_timestamp)
                    wait=target-time.monotonic()
                    if wait>0:self.stop_event.wait(wait)
                if self.stop_event.is_set():break
                slot.put(frame)
        except BaseException as exc:self.producer_error=exc
        finally:slot.close()

    def _run_realtime(self) -> dict:
        slot=LatestFrame(); thread=threading.Thread(target=self._producer,args=(slot,),daemon=True)
        thread.start(); generation=0; previous_id=None; processed=0; dropped=0
        try:
            while not self.stop_event.is_set():
                item=slot.wait_newer(generation)
                if item is None:
                    if self.producer_error:raise RuntimeError(str(self.producer_error))
                    if slot.closed:break
                    continue
                generation,frame=item
                if previous_id is not None:dropped += max(0,frame.frame_id-previous_id-1)
                previous_id=frame.frame_id
                result=self.client.infer(frame.image,frame.frame_id,frame.timestamp_sec,dropped)
                processed+=1
                annotated=draw(frame.image,result,self.options.draw_threshold,processed,dropped,"realtime simulation")
                self.writer.write(annotated,result,dropped)
                if not self._preview(annotated):break
                if self.options.max_frames is not None and processed>=self.options.max_frames:break
        finally:
            self.stop_event.set();thread.join(timeout=3.0)
        return self.writer.close()
