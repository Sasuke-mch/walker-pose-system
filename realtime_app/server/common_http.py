from __future__ import annotations
import base64
import json
import logging
import signal
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
import cv2
import numpy as np

LOG=logging.getLogger("pose-server")

def decode(payload: dict[str,Any]) -> np.ndarray:
    value=payload.get("image_base64")
    if not isinstance(value,str) or not value:raise ValueError("缺少image_base64")
    raw=base64.b64decode(value,validate=True)
    image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError("JPEG解码失败")
    return image

class Handler(BaseHTTPRequestHandler):
    adapter=None
    def log_message(self,fmt,*args):LOG.info("%s - %s",self.client_address[0],fmt%args)
    def send_json(self,status:int,data:dict[str,Any]):
        body=json.dumps(data,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode("utf-8")
        self.send_response(status);self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)
    def do_GET(self):
        if self.path!="/health":self.send_json(404,{"ok":False,"error":"not found"});return
        data={"ok":True,"status":"ready","model":self.adapter.model_name};data.update(self.adapter.health())
        self.send_json(200,data)
    def do_POST(self):
        if self.path!="/infer":self.send_json(404,{"ok":False,"error":"not found"});return
        try:
            length=int(self.headers.get("Content-Length","0"))
            if length<=0 or length>20*1024*1024:raise ValueError("请求大小异常")
            payload=json.loads(self.rfile.read(length).decode("utf-8"))
            image=decode(payload);start=time.perf_counter();result=self.adapter.infer(image,payload)
            data={"ok":True,"service_ms":(time.perf_counter()-start)*1000.0};data.update(result)
            self.send_json(200,data)
        except Exception as exc:
            LOG.exception("推理请求失败")
            self.send_json(500,{"ok":False,"error":str(exc),"traceback":traceback.format_exc(limit=8)})

def run_server(adapter,host:str,port:int):
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    class Bound(Handler):pass
    Bound.adapter=adapter;server=ThreadingHTTPServer((host,port),Bound);server.daemon_threads=True
    def stop(signum,frame):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    LOG.info("服务已就绪：%s",adapter.model_name)
    try:server.serve_forever(.25)
    finally:server.server_close()
