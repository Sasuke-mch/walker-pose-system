from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
DATA = Path(__file__).with_name("body.html").read_bytes()
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/body.html"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(DATA)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(DATA)
HTTPServer(("127.0.0.1", 8767), Handler).serve_forever()
