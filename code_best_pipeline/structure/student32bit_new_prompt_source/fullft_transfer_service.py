"""Private-interface, random-token transfer of one exact experiment bundle/result."""
from pathlib import Path
import json,time,hashlib,os,tarfile,zlib
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
ROOT=Path('/var/lib/structure32_dit_sweep_20261101')
while not (ROOT/'transport.json').exists():time.sleep(3)
M=json.loads((ROOT/'transport.json').read_text())
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path!='/'+M['token']+'/bundle':self.send_error(404);return
        p=Path(M['bundle']);start=0;end=p.stat().st_size-1
        if self.headers.get('Range'):
            a,b=self.headers['Range'].removeprefix('bytes=').split('-');start=int(a);end=int(b)
            assert 0<=start<=end<p.stat().st_size
            if self.headers.get('X-Compress')=='zlib':
                with p.open('rb') as f:f.seek(start);data=zlib.compress(f.read(end-start+1),1)
                self.send_response(206);self.send_header('Content-Range',f'bytes {start}-{end}/{p.stat().st_size}')
                self.send_header('Content-Length',str(len(data)));self.send_header('X-Compress','zlib');self.end_headers();self.wfile.write(data);return
            self.send_response(206);self.send_header('Content-Range',f'bytes {start}-{end}/{p.stat().st_size}')
        else:self.send_response(200)
        self.send_header('Content-Length',str(end-start+1));self.end_headers()
        with p.open('rb') as f:
            f.seek(start);remaining=end-start+1
            while remaining:
                chunk=f.read(min(4*1024*1024,remaining));self.wfile.write(chunk);remaining-=len(chunk)
    def do_POST(self):
        if self.path!='/'+M['token']+'/result':self.send_error(404);return
        n=int(self.headers.get('Content-Length','0'))
        if not 0<n<6*1024**3:self.send_error(400);return
        p=ROOT/'pixart_fullft_result.partial.tar';h=hashlib.sha256()
        with p.open('wb') as f:
            while n:
                b=self.rfile.read(min(n,4*1024*1024))
                if not b:raise EOFError()
                f.write(b);h.update(b);n-=len(b)
        if h.hexdigest()!=self.headers.get('X-Content-SHA256'):p.unlink();self.send_error(400);return
        dest=ROOT/'pixart512_fullft/result_from_23';dest.mkdir(exist_ok=True)
        with tarfile.open(p) as t:
            for m in t.getmembers():
                target=(dest/m.name).resolve()
                assert dest==target or dest in target.parents
                assert m.isfile()
            t.extractall(dest,filter='data')
        (ROOT/'pixart512_fullft/result_ready.json').write_text(json.dumps({'sha256':h.hexdigest(),'time':time.time(),'source':'.23'}))
        p.unlink();self.send_response(200);self.end_headers();self.wfile.write(b'VERIFIED')
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('10.1.21.21',M['port']),Handler)
server.timeout=5
deadline=time.time()+24*3600
while time.time()<deadline and not (ROOT/'pixart512_fullft/result_ready.json').exists():server.handle_request()
server.server_close()
bundle=Path(M['bundle'])
if (ROOT/'pixart512_fullft/result_ready.json').exists():bundle.unlink(missing_ok=True)
