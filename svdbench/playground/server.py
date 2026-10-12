"""Loopback-only SVD playground. Start with python -m svdbench.playground.server."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse
import uuid
import numpy as np
import scipy
import torch
from svdbench.standalone import ALGORITHMS
from .engine import normalize, request_spec

STATIC=Path(__file__).parent/'static'


def atomic_json(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,allow_nan=False));tmp.replace(path)


class Manager:
    def __init__(self,root):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock();self.active=None;self.proc=None;self.cancelled=threading.Event()
        # Interrupted experiments remain visible after a server restart.
        for p in self.root.glob('*/state.json'):
            state=json.loads(p.read_text())
            if state['status'] in ('queued','running'):
                state.update(status='interrupted',message='Server stopped before this experiment finished')
                atomic_json(p,state)

    def state(self,job):
        if len(job)!=32 or any(c not in '0123456789abcdef' for c in job): raise ValueError('Invalid experiment id')
        path=self.root/job/'state.json'
        if not path.exists(): raise FileNotFoundError('Experiment not found')
        return json.loads(path.read_text())

    def start(self,raw):
        cfg=normalize(raw)
        sizes=raw.get('sweep_sizes',[])
        if not isinstance(sizes,list) or len(sizes)>8 or any(isinstance(n,bool) or not isinstance(n,int) or not 1<=n<=512 for n in sizes):
            raise ValueError('Sweep sizes must be up to eight integers between 1 and 512')
        configs=[]
        for size in sizes:
            request={**request_spec(cfg),'m':size,'n':size,'matrix_rank':min(cfg['case']['rank'],size),'target_rank':min(cfg['rank'],size),'steps':min(cfg['steps'],size)}
            configs.append(normalize(request))
        configs=configs or [cfg]
        with self.lock:
            if self.active is not None: raise RuntimeError('An experiment is already running. Cancel it or wait for completion.')
            job=uuid.uuid4().hex;path=self.root/job;path.mkdir()
            state={'id':job,'created':datetime.now(timezone.utc).isoformat(),'status':'queued','request':raw,
                   'configs':configs,'rows':[],'previews':{},'message':'Preparing matrix and independent reference',
                   'completed':0,'total':sum(len(c['algorithms']) for c in configs),
                   'environment':{'python':sys.version.split()[0],'torch':torch.__version__,'numpy':np.__version__,
                                  'scipy':scipy.__version__,'device':cfg['device'],
                                  'accelerator':torch.cuda.get_device_name(0) if cfg['device']=='cuda:0' else None,
                                  'tf32':False,'timing':'Resident synchronized wall and CUDA events; preparation and CPU validation excluded.'}}
            atomic_json(path/'state.json',state)
            self.active=job;self.cancelled.clear()
            threading.Thread(target=self.run,args=(job,),daemon=True).start()
        return state

    def child(self,stage,path,cfg,name=None):
        if self.cancelled.is_set(): return 'cancelled'
        cmd=[sys.executable,'-m','svdbench.playground.worker',stage,str(path)]
        if name: cmd.append(name)
        with (path/f'{stage}-{name or "matrix"}.log').open('wb') as log:
            with self.lock:
                if self.cancelled.is_set(): return 'cancelled'
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                self.proc=proc
            try:
                code=proc.wait(timeout=cfg['timeout'])
                return 'cancelled' if self.cancelled.is_set() else ('ok' if code==0 else 'error')
            except subprocess.TimeoutExpired:
                proc.kill();proc.wait();return 'timeout'
            finally:
                with self.lock: self.proc=None

    def run(self,job):
        root=self.root/job;state=self.state(job)
        def save(**kw):
            state.update(kw);atomic_json(root/'state.json',state)
        try:
            save(status='running')
            for index,cfg in enumerate(state['configs']):
                path=root/str(index);path.mkdir();atomic_json(path/'config.json',cfg)
                save(message=f'Preparing {cfg["case"]["m"]} × {cfg["case"]["n"]} matrix and CPU reference')
                status=self.child('prepare',path,cfg)
                if status!='ok':
                    log=(path/'prepare-matrix.log')
                    save(status=status if status in ('cancelled','timeout') else 'error',
                         message=f'Matrix/reference preparation {status}.',
                         error_log=log.read_text(errors='replace')[-8000:] if log.exists() else '');return
                state['previews'][str(index)]=json.loads((path/'preview.json').read_text())
                save()
                for name in cfg['algorithms']:
                    if self.cancelled.is_set(): save(status='cancelled',message='Cancelled');return
                    save(message=f'{name} · {cfg["case"]["m"]} × {cfg["case"]["n"]}')
                    status=self.child('solve',path,cfg,name)
                    if status=='cancelled': save(status='cancelled',message='Cancelled');return
                    result=path/f'{name}.json'
                    if status=='ok' and result.exists(): row=json.loads(result.read_text())
                    else:
                        log=path/f'solve-{name}.log'
                        row={'id':name,'status':status,'error':f'{name}: {status} (per-stage limit {cfg["timeout"]:g}s)',
                             'worker_log':log.read_text(errors='replace')[-8000:] if log.exists() else '',
                             'assessment':{'passed':False,'reasons':[status]}}
                    row['case_index']=index;row['shape']=[cfg['case']['batch'],cfg['case']['m'],cfg['case']['n']]
                    state['rows'].append(row);state['completed']+=1;save()
            save(status='complete',message='Experiment complete')
        except Exception as exc:
            save(status='error',message=f'{type(exc).__name__}: {exc}')
        finally:
            with self.lock:self.active=None

    def cancel(self,job):
        with self.lock:
            if self.active!=job: raise ValueError('This experiment is not running')
            self.cancelled.set()
            if self.proc is not None and self.proc.poll() is None:self.proc.terminate()

    def history(self):
        rows=[]
        for path in sorted(self.root.glob('*/state.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:50]:
            s=json.loads(path.read_text());rows.append({k:s[k] for k in ('id','created','status','message','completed','total','request')})
        return rows


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass

    def send(self,data,status=200,mime='application/json',filename=None):
        if not isinstance(data,bytes):data=json.dumps(data,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        if filename:self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
        self.end_headers();self.wfile.write(data)

    def valid_host(self):
        host=self.headers.get('Host','')
        return host in (f'localhost:{self.server.server_port}',f'127.0.0.1:{self.server.server_port}')

    def do_GET(self):
        if not self.valid_host():self.send({'error':'Invalid Host'},403);return
        path=urlparse(self.path).path
        try:
            if path=='/api/info':
                cuda=torch.cuda.is_available()
                self.send({'token':self.server.token,'devices':['cpu']+(['cuda:0'] if cuda else []),
                           'gpu':torch.cuda.get_device_name(0) if cuda else None,'torch':torch.__version__,
                           'algorithms':list(ALGORITHMS)+['sparse_lanczos','vendor_default']+(['vendor_gesvd','vendor_gesvdj','vendor_gesvda'] if cuda and not torch.version.hip else []),
                           'history':self.server.manager.history()});return
            if path=='/api/history':self.send(self.server.manager.history());return
            parts=path.strip('/').split('/')
            if len(parts)>=3 and parts[:2]==['api','jobs']:
                job=parts[2];state=self.server.manager.state(job)
                if len(parts)==3:self.send(state);return
                if len(parts)==4 and parts[3]=='download':self.send(state,filename=f'svd-{job}.json');return
                if len(parts)==5 and parts[3]=='matrix' and parts[4].isdigit():
                    index=int(parts[4])
                    if str(index) not in state['previews']:raise FileNotFoundError('Matrix not ready')
                    self.send((self.server.manager.root/job/str(index)/'matrix.npy').read_bytes(),mime='application/octet-stream',filename=f'svd-{job}-{index}.npy');return
            files={'/':'index.html','/app.js':'app.js','/style.css':'style.css'}
            if path in files:
                mime={'/':'text/html; charset=utf-8','/app.js':'text/javascript; charset=utf-8','/style.css':'text/css; charset=utf-8'}[path]
                self.send((STATIC/files[path]).read_bytes(),mime=mime);return
            self.send({'error':'Not found'},404)
        except (ValueError,FileNotFoundError) as exc:self.send({'error':str(exc)},404)

    def do_POST(self):
        origin=self.headers.get('Origin')
        valid_origin=origin in (None,f'http://localhost:{self.server.server_port}',f'http://127.0.0.1:{self.server.server_port}')
        if not self.valid_host() or not valid_origin or not secrets.compare_digest(self.headers.get('X-Playground-Token',''),self.server.token):
            self.send({'error':'Invalid local session'},403);return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=65536:raise ValueError('Request must be 1–65536 bytes')
            raw=json.loads(self.rfile.read(length))
            path=urlparse(self.path).path
            if path=='/api/jobs':self.send(self.server.manager.start(raw),202);return
            parts=path.strip('/').split('/')
            if len(parts)==4 and parts[:2]==['api','jobs'] and parts[3]=='cancel':
                self.server.manager.cancel(parts[2]);self.send({'cancelled':True});return
            self.send({'error':'Not found'},404)
        except RuntimeError as exc:self.send({'error':str(exc)},409)
        except (ValueError,TypeError) as exc:self.send({'error':str(exc)},400)


def make_server(port=8765,root='.playground'):
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.manager=Manager(root);server.token=secrets.token_urlsafe(32)
    return server


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8765)
    p.add_argument('--data-dir',default='.playground');args=p.parse_args()
    server=make_server(args.port,args.data_dir)
    print(f'SVD playground: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        if server.manager.active:server.manager.cancel(server.manager.active)
        server.server_close()

if __name__=='__main__':main()
