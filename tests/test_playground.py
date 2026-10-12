import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from unittest.mock import patch
import numpy as np
import torch
from svdbench.matrices import make_matrix
from svdbench.metrics import reference,validate
from svdbench.playground.engine import normalize,request_spec,factory,algorithm_spec,plot_data,assess
from svdbench.playground.server import Manager,make_server


class PlaygroundNumerics(unittest.TestCase):
    def test_configuration_validation(self):
        for raw in [dict(m=0),dict(n=4097),dict(m=1024,n=1024),dict(repeat=0),dict(density=2),dict(seed=True),dict(algorithms=['unknown']),dict(algorithms=['jacobi','jacobi']),dict(m=float('nan')),dict(timeout=1),dict(device='shell')]:
            with self.subTest(raw=raw),self.assertRaises(ValueError):normalize(raw)
        cfg=normalize(dict(family='clustered',cluster_spread=.1,output='full',algorithms=['jacobi']))
        self.assertEqual(cfg,normalize(request_spec(cfg)))
        with self.assertRaises(ValueError):normalize(dict(m=4096,n=512,batch=4,dtype='complex128',output='full'))

    def test_cluster_generator_and_sparse_reproducibility(self):
        base=dict(m=16,n=12,family='clustered',dtype='float64',cluster_size=3,condition=100,seed=1)
        exact=make_matrix({**base,'cluster_spread':0})
        spread=make_matrix({**base,'cluster_spread':.01})
        s=np.linalg.svd(exact,compute_uv=False)
        self.assertLess(np.ptp(s[:3]),1e-14)
        self.assertGreater(np.ptp(np.linalg.svd(spread,compute_uv=False)[:3]),.009)
        self.assertTrue(np.array_equal(exact,make_matrix(base)))
        for dtype in ('float32','complex128'):
            case=dict(m=40,n=30,family='sparse',density=.1,dtype=dtype,seed=9)
            a=make_matrix(case);self.assertTrue(np.array_equal(a,make_matrix(case)))
            self.assertGreater(np.count_nonzero(a),60);self.assertLess(np.count_nonzero(a),190)
            self.assertEqual(np.count_nonzero(make_matrix({**case,'density':0})),0)

    def test_sparse_callback_real_complex_batch_and_modes(self):
        for dtype in ('float64','complex128'):
            for mode in ('topk','reduced','full','values'):
                cfg=normalize(dict(m=12,n=7,batch=2,family='sparse',density=.3,dtype=dtype,
                                   algorithms=['sparse_lanczos'],output=mode,target_rank=3,steps=7))
                a=make_matrix(cfg['case']); fn=factory(torch.from_numpy(a),algorithm_spec(cfg,'sparse_lanczos'))
                result=tuple(x.resolve_conj().resolve_neg().numpy() if x is not None else None for x in fn())
                row={'status':'ok','metrics':validate(a,result,reference(a))}
                self.assertTrue(assess(row,cfg)['passed'],(dtype,mode,row))
                self.assertTrue(fn.metadata['sparse_storage'])

    def test_preview_and_topk_baseline(self):
        cfg=normalize(dict(m=9,n=7,dtype='complex128',target_rank=3,algorithms=['vendor_default']))
        a=make_matrix(cfg['case'])
        u,s,vh=factory(torch.from_numpy(a),algorithm_spec(cfg,'vendor_default'))()
        self.assertEqual(s.shape,(3,))
        data=plot_data(a,s.numpy(),u.resolve_conj().numpy(),vh.resolve_conj().numpy())
        self.assertEqual(len(data['reconstruction']),9)
        self.assertTrue(data['complex_magnitude'])
        row={'status':'ok','metrics':validate(a,(u.numpy(),s.numpy(),vh.resolve_conj().numpy()),reference(a))}
        self.assertTrue(assess(row,cfg)['passed'])
        row['metrics']['worst']['orthogonality_u_fro_over_sqrt_cols']=1
        self.assertFalse(assess(row,cfg)['passed'])


class PlaygroundLifecycle(unittest.TestCase):
    def test_timeout_reaps_worker_and_releases_handle(self):
        with tempfile.TemporaryDirectory() as root,patch('svdbench.playground.server.subprocess.Popen') as spawn:
            proc=spawn.return_value
            proc.wait.side_effect=[subprocess.TimeoutExpired('worker',5),-9]
            manager=Manager(root)
            self.assertEqual(manager.child('solve',Path(root),{'timeout':5},'jacobi'),'timeout')
            proc.kill.assert_called_once()
            self.assertEqual(proc.wait.call_count,2)
            self.assertIsNone(manager.proc)

    def test_sweep_and_cancel_without_solver_threads(self):
        with tempfile.TemporaryDirectory() as root,patch('threading.Thread.start'):
            m=Manager(root)
            state=m.start(dict(m=32,n=16,matrix_rank=8,target_rank=8,steps=16,sweep_sizes=[4,8],algorithms=['jacobi']))
            self.assertEqual([c['case']['m'] for c in state['configs']],[4,8])
            self.assertEqual(state['configs'][0]['rank'],4)
            with self.assertRaises(RuntimeError):m.start({})
            m.cancel(state['id']);self.assertTrue(m.cancelled.is_set())
            restarted=Manager(root);self.assertEqual(restarted.state(state['id'])['status'],'interrupted')
            with self.assertRaises(ValueError):m.state('../config')

    def test_http_session_and_real_worker(self):
        with tempfile.TemporaryDirectory() as root:
            server=make_server(0,root);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            url=f'http://127.0.0.1:{server.server_port}'
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            def request(path,body=None,token=None,origin=None):
                headers={}
                if token:headers['X-Playground-Token']=token
                if origin:headers['Origin']=origin
                data=None if body is None else json.dumps(body).encode()
                return opener.open(urllib.request.Request(url+path,data=data,headers=headers),timeout=10)
            try:
                info=json.load(request('/api/info'))
                with self.assertRaises(urllib.error.HTTPError) as err:request('/api/jobs',{})
                self.assertEqual(err.exception.code,403)
                with self.assertRaises(urllib.error.HTTPError):request('/api/jobs',{},info['token'],'https://example.com')
                payload=dict(m=12,n=8,target_rank=3,steps=8,algorithms=['jacobi','vendor_default'],repeat=1,warmup=0,timeout=120)
                state=json.load(request('/api/jobs',payload,info['token']))
                deadline=time.monotonic()+150
                while state['status'] in ('queued','running') and time.monotonic()<deadline:
                    time.sleep(.1);state=json.load(request('/api/jobs/'+state['id']))
                self.assertEqual(state['status'],'complete',state)
                self.assertEqual(len(state['rows']),2)
                self.assertTrue(all(r['status']=='ok' for r in state['rows']),state['rows'])
                self.assertIn('visualization',state['rows'][0])
                self.assertEqual(len(state['rows'][0]['source_sha256']),64)
                self.assertEqual(state['rows'][0]['input_sha256'],state['previews']['0']['sha256'])
                self.assertTrue(request('/api/jobs/'+state['id']+'/matrix/0').read().startswith(b'\x93NUMPY'))
                self.assertEqual(json.load(request('/api/jobs/'+state['id']+'/download'))['id'],state['id'])
                # Cancellation is exercised through the API against a real child process.
                state=json.load(request('/api/jobs',payload,info['token']))
                request('/api/jobs/'+state['id']+'/cancel',{},info['token']).close()
                deadline=time.monotonic()+10
                while state['status'] in ('queued','running') and time.monotonic()<deadline:
                    time.sleep(.1);state=json.load(request('/api/jobs/'+state['id']))
                self.assertEqual(state['status'],'cancelled',state)
            finally:
                if server.manager.active:server.manager.cancel(server.manager.active)
                server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
