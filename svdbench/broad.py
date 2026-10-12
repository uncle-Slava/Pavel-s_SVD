"""Isolated larger-matrix benchmark, with explicit quality and memory contracts.

Run: python -m svdbench.broad --config configs/broad.json --output-dir DIR
Each preparation and solver executes in a fresh child. Original inputs and complete
FP64/complex128 reference factors are cached on disk, outside all solve timings.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .algorithms import prepare
from .cli import inventory, run_one
from .matrices import make_matrix
from .metrics import reference
from .playground.engine import factory as playground_factory


def factory(a, spec):
    """Additional controls; standalone adapters keep their existing implementations."""
    method=spec['method'];rank=spec['rank'];partial=rank<min(a.shape[-2:])
    if method.startswith('native_'):
        routine=method.removeprefix('native_')
        fn=prepare(a,dict(name='svdbench.legacy_cuda:prepare',routine=routine,
                          rank=rank,oversample=spec['oversample'],iterations=spec['iterations']),spec['seed'])
    elif method in ('randomized_library','gram_library','qr_library'):
        base={'randomized_library':'randomized','gram_library':'gram','qr_library':'qr_svd'}[method]
        fn=prepare(a,dict(name=base,rank=rank,oversample=spec['oversample'],iterations=spec['iterations']),spec['seed'])
        fn.metadata={'implementation':'Torch library composition',
                     'method':base,'full_solve_then_slice':partial and base=='qr_svd'}
    elif method=='sparse_randomized_library':
        sparse=a.to_sparse().coalesce()
        adjoint=sparse.mH.coalesce()
        k=min(rank+spec['oversample'],min(a.shape))
        gen=torch.Generator(device=a.device).manual_seed(spec['seed'])
        omega=torch.randn(a.shape[1],k,dtype=a.dtype,device=a.device,generator=gen)
        def fn():
            q=torch.linalg.qr(torch.sparse.mm(sparse,omega),mode='reduced')[0]
            for _ in range(spec['iterations']):
                z=torch.linalg.qr(torch.sparse.mm(adjoint,q),mode='reduced')[0]
                q=torch.linalg.qr(torch.sparse.mm(sparse,z),mode='reduced')[0]
            u,s,vh=torch.linalg.svd(torch.sparse.mm(adjoint,q).mH,full_matrices=False)
            return q@u[:,:rank],s[:rank],vh[:rank]
        fn.metadata={'implementation':'Sparse COO products with library QR and projected SVD',
                     'sparse_bytes':sum(x.values().numel()*x.values().element_size()+x.indices().numel()*8 for x in (sparse,adjoint)),
                     'dense_input_retained':True,'sketch_prepared':True}
    else:
        return playground_factory(a,dict(algorithm=method,output='topk' if partial else 'reduced',
            rank=rank,family=spec['family'],oversample=spec['oversample'],iterations=spec['iterations'],
            max_sweeps=spec['max_sweeps'],steps=spec['steps'],seed=spec['seed']))
    def solve():
        u,s,vh=fn()
        if hasattr(fn,'diagnostics'):solve.diagnostics=fn.diagnostics
        return u[...,:rank],s[...,:rank],vh[...,:rank,:]
    solve.metadata=getattr(fn,'metadata',{})
    return solve


def grade(row, cfg):
    if row.get('status')!='ok':return {'passed':False,'reasons':[row.get('error',row['status'])]}
    w=row['metrics']['worst'];double=cfg['case']['dtype'] in ('float64','complex128')
    tol=1e-9 if double else 2e-4
    partial=cfg['target_rank']<min(cfg['case']['m'],cfg['case']['n'])
    checks=dict(orthogonality_u_fro_over_sqrt_cols=tol,orthogonality_v_fro_over_sqrt_rows=tol)
    if partial:
        if w.get('residual_over_optimal_tail') is None:checks['reconstruction_relative_fro']=tol
        else:checks['residual_over_optimal_tail']=1.05
        checks['resolved_spectrum_max_relative_error']=.01
    else:
        checks['reconstruction_relative_fro']=tol
        checks['spectrum_error_over_sigma1']=tol
    reasons=[f'{k}={w.get(k)} > {v}' for k,v in checks.items() if w.get(k) is None or w[k]>v]
    if not w.get('singular_values_nonnegative_sorted'):reasons.append('Values are not sorted/nonnegative')
    return {'passed':not reasons,'reasons':reasons,'limits':checks,'partial':partial}


def write_json(path, data):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,allow_nan=False));tmp.replace(path)


def worker(stage, path, method_index):
    path=Path(path);cfg=json.loads((path/'config.json').read_text());case=cfg['case']
    torch.set_num_threads(cfg['threads']);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with threadpool_limits(cfg['threads']),torch.inference_mode():
        if stage=='prepare':
            t=time.perf_counter();a=make_matrix(case);np.save(path/'matrix.npy',a,allow_pickle=False)
            refs=reference(a)
            np.savez(path/'reference.npz',u=np.stack([r[0] for r in refs]),s=np.stack([r[1] for r in refs]),vh=np.stack([r[2] for r in refs]))
            write_json(path/'input.json',dict(input_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                       preparation_s=time.perf_counter()-t,input_bytes=a.nbytes,density=float(np.count_nonzero(a)/a.size)))
            return
        method=cfg['methods'][method_index]
        a=np.load(path/'matrix.npy',allow_pickle=False)
        with np.load(path/'reference.npz',allow_pickle=False) as ref:refs=list(zip(ref['u'],ref['s'],ref['vh']))
        spec=dict(name='svdbench.broad:factory',method=method['method'],rank=cfg['target_rank'],
                  family=case['family'],seed=method.get('seed',case['seed']),
                  oversample=method.get('oversample',16),iterations=method.get('iterations',2),
                  max_sweeps=method.get('max_sweeps',100),steps=method.get('steps',min(min(a.shape[-2:]),4*cfg['target_rank']+16)))
        opts=SimpleNamespace(repeat=cfg['repeat'],warmup=cfg['warmup'],end_to_end=False)
        row=run_one(a,case,spec,torch.device('cuda:0'),opts,refs)
        row.update(id=method['id'],method_index=method_index,assessment=grade(row,cfg),
                   input_sha256=json.loads((path/'input.json').read_text())['input_sha256'])
        write_json(path/f'result-{method_index}.json',row)


def child(stage,path,index,timeout):
    cmd=[sys.executable,'-m','svdbench.broad','--worker',stage,'--case-dir',str(path),'--method-index',str(index)]
    started=time.monotonic()
    with (path/f'{stage}-{index}.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        try:code=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill();proc.wait();return 'timeout',time.monotonic()-started
    return ('ok' if code==0 else 'error'),time.monotonic()-started


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/broad.json');p.add_argument('--output-dir')
    p.add_argument('--worker',choices=['prepare','solve']);p.add_argument('--case-dir');p.add_argument('--method-index',type=int,default=0)
    args=p.parse_args()
    if args.worker:worker(args.worker,args.case_dir,args.method_index);return
    if not args.output_dir:p.error('--output-dir is required')
    config=json.loads(Path(args.config).read_text());root=Path(args.output_dir).resolve();root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(config['threads']);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    write_json(root/'config.json',config);write_json(root/'environment.json',inventory(torch.device('cuda:0')))
    telemetry=(root/'telemetry.csv').open('w')
    sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem','--format=csv','--loop-ms=500'],stdout=telemetry,stderr=subprocess.STDOUT)
    try:
        with (root/'measurements.jsonl').open('x') as results:
            for case in config['cases']:
                path=root/case['name'];path.mkdir()
                cfg=dict(case={k:v for k,v in case.items() if k not in ('methods','target_rank')},
                         target_rank=case.get('target_rank',min(case['m'],case['n'])),methods=case['methods'],
                         **{k:config[k] for k in ('threads','repeat','warmup')})
                write_json(path/'config.json',cfg)
                print('PREPARE',case['name'],flush=True)
                status,elapsed=child('prepare',path,0,config['reference_timeout_s'])
                if status!='ok':
                    write_json(root/'preparation-error.json',dict(case=case['name'],status=status,elapsed_s=elapsed))
                    raise RuntimeError(f'Reference preparation {status}: {case["name"]}')
                for i,method in enumerate(case['methods']):
                    write_json(root/'progress.json',dict(case=case['name'],method=method['id'],started=datetime.now(timezone.utc).isoformat()))
                    status,elapsed=child('solve',path,i,config['worker_timeout_s'])
                    result=path/f'result-{i}.json'
                    if status=='ok' and result.exists():row=json.loads(result.read_text())
                    else:
                        row=dict(type='measurement',case=cfg['case'],id=method['id'],algorithm={**method,'rank':cfg['target_rank']},status=status,
                                 input_sha256=json.loads((path/'input.json').read_text())['input_sha256'],
                                 error=f'Worker {status}; limit {config["worker_timeout_s"]} seconds covers startup, warmup, timed repetitions and validation.',
                                 worker_log=(path/f'solve-{i}.log').read_text(errors='replace')[-4000:])
                        row['assessment']=grade(row,cfg)
                    row['worker_elapsed_s']=elapsed
                    results.write(json.dumps(row,allow_nan=False)+'\n');results.flush()
                    print(case['name'],method['id'],row['status'],row.get('assessment',{}).get('passed'),row.get('resident_solver',{}).get('wall',{}).get('median_ms'),flush=True)
        write_json(root/'complete.json',dict(completed=datetime.now(timezone.utc).isoformat()))
    finally:
        sampler.terminate();sampler.wait(timeout=10);telemetry.close()


if __name__=='__main__':main()
