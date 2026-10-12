"""Benchmark updates, warm starts, sparse operators, and tensor decompositions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from svdbench.cli import inventory, measure, to_host, synchronize
from svdbench.metrics import reference, validate
from svdbench.standalone import svd
from svdbench.standalone import primitives as P
from svdbench.standalone.iterative import rank_one_update, append_columns, lanczos_svd
from svdbench.standalone.tensors import tucker, tucker_reconstruct, tensor_train, tensor_train_reconstruct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    p.add_argument('--repeat', type=int, default=3)
    args = p.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    gen = torch.Generator(device=device).manual_seed(20261011)
    def rand(*shape):
        return torch.randn(*shape, dtype=torch.float64, device=device, generator=gen)
    def rel(a, b):
        return float((a - b).abs().square().sum().sqrt() / a.abs().square().sum().sqrt())
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out, torch.inference_mode():
        out.write(json.dumps({'type':'metadata', 'environment':inventory(device), 'arguments':vars(args),
                              'contract':'Resident operands and initial factors; preparation excluded; no graph capture. FP64.'}) + '\n')
        def record(name, fn, check, input_tensor):
            row = {'type':'application', 'name':name,
                   'input_sha256':hashlib.sha256(input_tensor.cpu().numpy().tobytes()).hexdigest(),
                   'input_shape':list(input_tensor.shape), 'dtype':str(input_tensor.dtype)}
            try:
                fn(); synchronize(device)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                    base = torch.cuda.memory_allocated(device)
                row['timing'] = measure(fn, device, args.repeat, events=True)
                if device.type == 'cuda':
                    row['allocator_peak_increment_bytes'] = torch.cuda.max_memory_allocated(device) - base
                row['validation'] = check(fn())
                row['status'] = 'ok'
            except Exception as e:
                row.update(status='error',error=f'{type(e).__name__}: {e}')
            out.write(json.dumps(row, allow_nan=False)+'\n'); out.flush()
            print(name, row['status'], row.get('timing',{}).get('wall',{}).get('median_ms'), row.get('error',''), flush=True)
        def svd_check(a):
            cpu = a.cpu().numpy()
            refs = reference(cpu)
            return lambda result: validate(cpu, to_host(result[:3]), refs)['worst']
        a = rand(128,8) @ rand(8,96)
        u, s, vh = svd(a, 'jacobi', rank=8)
        left, right = rand(128), rand(96)
        updated = a + left[:,None] * right.conj()
        record('rank_one_update',lambda:rank_one_update(u,s,vh,left,right),svd_check(updated),updated)
        columns = rand(128,4)
        appended = torch.cat([a,columns],1)
        record('append_columns',lambda:append_columns(u,s,vh,columns),svd_check(appended),appended)
        _, _, initial = svd(a,'jacobi',rank=12)
        record('warm_subspace',lambda:svd(updated,'subspace',rank=8,oversample=4,iterations=2,initial_basis=initial.mH),svd_check(updated),updated)
        record('cold_subspace',lambda:svd(updated,'subspace',rank=8,oversample=4,iterations=2),svd_check(updated),updated)
        record('vendor_update_baseline',lambda:torch.linalg.svd(updated,full_matrices=False),svd_check(updated),updated)
        # Fixed sequence, restarting from the same initial factors for every timing sample.
        pairs=[(0.02*rand(128),rand(96)) for _ in range(12)]
        target=a.clone()
        for x,y in pairs: target=target+x[:,None]*y.conj()
        def sequence():
            uu,ss,vv=u,s,vh
            for x,y in pairs:
                uu,ss,vv=rank_one_update(uu,ss,vv,x,y,rank=min(12,len(ss)+1))
            return uu,ss,vv
        record('twelve_updates_truncated',sequence,svd_check(target),target)
        # Sparse callback path: a rectangular diagonal operator, exact top eight.
        idx=torch.arange(24,device=device)
        sp=torch.sparse_coo_tensor(torch.stack([idx,idx]),torch.arange(24,0,-1,dtype=torch.float64,device=device),(96,64)).coalesce()
        class Sparse:
            shape,dtype=(96,64),torch.float64
            def __init__(self): self.device=device
            def matmat(self,x): return torch.sparse.mm(sp,x)
            def rmatmat(self,x): return torch.sparse.mm(sp.transpose(0,1),x)
        def sparse():
            uu,ss,vv,info=lanczos_svd(Sparse(),rank=8,steps=32)
            return uu,ss,vv.mH
        record('sparse_lanczos',sparse,svd_check(sp.to_dense()),sp.to_dense())
        core=rand(4,3,2)
        factors=[P.qr(rand(n,r)[None])[0][0] for n,r in zip((16,12,10),(4,3,2))]
        x=tucker_reconstruct(core,factors)
        def check_tucker(res):
            c,fac,info=res
            return {'reconstruction_relative_fro':rel(x,tucker_reconstruct(c,fac)),
                    'stored_scalars':c.numel()+sum(q.numel() for q in fac),'input_scalars':x.numel(),'unfoldings':info}
        for seq in (False,True):
            record('st_hosvd' if seq else 'hosvd',lambda seq=seq:tucker(x,[4,3,2],sequential=seq),check_tucker,x)
        tt=[rand(1,12,3),rand(3,10,3),rand(3,8,1)]
        xx=tensor_train_reconstruct(tt)
        def check_tt(res):
            cs,info=res
            return {'reconstruction_relative_fro':rel(xx,tensor_train_reconstruct(cs)),
                    'stored_scalars':sum(q.numel() for q in cs),'input_scalars':xx.numel(),'unfoldings':info}
        record('tt_svd',lambda:tensor_train(xx,[3,3]),check_tt,xx)
    print(path)

if __name__=='__main__': main()
