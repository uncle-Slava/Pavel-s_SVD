"""A bench for individual operations. Edit OPERATIONS without writing a full SVD."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from svdbench.laboratory import MatrixCase, Trace, benchmark, orthogonality, Run
from svdbench.laboratory.trace import write_json
from svdbench.standalone import primitives as P


OPERATIONS = {
    'standalone_householder_qr': lambda a: P.qr(a[None]),
    'library_householder_qr': lambda a: torch.linalg.qr(a[None],mode='reduced'),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--m',type=int,default=128);p.add_argument('--n',type=int,default=16)
    p.add_argument('--repeat',type=int,default=10)
    args=p.parse_args()
    if args.m<args.n:p.error('This QR example expects m >= n')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(4),torch.inference_mode():
        case=MatrixCase.create(m=args.m,n=args.n,family='geometric',condition=1e6)
        a=case.tensor(args.device);answers={}
        with Trace(args.output,case,args.device,parameters=vars(args),source_files=[__file__]) as trace:
            for name,operation in OPERATIONS.items():
                with trace.stage(name):q,r=operation(a)
                q,r=q[0],r[0]
                scalars=orthogonality(q)
                scalars['qr_reconstruction_relative_fro']=float(torch.linalg.vector_norm(a-q@r)/torch.linalg.vector_norm(a))
                trace.checkpoint(name,tensors={'Q':q,'R':r,'QR_minus_A':q@r-a},scalars=scalars)
                del q,r
        for name,operation in OPERATIONS.items():
            answers[name]=benchmark(lambda op=operation:op(a),args.device,args.repeat,2)
        write_json(Path(args.output)/'benchmarks.json',answers)
        Run(args.output).plot()
    print(f'Saved independent operation comparisons in {Path(args.output).resolve()}')


if __name__=='__main__':main()
