"""Editable parameter study: fixed input bytes, several algorithm seeds, all outcomes."""
import argparse
import itertools
from pathlib import Path
import torch
from threadpoolctl import threadpool_limits
from svdbench.laboratory import MatrixCase, NullTrace, accuracy, benchmark, describe
from svdbench.laboratory.trace import write_json
from .algorithms import randomized


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--input');p.add_argument('--m',type=int,default=96);p.add_argument('--n',type=int,default=64)
    p.add_argument('--rank',type=int,default=8);p.add_argument('--iterations',default='0,1,2,4')
    p.add_argument('--oversamples',default='4,8');p.add_argument('--seeds',default='3,7,11')
    p.add_argument('--repeat',type=int,default=5);p.add_argument('--primitives',choices=['standalone','library'],default='standalone')
    args=p.parse_args()
    iterations=[int(v) for v in args.iterations.split(',')]
    oversamples=[int(v) for v in args.oversamples.split(',')]
    seeds=[int(v) for v in args.seeds.split(',')]
    if min(iterations+oversamples)<0 or args.repeat<1:p.error('Invalid study parameters')
    if len(seeds)!=len(set(seeds)):p.error('Repeated seeds are not independent trials')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(4),torch.inference_mode():
        # EDIT this matrix family, or pass --input to reuse any saved matrix.
        case=MatrixCase.load(args.input) if args.input else MatrixCase.create(m=args.m,n=args.n,family='gaussian',seed=123)
        case.save(root)
        a=case.tensor(args.device)
        write_json(root/'study.json',dict(parameters=vars(args),input_sha256=case.sha256,
                                        note='Same input across trials; seeds vary sketches, not matrices. Timing repetitions are separate from seed trials.'))
        rows=[]
        for q,pad,seed in itertools.product(iterations,oversamples,seeds):
            row=dict(iterations=q,oversample=pad,seed=seed)
            try:
                def solve():return randomized(a,NullTrace(),rank=args.rank,iterations=q,oversample=pad,seed=seed,primitives=args.primitives)
                answer=solve();row['accuracy']=accuracy(case,answer);del answer
                row['benchmark']=benchmark(solve,args.device,args.repeat,1)
                row['status']='ok'
            except Exception as exc:
                row.update(status='error',error=f'{type(exc).__name__}: {exc}')
            rows.append(row)
            write_json(root/'trials.json',rows)
        groups=[]
        for q,pad in itertools.product(iterations,oversamples):
            subset=[r for r in rows if r['iterations']==q and r['oversample']==pad]
            good=[r for r in subset if r['status']=='ok']
            item=dict(iterations=q,oversample=pad,attempts=len(subset),completed=len(good))
            if good:
                item['per_seed_median_wall_ms']=describe([r['benchmark']['wall_ms']['median'] for r in good])
                item['leading_spectrum_relative_error']=describe([r['accuracy']['worst']['resolved_spectrum_max_relative_error'] for r in good if r['accuracy']['worst']['resolved_spectrum_max_relative_error'] is not None]) if any(r['accuracy']['worst']['resolved_spectrum_max_relative_error'] is not None for r in good) else None
            groups.append(item)
        write_json(root/'summary.json',groups)
    print(f'Saved {len(rows)} trials in {root.resolve()} (each retains all timing samples and metrics).')


if __name__=='__main__':main()
