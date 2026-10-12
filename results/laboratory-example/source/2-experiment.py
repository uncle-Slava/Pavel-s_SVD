"""Run an editable algorithm, inspect its steps, then time it without inspection."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import torch
from threadpoolctl import threadpool_limits
from svdbench.laboratory import MatrixCase, Trace, NullTrace, Run, accuracy, benchmark
from svdbench.laboratory.trace import write_json
from . import algorithms, my_algorithm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--algorithm', choices=['randomized','jacobi','mine'], default='randomized')
    p.add_argument('--device',default='cpu');p.add_argument('--output');p.add_argument('--input')
    p.add_argument('--family',default='low_rank',choices=['gaussian','low_rank','clustered','geometric','sparse','zero'])
    p.add_argument('--m',type=int,default=96);p.add_argument('--n',type=int,default=64)
    p.add_argument('--rank',type=int,default=8);p.add_argument('--matrix-rank',type=int,default=8)
    p.add_argument('--dtype',default='float64',choices=['float32','float64','complex64','complex128'])
    p.add_argument('--seed',type=int,default=7);p.add_argument('--matrix-seed',type=int,default=23)
    p.add_argument('--iterations',type=int,default=2);p.add_argument('--oversample',type=int,default=8)
    p.add_argument('--condition',type=float,default=1e6);p.add_argument('--density',type=float,default=.01)
    p.add_argument('--tail',type=float,default=1e-5);p.add_argument('--cluster-size',type=int,default=4)
    p.add_argument('--cluster-spread',type=float,default=1e-6)
    p.add_argument('--max-sweeps',type=int,default=40);p.add_argument('--record-rotations',action='store_true')
    p.add_argument('--allow-unconverged',action='store_true',help='Return Jacobi after the sweep budget for convergence studies')
    p.add_argument('--primitives',choices=['standalone','library'],default='standalone')
    p.add_argument('--sparse-products',action='store_true')
    p.add_argument('--snapshots',choices=['full','stats'],default='full')
    p.add_argument('--snapshot-mib',type=float,default=64);p.add_argument('--trace-every',type=int,default=1)
    p.add_argument('--repeat',type=int,default=10);p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--power-seconds',type=float,default=0,help='Separate sustained NVIDIA load pass with coarse telemetry')
    args=p.parse_args()
    if args.threads<1 or args.repeat<1 or args.warmup<0 or args.power_seconds<0:
        p.error('Invalid thread/repetition/power settings')
    if args.sparse_products and args.algorithm=='jacobi':p.error('Jacobi requires dense storage')
    device=torch.device(args.device)
    if device.type not in ('cpu','cuda'):p.error('Use cpu or cuda[:index]; ROCm uses cuda spelling')
    if args.power_seconds and (device.type!='cuda' or torch.version.hip):p.error('Power sampling requires NVIDIA CUDA')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    output=Path(args.output or ('laboratory_runs/'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'-'+args.algorithm))
    with threadpool_limits(args.threads),torch.inference_mode():
        case=MatrixCase.load(args.input) if args.input else MatrixCase.create(
            m=args.m,n=args.n,family=args.family,dtype=args.dtype,seed=args.matrix_seed,
            rank=args.matrix_rank,tail=args.tail,condition=args.condition,density=args.density,
            cluster_size=args.cluster_size,cluster_spread=args.cluster_spread)
        if case.array.ndim!=2:p.error('Editable starter solvers use one matrix; loop over a batch explicitly')
        if not 1<=args.rank<=min(case.array.shape):p.error('Requested rank is outside the input dimensions')
        a=case.tensor(device,sparse=args.sparse_products)
        fn={'randomized':algorithms.randomized,'jacobi':algorithms.jacobi,'mine':my_algorithm.solve}[args.algorithm]
        parameters=dict(rank=args.rank,oversample=args.oversample,iterations=args.iterations,seed=args.seed,primitives=args.primitives)
        if args.algorithm=='jacobi':parameters=dict(max_sweeps=args.max_sweeps,record_rotations=args.record_rotations,require_convergence=not args.allow_unconverged)
        def solve(trace):return fn(a,trace=trace,**parameters)
        source=list(Path(__file__).parent.glob('*.py'))+list((Path(__file__).parents[1]/'svdbench'/'laboratory').glob('*.py'))
        source+=list((Path(__file__).parents[1]/'svdbench'/'standalone').glob('*.py'))
        source += [Path(__file__).parents[1]/'svdbench'/name for name in ('cli.py','metrics.py','matrices.py')]
        with Trace(output,case,device,snapshots=args.snapshots,snapshot_budget_bytes=int(args.snapshot_mib*2**20),
                   every=args.trace_every,parameters=vars(args),source_files=source) as trace:
            answer=solve(trace)
            validation=accuracy(case,answer)
            del answer
        # Trace only stores CPU arrays; no previous GPU result is retained here.
        timing=benchmark(lambda:solve(NullTrace()),device,args.repeat,args.warmup)
        report=dict(accuracy=validation,benchmark=timing,parameters=parameters,input_sha256=case.sha256)
        if args.power_seconds:
            from .power import measure_power
            report['power']=measure_power(lambda:solve(NullTrace()),device,args.power_seconds,output/'power.jsonl')
        write_json(output/'result.json',report)
        Run(output).plot()
    print(f'Saved laboratory run: {output.resolve()}')
    print(f'Uninstrumented median: {timing["wall_ms"]["median"]:.6g} ms')
    print(f'Reconstruction: {validation["worst"]["reconstruction_relative_fro"]:.6g}')
    print('Open events.jsonl for every stage/checkpoint, snapshots/ for arrays, and trace.png for plots.')


if __name__=='__main__':main()
