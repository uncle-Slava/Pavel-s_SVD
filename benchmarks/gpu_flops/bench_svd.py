"""GPU benchmark of every SVD in Pavel-s_SVD (svdlab blocks + recipes) vs torch.linalg.svd.

Square Gaussian matrices n in {32, 64, 128, 256}, full rank, float32 on CUDA (plus float64
for a few methods). Records time (median/min over timed runs, CUDA events), accuracy, the
iteration counts the implementation reports, and matmul-family FLOPs from
torch.utils.flop_counter (GEMM work only; element-wise ops are not counted).

Run from the repository root:
    .venv/bin/python benchmarks/gpu_flops/bench_svd.py --sizes 32 64 128 256 --device cuda
"""
import argparse, json, os, statistics, sys, time, traceback
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from svdbench.standalone import ALGORITHMS, svd as standalone_svd
from svdbench.standalone import primitives as P
from svdlab import recipes

OUT = 'benchmarks/gpu_flops/out'
SIZES = [32, 64, 128, 256]
DEV = 'cuda'
WARMUP, REPS = 3, 20
SLOW_RUN_S = 2.0        # a single run slower than this -> 1 warmup + 3 timed runs
SKIP_AFTER_128_S = 5.0  # median at n=128 above this -> skip n=256

# --- instrumentation: count inner-solver iterations for every implementation ----------
COUNTS = {}
_jacobi_svd, _jacobi_eigh, _bdsqr = P.jacobi_svd, P.jacobi_eigh, P.bdsqr


def _jsvd(*a, **k):
    out = _jacobi_svd(*a, **k)
    COUNTS.setdefault('jacobi_svd_sweeps', []).append(int(out[3]))
    return out


def _jeigh(*a, **k):
    out = _jacobi_eigh(*a, **k)
    COUNTS.setdefault('jacobi_eigh_sweeps', []).append(int(out[2]))
    return out


def _bd(*a, **k):
    out = _bdsqr(*a, **k)
    COUNTS.setdefault('bdsqr_iterations', []).append(int(getattr(_bd, 'last_iters', getattr(_bdsqr, 'last_iters', -1))))
    return out


P.jacobi_svd, P.jacobi_eigh, P.bdsqr = _jsvd, _jeigh, _bd
# divide_conquer imported jacobi helpers through the module object `P`, so it is covered too.


def make_impls():
    impls = {'torch.linalg.svd': lambda a, r: (*torch.linalg.svd(a, full_matrices=False), {})}
    for alg in ALGORITHMS:
        def f(a, r, alg=alg):
            u, s, vh, info = standalone_svd(a, alg, rank=r, return_info=True)
            return u, s, vh, info
        impls[f'blocks.svd:{alg}'] = f
    for name in ('golub_kahan', 'qr_then_svd', 'randomized', 'lanczos'):
        fn = getattr(recipes, name)
        impls[f'recipes.{name}'] = lambda a, r, fn=fn: (*fn(a, r), {})
    return impls


def matrix(n, dtype):
    g = torch.Generator().manual_seed(1000 + n)
    return torch.randn(n, n, generator=g, dtype=torch.float64).to(dtype).to(DEV)


def accuracy(a, u, s, vh, s_ref):
    a64, u, s, vh = a.double(), u.double(), s.double(), vh.double()
    k = s.shape[0]
    rec = torch.linalg.norm(a64 - (u * s) @ vh) / torch.linalg.norm(a64)
    eye = torch.eye(k, dtype=torch.float64, device=a.device)
    ou = torch.linalg.norm(u.mT @ u - eye)
    ov = torch.linalg.norm(vh @ vh.mT - eye)
    sr = s_ref[:k]
    sv = ((s - sr).abs() / sr.clamp_min(1e-300)).max()
    return float(rec), float(ou), float(ov), float(sv)


def timed(fn, a, n, warmup, reps):
    times = []
    for i in range(warmup + reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        out = fn(a, n)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        if i >= warmup:
            times.append(ms)
        if i == 0 and ms > SLOW_RUN_S * 1000 and reps > 3:
            warmup, reps = 1, 3
    return out, times


def gemm_flops(fn, a, n):
    try:
        with FlopCounterMode(display=False) as fc:
            fn(a, n)
        return int(fc.get_total_flops())
    except Exception as e:
        return f'error: {type(e).__name__}: {e}'


def run(name, fn, n, dtype, s_ref):
    rec = dict(impl=name, n=n, dtype=str(dtype).replace('torch.', ''), median_ms=None, min_ms=None,
               recon_err=None, orth_u=None, orth_v=None, sv_err=None, iters=None, stats=None,
               gemm_flops=None, timed_runs=0, status='ok', error=None)
    a = matrix(n, dtype)
    try:
        COUNTS.clear()
        u, s, vh, info = fn(a, n)          # first call: counts + stats for one solve
        counts = {k: (v[0] if len(v) == 1 else dict(calls=len(v), total=sum(v), max=max(v)))
                  for k, v in COUNTS.items()}
        info = {k: v for k, v in info.items() if k not in ('algorithm',)}
        rec['iters'] = counts or None
        rec['stats'] = info or None
        (u, s, vh, _), times = timed(fn, a, n, WARMUP - 1, REPS)
        rec['median_ms'], rec['min_ms'] = statistics.median(times), min(times)
        rec['timed_runs'] = len(times)
        rec['recon_err'], rec['orth_u'], rec['orth_v'], rec['sv_err'] = accuracy(a, u, s, vh, s_ref)
        if s.shape[0] != n:
            rec['status'] = f'ok (returned rank {s.shape[0]})'
        rec['gemm_flops'] = gemm_flops(fn, a, n)
    except Exception as e:
        rec['status'] = 'failed'
        rec['error'] = f'{type(e).__name__}: {e}'.strip()[:500]
        rec['traceback_tail'] = traceback.format_exc().strip().splitlines()[-4:]
    torch.cuda.synchronize()
    return rec


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--sizes', type=int, nargs='+', default=SIZES)
    p.add_argument('--device', default=DEV, help='cuda or cuda:N')
    p.add_argument('--warmup', type=int, default=WARMUP)
    p.add_argument('--repeat', type=int, default=REPS, help='timed runs per case')
    p.add_argument('--out', default=OUT, help='output directory (results.json, env.json)')
    return p.parse_args(argv)


def main():
    global SIZES, DEV, WARMUP, REPS, OUT
    a = parse_args()
    SIZES, DEV, WARMUP, REPS, OUT = a.sizes, a.device, a.warmup, a.repeat, a.out
    os.makedirs(OUT, exist_ok=True)
    impls = make_impls()
    f64 = ['torch.linalg.svd', 'blocks.svd:jacobi', 'blocks.svd:bidiag_qr', 'blocks.svd:polar_qdwh']
    plan = [(name, torch.float32) for name in impls] + [(name, torch.float64) for name in f64]
    results, skip = [], set()
    t0 = time.time()
    for n in SIZES:
        s_ref = {}
        for name, dtype in plan:
            key = (name, dtype)
            if key in skip:
                results.append(dict(impl=name, n=n, dtype=str(dtype).replace('torch.', ''), status='skipped',
                                    error=f'n=128 median exceeded {SKIP_AFTER_128_S} s'))
                continue
            if dtype not in s_ref:
                s_ref[dtype] = torch.linalg.svdvals(matrix(n, dtype).double())
            rec = run(name, impls[name], n, dtype, s_ref[dtype])
            results.append(rec)
            print(f'[{time.time()-t0:7.1f}s] n={n:3d} {rec["dtype"]:7s} {name:30s} {rec["status"]:8s} '
                  f'med={rec["median_ms"]} rec={rec["recon_err"]} iters={rec["iters"]} {rec["error"] or ""}',
                  flush=True)
            if n == 128 and rec.get('median_ms') and rec['median_ms'] > SKIP_AFTER_128_S * 1000:
                skip.add(key)
            with open(f'{OUT}/results.json', 'w') as fh:
                json.dump(results, fh, indent=1)
    env = dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               gpu=torch.cuda.get_device_name(DEV), capability=torch.cuda.get_device_capability(DEV),
               python=sys.version.split()[0], seed='torch.Generator().manual_seed(1000+n), randn float64 -> cast',
               warmup=WARMUP, reps=REPS, sizes=SIZES, total_s=round(time.time() - t0, 1))
    with open(f'{OUT}/env.json', 'w') as fh:
        json.dump(env, fh, indent=1)
    print(json.dumps(env))


if __name__ == '__main__':
    main()
