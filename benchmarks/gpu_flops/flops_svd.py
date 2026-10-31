"""Executed-FLOP count for every SVD in Pavel-s_SVD, square n x n, float32 on CUDA.

total = matmul FLOPs (torch.utils.flop_counter, measured)
      + element-wise FLOPs of the repository primitives, counted exactly from the shapes
        and the actual iteration counts of each call (Householder QR / bidiagonalization /
        tridiagonalization, Cholesky, triangular solves, one-sided and Hermitian Jacobi,
        bidiagonal QR). These primitives use no matmul, so nothing is counted twice.
Counts are executed operations (masked work included), real arithmetic, 1 mul or add = 1 FLOP.
Not counted: Python-side scalar work, divide-and-conquer secular solves (float64), norms,
sorting, sketch generation. torch.linalg.svd is a vendor kernel: shown as the LAPACK
textbook estimate 21 n^3 elsewhere, here only its (zero) matmul FLOPs.

Run from the repository root after bench_svd.py (it skips the cases bench_svd.py skipped):
    .venv/bin/python benchmarks/gpu_flops/flops_svd.py --sizes 32 64 128 256 --device cuda
"""
import argparse, json, os, sys, time
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svdbench.standalone import primitives as P
import bench_svd as B   # reuses make_impls / matrix and its iteration counters

PRIM = {}


def add(kind, flops):
    PRIM[kind] = PRIM.get(kind, 0) + int(flops)


def qr_flops(m, n):
    fac = sum(4 * (m - j) * (n - j - 1) for j in range(n))
    acc = sum(4 * (m - j) * n for j in range(n))
    return fac + acc


def bidiag_flops(m, n):
    left = sum(4 * (m - j) * (n - j - 1) for j in range(n))
    right = sum(4 * (m - j - 1) * (n - j - 1) + 4 * n * (n - j - 1) for j in range(n - 1))
    acc = sum(4 * (m - j) * n for j in range(n))
    return left + right + acc


def tridiag_flops(n):
    return sum(8 * (n - j - 1) ** 2 + 4 * n * (n - j - 1) for j in range(n - 1))


def wrap(name, model):
    orig = getattr(P, name)

    def f(*a, **k):
        out = orig(*a, **k)
        add(name, model(a, k, out))
        return out
    f.__name__ = name
    if hasattr(orig, 'last_iters'):
        f.last_iters = orig.last_iters
    setattr(P, name, f)
    return orig


def jsvd_model(a, k, out):
    _, m, n = a[0].shape
    n += n % 2
    return int(out[3]) * (n - 1) * (n // 2) * (12 * m + 6 * n)


def jeigh_model(a, k, out):
    n = a[0].shape[-1]
    n += n % 2
    return int(out[2]) * (n - 1) * ((n // 2) * 18 * n + n * n)


# bench_svd already wrapped jacobi_svd / jacobi_eigh / bdsqr for iteration counts; wrap again on top.
wrap('qr', lambda a, k, o: qr_flops(*a[0].shape[-2:]))
wrap('bidiag', lambda a, k, o: bidiag_flops(*a[0].shape[-2:]))
wrap('tridiag', lambda a, k, o: tridiag_flops(a[0].shape[-1]))
wrap('cholesky', lambda a, k, o: sum(2 * j + 2 * (a[0].shape[-1] - j - 1) * j for j in range(a[0].shape[-1])))
wrap('trsm_lower', lambda a, k, o: a[0].shape[-1] ** 2 * a[1].shape[-1])
wrap('trsm_upper', lambda a, k, o: a[0].shape[-1] ** 2 * a[1].shape[-1])
wrap('jacobi_svd', jsvd_model)
wrap('jacobi_eigh', jeigh_model)
_bd_prev = P.bdsqr


def _bd(*a, **k):
    out = _bd_prev(*a, **k)
    it = int(getattr(_bd, 'last_iters', 0))  # primitives.bdsqr stores it on the name P.bdsqr, i.e. here
    n = a[0].shape[-1]
    add('bdsqr', it * (n - 1) * 2 * 6 * n)
    _bd.last_iters = it
    return out


P.bdsqr = _bd


def main():
    p = argparse.ArgumentParser(description='Executed-FLOP count per SVD implementation')
    p.add_argument('--sizes', type=int, nargs='+', default=B.SIZES)
    p.add_argument('--device', default=B.DEV)
    p.add_argument('--out', default=B.OUT, help='directory holding results.json; flops.json is written there')
    a = p.parse_args()
    B.SIZES, B.DEV, OUT = a.sizes, a.device, a.out
    impls = B.make_impls()
    rows = []
    t0 = time.time()
    path = f'{OUT}/results.json'
    done = json.load(open(path)) if os.path.exists(path) else []
    skip = {(r['impl'], r['n']) for r in done if r['dtype'] == 'float32' and r['status'] != 'ok'}
    for n in B.SIZES:
        for name, fn in impls.items():
            if (name, n) in skip:
                continue
            a = B.matrix(n, torch.float32)
            PRIM.clear()
            rec = dict(impl=name, n=n)
            try:
                with FlopCounterMode(display=False) as fc:
                    fn(a, n)
                torch.cuda.synchronize()
                g = int(fc.get_total_flops())
                rec.update(matmul_flops=g, primitive_flops=dict(PRIM), elementwise_flops=sum(PRIM.values()),
                           total_flops=g + sum(PRIM.values()), status='ok')
            except Exception as e:
                rec.update(status='failed', error=f'{type(e).__name__}: {e}'[:300])
            rows.append(rec)
            print(f'[{time.time()-t0:6.1f}s] n={n} {name:30s} {rec.get("total_flops")} {rec.get("primitive_flops")}', flush=True)
            with open(f'{OUT}/flops.json', 'w') as fh:
                json.dump(rows, fh, indent=1)


if __name__ == '__main__':
    main()
