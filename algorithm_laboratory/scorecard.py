"""Plug in any SVD function and get accuracy and timing on a fixed set of matrices.

Your function receives one matrix and the requested rank and returns (U, s, Vh):

    def my_svd(a, rank):           # a: torch tensor (m, n); rank: int <= min(m, n)
        ...
        return u, s, vh            # (m, k), (k,), (k, n); torch or NumPy; k >= rank

Extra columns are truncated to `rank`, so a full SVD is a valid answer for every case.
Returning only `s` (a 1-D array) evaluates the spectrum alone.

    from algorithm_laboratory.scorecard import evaluate, compare
    rows = evaluate(my_svd)                          # list of dicts, one per matrix
    rows = compare({'mine': my_svd, 'torch': 'torch', 'jacobi': 'jacobi'})

Command line (methods are builtin names or module:function):

    python -m algorithm_laboratory.scorecard --method torch --method mine \
        --method my_package.module:my_svd --device cpu --output scorecard.csv
"""
import argparse
import csv
import importlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from svdbench.laboratory import MatrixCase, benchmark
from svdbench.laboratory.diagnostics import host
from svdbench.metrics import validate


# Small enough for a CPU FP64 reference in seconds; `rank` is the requested output rank.
DEFAULT_SUITE = (
    dict(name='gaussian_tall', m=200, n=120, family='gaussian', dtype='float64'),
    dict(name='gaussian_wide', m=100, n=180, family='gaussian', dtype='float64'),
    dict(name='ill_conditioned_1e8', m=160, n=120, family='geometric', condition=1e8, dtype='float64'),
    dict(name='clustered', m=160, n=120, family='clustered', cluster_size=4, cluster_spread=1e-6,
         condition=1e3, dtype='float64'),
    dict(name='sparse_dense_storage', m=200, n=150, family='sparse', density=.05, dtype='float64'),
    dict(name='complex_gaussian', m=120, n=80, family='gaussian', dtype='complex128'),
    dict(name='low_rank_k10', m=300, n=200, family='low_rank', rank=10, tail=1e-6, dtype='float64',
         output_rank=10),
    dict(name='low_rank_k16_fp32', m=400, n=300, family='low_rank', rank=16, tail=1e-4, dtype='float32',
         output_rank=16),
)

# `ok` uses these: dtype-scaled limits for exact problems, near-optimal tail for truncated ones.
TOLERANCE_FACTOR = 100
TAIL_RATIO_LIMIT = 1.05

COLUMNS = ('method', 'case', 'm', 'n', 'dtype', 'rank', 'status', 'ok', 'time_median_ms', 'time_mad_ms',
           'reconstruction_relative_fro', 'residual_over_optimal_tail', 'resolved_spectrum_max_relative_error',
           'spectrum_error_over_sigma1', 'orthogonality_u_fro_over_sqrt_cols',
           'orthogonality_v_fro_over_sqrt_rows', 'left_triplet_relative_fro', 'right_triplet_relative_fro',
           'right_subspace_leakage_fro_over_sqrt_k', 'singular_values_nonnegative_sorted', 'error')


def make_case(spec):
    """MatrixCase from a suite entry; a MatrixCase is passed through unchanged."""
    if isinstance(spec, MatrixCase):
        return spec.spec.get('name') or spec.spec.get('family') or spec.sha256[:8], spec, None
    spec = dict(spec)
    name, output_rank = spec.pop('name', None), spec.pop('output_rank', None)
    case = MatrixCase.create(**spec)
    return name or f"{spec.get('family')}_{spec['m']}x{spec['n']}", case, output_rank


def builtin(name):
    """Reference methods: torch, numpy, any svdbench.standalone algorithm, lab.<name> or mine."""
    if name == 'torch':
        return lambda a, rank: torch.linalg.svd(a, full_matrices=False)
    if name == 'numpy':
        return lambda a, rank: np.linalg.svd(host(a), full_matrices=False)
    if name in ('mine', 'lab.mine'):
        from .my_algorithm import solve
        return lambda a, rank: solve(a, rank=rank)
    if name in ('lab.randomized', 'lab.jacobi'):
        from . import algorithms
        fn = getattr(algorithms, name[4:])
        return lambda a, rank: fn(a, rank=rank)
    from svdbench.standalone.solvers import ALGORITHMS, svd
    if name in ALGORITHMS:
        return lambda a, rank: tuple(x[0] for x in svd(a[None], algorithm=name, rank=rank))
    raise ValueError(f'Unknown method {name!r}; use torch, numpy, mine, lab.randomized, lab.jacobi, '
                     f'one of {ALGORITHMS} or module:function')


def resolve(method):
    if callable(method):
        return method
    if ':' in method:
        module, attribute = method.split(':', 1)
        return getattr(importlib.import_module(module), attribute)
    return builtin(method)


def _factors(result, rank):
    if isinstance(result, (torch.Tensor, np.ndarray)) or (isinstance(result, tuple) and len(result) == 1):
        s = host(result[0] if isinstance(result, tuple) else result)
        if s.ndim != 1 or len(s) < rank:
            raise ValueError(f'Expected at least {rank} singular values in a 1-D array, got shape {s.shape}')
        return None, s[:rank], None
    if hasattr(result, 'U') and hasattr(result, 'S'):
        result = (result.U, result.S, result.Vh)
    u, s, vh = (host(x) for x in result)
    if s.ndim != 1 or len(s) < rank:
        raise ValueError(f'Expected at least {rank} singular values in a 1-D array, got shape {s.shape}')
    if u.ndim != 2 or vh.ndim != 2 or u.shape[1] < rank or vh.shape[0] < rank:
        raise ValueError(f'Expected U (m, k) and Vh (k, n) with k >= {rank}, got {u.shape} and {vh.shape}')
    return u[:, :rank], s[:rank], vh[:rank]


def _ok(metrics, dtype, shape, exact):
    tol = TOLERANCE_FACTOR*np.finfo(np.dtype(dtype)).eps*max(shape)
    if not metrics.get('singular_values_nonnegative_sorted'):
        return False
    if 'reconstruction_relative_fro' not in metrics:
        return metrics['spectrum_error_over_sigma1'] <= tol
    for key in ('orthogonality_u_fro_over_sqrt_cols', 'orthogonality_v_fro_over_sqrt_rows'):
        if not metrics[key] <= tol:
            return False
    if exact or metrics['residual_over_optimal_tail'] is None:
        return metrics['reconstruction_relative_fro'] <= tol
    return metrics['residual_over_optimal_tail'] <= TAIL_RATIO_LIMIT


def evaluate(svd, cases=DEFAULT_SUITE, *, name=None, rank=None, device='cpu', repeat=5, warmup=1,
             threads=4, numpy_input=False, verbose=True):
    """One row per matrix. `rank=None` uses each case's output_rank, or the full rank."""
    fn = resolve(svd)
    label = name or (svd if isinstance(svd, str) else getattr(svd, '__name__', 'method'))
    device = torch.device(device)
    if numpy_input and device.type != 'cpu':
        raise ValueError('NumPy input is CPU only')
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    with threadpool_limits(threads), torch.inference_mode():
        for spec in cases:
            case_name, case, output_rank = make_case(spec)
            m, n = case.array.shape
            k = min(rank or output_rank or min(m, n), m, n)
            row = dict(method=label, case=case_name, m=m, n=n, dtype=str(case.array.dtype), rank=k,
                       status='error', ok=False, error=None)
            try:
                a = case.array.copy() if numpy_input else case.tensor(device)
                metrics = validate(case.array, _factors(fn(a, k), k), case.reference)['worst']
                row.update(metrics)
                row['ok'] = bool(_ok(metrics, case.array.dtype, (m, n), exact=k == min(m, n)))
                timing = benchmark(lambda: fn(a, k), device, repeat, warmup)['wall_ms']
                row.update(status='done', time_median_ms=timing['median'], time_mad_ms=timing['mad'])
            except Exception as exc:  # A failing case must not hide the other results.
                row['error'] = f'{type(exc).__name__}: {exc}'
            rows.append(row)
            if verbose:
                print(_line(row))
    return rows


def compare(methods, cases=DEFAULT_SUITE, **options):
    """methods: {label: function or method name}. Rows of all methods, same matrices."""
    if not isinstance(methods, dict):
        methods = {m if isinstance(m, str) else getattr(m, '__name__', str(i)): m for i, m in enumerate(methods)}
    cases = list(cases)
    return [row for label, method in methods.items() for row in evaluate(method, cases, name=label, **options)]


def table(rows):
    """pandas DataFrame when pandas is installed (pip install -e '.[laboratory]')."""
    import pandas as pd
    return pd.DataFrame(rows, columns=[c for c in COLUMNS if any(c in r for r in rows)])


def summary(rows):
    """Per method: cases passed, worst errors and total median time."""
    out = {}
    for r in rows:
        s = out.setdefault(r['method'], dict(method=r['method'], cases=0, ok=0, failed_cases=[],
                                              worst_reconstruction=0., worst_orthogonality=0., total_median_ms=0.))
        s['cases'] += 1
        s['ok'] += bool(r['ok'])
        if not r['ok']:
            s['failed_cases'].append(r['case'])
        if r['status'] == 'done':
            s['total_median_ms'] += r['time_median_ms']
            s['worst_reconstruction'] = max(s['worst_reconstruction'], r.get('reconstruction_relative_fro') or 0.)
            s['worst_orthogonality'] = max(s['worst_orthogonality'], r.get('orthogonality_u_fro_over_sqrt_cols') or 0.,
                                           r.get('orthogonality_v_fro_over_sqrt_rows') or 0.)
    return list(out.values())


def save(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.json':
        path.write_text(json.dumps(rows, indent=2, default=str))
    else:
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    return path


def _number(x):
    return '-' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.2e}'


def _line(r):
    if r['status'] != 'done':
        return f"{r['method']:<16} {r['case']:<22} ERROR {r['error']}"
    return (f"{r['method']:<16} {r['case']:<22} {'ok  ' if r['ok'] else 'FAIL'} "
            f"{r['time_median_ms']:>9.3f} ms  recon {_number(r.get('reconstruction_relative_fro'))}  "
            f"tail× {_number(r.get('residual_over_optimal_tail'))}  "
            f"σ-rel {_number(r.get('resolved_spectrum_max_relative_error'))}  "
            f"orth {_number(max(r.get('orthogonality_u_fro_over_sqrt_cols') or 0, r.get('orthogonality_v_fro_over_sqrt_rows') or 0))}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method', action='append', help='torch, numpy, mine, lab.randomized, a standalone '
                   'algorithm (jacobi, bidiag_qr, ...) or module:function; repeatable. Default: mine and torch')
    p.add_argument('--device', default='cpu')
    p.add_argument('--rank', type=int, help='Override every case; default is each case\'s rank')
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--output', help='.csv or .json')
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    rows = compare({m: m for m in args.method or ['mine', 'torch']}, rank=args.rank, device=args.device,
                   repeat=args.repeat, warmup=args.warmup, threads=args.threads)
    print()
    for s in summary(rows):
        print(f"{s['method']}: {s['ok']}/{s['cases']} ok, total median {s['total_median_ms']:.3f} ms"
              + (f", failed: {', '.join(s['failed_cases'])}" if s['failed_cases'] else ''))
    if args.output:
        print(f'Saved {save(rows, args.output).resolve()}')


if __name__ == '__main__':
    main()
