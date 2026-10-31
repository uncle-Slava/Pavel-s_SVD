"""svdlab command line.

    python -m svdlab check examples/my_svd.py                 # your SVD vs torch, CPU
    python -m svdlab check examples/my_svd.py --device cuda   # same on the GPU
    python -m svdlab check examples/native/my_svd.cu --device cuda
    python -m svdlab check my_svd.py --compare torch,jacobi --csv results.csv
    python -m svdlab check my_svd.py --gpu-check              # CPU vs GPU: same answers, on the GPU
    python -m svdlab list                                     # builtin methods, blocks, recipes, matrices

A method is a file (.py with a function `svd(a, rank)`, or .cpp/.cu), file.py:function,
module:function, or a builtin name such as torch or jacobi. Exit code 1 when a checked
method fails a case (methods given with --compare do not affect the exit code).
"""
import argparse
import sys
import torch
from . import blocks, recipes
from .scorecard import SUITES, check, gpu_check, label_of, resolve, save, summary, _number

LEGEND = ('recon  = |A - U diag(s) Vh| / |A|          tail× = recon / best possible (truncated cases)\n'
          'σ-rel  = max relative error of the values   orth  = max(|UᴴU - I|, |V Vᴴ - I|) / sqrt(k)\n'
          'ok     = orth and recon within 100·eps·max(m, n), or tail× <= 1.05')


def _cases(args):
    cases = list(SUITES[args.suite])
    if args.case:
        cases = [c for c in cases if c['name'] in args.case]
        unknown = set(args.case) - {c['name'] for c in cases}
        if unknown:
            raise SystemExit(f'Unknown case(s) {sorted(unknown)}; see python -m svdlab list')
    return cases


def run_check(args):
    torch.set_num_threads(args.threads)
    baselines = [m for item in args.compare for m in item.split(',') if m]
    cases = _cases(args)
    options = dict(rank=args.rank, repeat=args.repeat, warmup=args.warmup, threads=args.threads)
    methods = {}                                  # load (and compile) everything before measuring
    for method in args.methods + baselines:
        try:
            methods[method] = resolve(method)
        except Exception as exc:
            print(f'Cannot load {method}: {type(exc).__name__}: {exc}', file=sys.stderr)
            return 2
    if args.gpu_check:
        device = args.device if args.device != 'cpu' else 'cuda:0'
        failed = 0
        for method, fn in methods.items():
            for r in gpu_check(fn, cases, device=device, name=label_of(method), verbose=False, **options):
                failed += method in args.methods and not r['passed']
                speed = f"{r['speedup']:.2f}x" if r['speedup'] else '-'
                print(f"{r['method']:<20} {r['case']:<22} {'PASS' if r['passed'] else 'FAIL'} "
                      f"cpu_ok={r['cpu_ok']} gpu_ok={r['gpu_ok']} output={r['output_device']} "
                      f"Δσ/σ1={_number(r['max_singular_value_difference_over_sigma1'])} speedup={speed}"
                      + (f"  {r['error']}" if r['error'] else ''))
        return 1 if failed else 0
    print(LEGEND, end='\n\n')
    rows, checked = [], set()
    for method, fn in methods.items():
        label = label_of(method)
        if method in args.methods:
            checked.add(label)
        rows += check(fn, cases, name=label, device=args.device, **options)
    print()
    failed = 0
    for s in summary(rows):
        failed += s['method'] in checked and s['ok'] < s['cases']
        print(f"{s['method']}: {s['ok']}/{s['cases']} ok, total median {s['total_median_ms']:.3f} ms"
              + (f", failed: {', '.join(s['failed_cases'])}" if s['failed_cases'] else ''))
    for r in rows:
        if r['error']:
            print(f"\n{r['method']} / {r['case']}: {r['error']}")
            break
    if args.csv:
        print(f'Saved {save(rows, args.csv).resolve()}')
    return 1 if failed else 0


def run_list(_args):
    from svdbench.standalone import ALGORITHMS
    print('Builtin methods: torch, numpy, mine (algorithm_laboratory/my_algorithm.py), lab.randomized, lab.jacobi')
    print('Repository SVDs:', ', '.join(ALGORITHMS))
    print('Recipes (svdlab.recipes:<name>):', ', '.join(recipes.__all__))
    print('Blocks (svdlab.blocks):', ', '.join(n for n in blocks.__all__ if n != 'ALGORITHMS'))
    for name, suite in SUITES.items():
        print(f'\n{name} matrices' + (' (--quick)' if name == 'quick' else ''))
        for c in suite:
            extra = {k: v for k, v in c.items() if k not in ('name', 'm', 'n', 'family', 'dtype')}
            print(f"  {c['name']:<22} {c['m']}x{c['n']:<5} {c['family']:<10} {c.get('dtype', 'float64'):<10} "
                  + ' '.join(f'{k}={v}' for k, v in extra.items()))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m svdlab', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check', help='measure one or more SVD methods on the standard matrices')
    c.add_argument('methods', nargs='+', help='file.py, file.cu, file.cpp, module:function or builtin name')
    c.add_argument('--compare', action='append', default=[], metavar='M1,M2',
                   help='reference methods shown next to yours, e.g. torch,jacobi')
    c.add_argument('--device', default='cpu', help='cpu, cuda or cuda:N')
    c.add_argument('--quick', dest='suite', action='store_const', const='quick', default='default',
                   help='smaller matrices for a fast first run')
    c.add_argument('--case', action='append', help='run only this case (repeatable)')
    c.add_argument('--rank', type=int, help="requested rank for every case (default: each case's own)")
    c.add_argument('--repeat', type=int, default=5, help='timed runs per case (median reported)')
    c.add_argument('--warmup', type=int, default=1)
    c.add_argument('--threads', type=int, default=4, help='CPU threads')
    c.add_argument('--csv', help='save all rows (.csv or .json)')
    c.add_argument('--gpu-check', action='store_true',
                   help='run on CPU and --device (default cuda:0): outputs on GPU, both ok, same values')
    c.set_defaults(run=run_check)
    sub.add_parser('list', help='builtin methods, blocks, recipes and matrices').set_defaults(run=run_list)
    args = p.parse_args(argv)
    return args.run(args)


if __name__ == '__main__':
    sys.exit(main())
