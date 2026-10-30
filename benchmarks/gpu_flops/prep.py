"""Usage: python3 prep.py <dir with results.json, flops.json, algorithms.json, scorecard files> <out dir>

Turn raw benchmark output (results.json, flops.json, algorithms.json, scorecard) into the
three tables the report reads: runs.json, algorithms_table.json, scorecard_table.json."""
import json, sys, os

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
res = json.load(open(f'{src}/results.json'))
flops = {(r['impl'], r['n']): r for r in json.load(open(f'{src}/flops.json'))} if os.path.exists(f'{src}/flops.json') else {}
alg = json.load(open(f'{src}/algorithms.json'))

FAMILY = {
    'torch.linalg.svd': 'Reference (cuSOLVER)',
    'bidiag_qr': 'Golub-Kahan bidiagonal', 'bidiag_dc': 'Golub-Kahan bidiagonal', 'recipes.golub_kahan': 'Golub-Kahan bidiagonal',
    'jacobi': 'One-sided Jacobi', 'block_jacobi': 'One-sided Jacobi',
    'qr': 'QR then Jacobi', 'tsqr': 'QR then Jacobi', 'recipes.qr_then_svd': 'QR then Jacobi',
    'gram_jacobi': 'Gram eigensolver', 'gram_dc': 'Gram eigensolver',
    'polar_qdwh': 'Polar decomposition', 'polar_newton_schulz': 'Polar decomposition', 'polar_halley': 'Polar decomposition',
    'polar_polynomial': 'Polar decomposition', 'polar_cans': 'Polar decomposition',
    'randomized': 'Randomized / Krylov', 'subspace': 'Randomized / Krylov', 'block_krylov': 'Randomized / Krylov',
    'recipes.randomized': 'Randomized / Krylov',
    'lanczos': 'Lanczos bidiagonal', 'recipes.lanczos': 'Lanczos bidiagonal',
}


def label(impl):
    if impl.startswith('blocks.svd:'):
        return impl.split(':')[1]
    return impl


def iter_text(r):
    st, it = r.get('stats') or {}, r.get('iters') or {}
    parts = []
    if 'polar_iterations' in st:
        parts.append(f"polar {st['polar_iterations']} it")
    if 'block_sweeps' in st:
        parts.append(f"outer {st['block_sweeps']} sweeps")
    if 'lanczos_steps' in st:
        parts.append(f"Lanczos {st['lanczos_steps']} steps")
    if 'power_iterations' in st:
        parts.append(f"power {st['power_iterations']} it")
    for k, name in (('bdsqr_iterations', 'bdsqr'), ('jacobi_svd_sweeps', 'Jacobi'), ('jacobi_eigh_sweeps', 'Jacobi-eigh')):
        v = it.get(k)
        if v is None:
            continue
        if isinstance(v, dict):
            parts.append(f"{name} {v['total']} sweeps in {v['calls']} calls (max {v['max']})" if name != 'bdsqr'
                         else f"bdsqr {v['total']} it in {v['calls']} calls")
        else:
            parts.append(f"{name} {v} {'it' if name == 'bdsqr' else 'sweeps'}")
    for k in ('jacobi_merge_refinements', 'jacobi_eigen_refinements'):
        if st.get(k):
            parts.append(f"{st[k]} merge refinements")
    return ', '.join(parts) if parts else ('vendor kernel' if r['impl'] == 'torch.linalg.svd' else '')


runs = []
for r in res:
    impl, n = r['impl'], r['n']
    lab = label(impl)
    row = dict(key=f"{lab}|{n}|{r['dtype']}", impl=lab, family=FAMILY[lab], n=n, dtype=r['dtype'],
               status=r['status'] if r['status'] in ('ok', 'failed', 'skipped') else 'ok',
               error=r.get('error') or '', median_ms=r.get('median_ms'), min_ms=r.get('min_ms'),
               timed_runs=r.get('timed_runs'), recon_err=r.get('recon_err'), orth_u=r.get('orth_u'),
               orth_v=r.get('orth_v'), sv_err=r.get('sv_err'), iterations=iter_text(r) if r['status'] == 'ok' else '',
               matmul_flops=None, elementwise_flops=None, total_flops=None, gflops=None, flops_basis='')
    if r['dtype'] == 'float32':
        if lab == 'torch.linalg.svd':
            row.update(total_flops=21 * n ** 3, flops_basis='textbook estimate 21n^3 (Golub-Reinsch with U, V)')
        else:
            f = flops.get((impl, n))
            if f and f.get('status') == 'ok':
                row.update(matmul_flops=f['matmul_flops'], elementwise_flops=f['elementwise_flops'],
                           total_flops=f['total_flops'], flops_basis='counted: matmul measured + primitives exact')
        if row['total_flops'] and row['median_ms'] and row['status'] == 'ok':
            row['gflops'] = row['total_flops'] / (row['median_ms'] * 1e6)
    runs.append(row)

algos = []
for impl, d in alg.items():
    if impl.startswith('_'):
        continue
    lab = label(impl)
    algos.append(dict(impl=lab, family=FAMILY.get(lab, 'One-sided Jacobi'), steps=d['steps'], iterative=d['iterative'], rank=d['rank']))

score = []
sc = f'{src}/scorecard_cuda_table.json'
for r in (json.load(open(sc)) if os.path.exists(sc) else []):
    score.append(dict(key=f"{r['method']}|{r['case']}", method=r['method'], case=r['case'],
                      shape=f"{r['m']}x{r['n']}", dtype=r['dtype'], ok='ok' if r['ok'] else 'FAIL',
                      time_ms=r['time_median_ms'], recon=r['reconstruction_relative_fro'],
                      sv_rel=r['resolved_spectrum_max_relative_error'],
                      orth=max(r['orthogonality_u_fro_over_sqrt_cols'] or 0, r['orthogonality_v_fro_over_sqrt_rows'] or 0)))
gpu = {}
txt = f'{src}/scorecard_gpu_check.txt'
if os.path.exists(txt):
    for line in open(txt):
        p = line.split()
        if len(p) > 3 and p[2] in ('PASS', 'FAIL'):
            gpu[(p[0], p[1])] = p[2]
for s in score:
    s['gpu_check'] = gpu.get((s['method'], s['case']), '')

json.dump(runs, open(f'{out}/runs.json', 'w'), indent=0)
json.dump(algos, open(f'{out}/algorithms_table.json', 'w'), indent=0)
json.dump(score, open(f'{out}/scorecard_table.json', 'w'), indent=0)
print(len(runs), len(algos), len(score))
