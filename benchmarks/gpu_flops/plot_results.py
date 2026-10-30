"""Static charts from runs.json (prep.py output): time vs n, FLOPs vs n, latency vs reconstruction error.

    .venv/bin/python benchmarks/gpu_flops/plot_results.py benchmarks/gpu_flops/results/2026-10-30-gb10
"""
import json, sys
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FAM = ['Reference (cuSOLVER)', 'Golub-Kahan bidiagonal', 'One-sided Jacobi', 'QR then Jacobi', 'Gram eigensolver',
       'Polar decomposition', 'Randomized / Krylov', 'Lanczos bidiagonal']
COLORS = ['#222222', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
MARKERS = ['o', 's', '^', 'D', '*', 'v', 'P']

d = sys.argv[1]
runs = [r for r in json.load(open(f'{d}/report_data/runs.json')) if r['dtype'] == 'float32' and r['status'] == 'ok']
by = defaultdict(list)
for r in runs:
    by[r['impl']].append(r)
style = {}
for fam in FAM:
    for i, impl in enumerate(sorted({r['impl'] for r in runs if r['family'] == fam})):
        style[impl] = dict(color=COLORS[FAM.index(fam)], marker=MARKERS[i % len(MARKERS)])
order = sorted(by, key=lambda k: (FAM.index(by[k][0]['family']), k))


def finish(ax, fig, name, title, xl, yl):
    ax.set_xscale('log', base=2 if xl == 'n' else 10)
    ax.set_yscale('log')
    if xl == 'n':
        ax.set_xticks([32, 64, 128, 256], ['32', '64', '128', '256'])
    ax.set_title(title)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid(True, which='major', alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(f'{d}/{name}.svg')
    fig.savefig(f'{d}/{name}.png', dpi=130)
    plt.close(fig)


for field, name, title, yl in (('median_ms', 'time_vs_n', 'Median time per SVD (float32, NVIDIA GB10)', 'ms'),
                               ('total_flops', 'flops_vs_n', 'Executed FLOPs per SVD (float32)', 'FLOP')):
    fig, ax = plt.subplots(figsize=(9, 5))
    for impl in order:
        pts = sorted((r['n'], r[field]) for r in by[impl] if r[field])
        if pts:
            ax.plot(*zip(*pts), label=impl, lw=1.6, ms=5, **style[impl])
    finish(ax, fig, name, title, 'n', yl)

fig, ax = plt.subplots(figsize=(9, 5.5))
for impl in order:
    pts = sorted((r['n'], r['median_ms'], r['recon_err']) for r in by[impl])
    ax.plot([p[1] for p in pts], [p[2] for p in pts], lw=1, alpha=0.6, color=style[impl]['color'])
    ax.scatter([p[1] for p in pts], [p[2] for p in pts], s=[12 + 10 * (p[0] // 32) for p in pts],
               label=impl, **{'color': style[impl]['color'], 'marker': style[impl]['marker']})
finish(ax, fig, 'latency_vs_error', 'Latency vs reconstruction error (float32; marker size grows with n = 32..256)',
       'median time, ms', '||A - U S V^T||_F / ||A||_F')
print('charts written to', d)
