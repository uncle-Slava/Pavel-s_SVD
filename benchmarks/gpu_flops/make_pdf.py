"""Static PDF of the benchmark report: algorithm diagrams, charts and tables, viewable on GitHub.

Builds a self-contained report_static.html from report_data/*.json, the chart SVGs (plot_results.py)
and the diagram section of report.html, then prints it with headless Chromium:

    python3 benchmarks/gpu_flops/make_pdf.py benchmarks/gpu_flops/results/2026-10-30-gb10

Chromium is found via $CHROME, then chromium / chromium-browser / google-chrome on PATH.
"""
import html, json, os, re, shutil, subprocess, sys
from collections import defaultdict

FAM = ['Reference (cuSOLVER)', 'Golub-Kahan bidiagonal', 'One-sided Jacobi', 'QR then Jacobi', 'Gram eigensolver',
       'Polar decomposition', 'Randomized / Krylov', 'Lanczos bidiagonal']
COLORS = ['#222222', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']  # as plot_results.py
SIZES = [32, 64, 128, 256]

d = sys.argv[1]
load = lambda name: json.load(open(f'{d}/report_data/{name}.json'))
runs, algos, score = load('runs'), load('algorithms_table'), load('scorecard_table')
env = json.load(open(f'{d}/env.json'))
src = open(f'{d}/report.html').read()
esc = html.escape


def section(sid):
    m = re.search(rf'<section id="{sid}"[^>]*>.*?</section>', src, re.S)
    return m.group(0) if m else ''


def ms(v):
    if v is None:
        return ''
    return f'{v:,.0f}' if v >= 100 else f'{v:.1f}' if v >= 10 else f'{v:.2f}' if v >= 1 else f'{v:.3f}'


sci = lambda v: '' if v is None else f'{v:.1e}'
f32 = [r for r in runs if r['dtype'] == 'float32']
by = defaultdict(dict)
for r in f32:
    by[r['impl']][r['n']] = r
order = sorted(by, key=lambda k: (FAM.index(next(iter(by[k].values()))['family']), k))
fam_of = {k: next(iter(by[k].values()))['family'] for k in by}


def cell(r, key, fmt, cls='num'):
    if r is None:
        return f'<td class="{cls} muted">skipped</td>'
    if r['status'] != 'ok':
        return f'<td class="{cls} bad">{esc(r["status"])}</td>'
    return f'<td class="{cls}">{fmt(r[key])}</td>'


def name_cell(impl):
    c = COLORS[FAM.index(fam_of[impl])]
    return f'<td><i class="dot" style="background:{c}"></i>{esc(impl)}</td>'


def table(head, rows, cls=''):
    th = ''.join(f'<th class="{"num" if i else ""}">{h}</th>' for i, h in enumerate(head))
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


# Time and throughput
t_rows = []
for impl in order:
    rs = by[impl]
    big = max(rs)
    t_rows.append('<tr>' + name_cell(impl) + ''.join(cell(rs.get(n), 'median_ms', ms) for n in SIZES)
                  + cell(rs[big], 'total_flops', lambda v: f'{v / 1e9:.3f}') + cell(rs[big], 'gflops', lambda v: f'{v:.1f}')
                  + f'<td class="num">{big}</td></tr>')
time_table = table(['method', 'n=32', '64', '128', '256', 'GFLOP at largest n', 'GFLOP/s', 'largest n'], t_rows)

# Accuracy at n = 128 (every method ran there)
a_rows = []
for impl in order:
    r = by[impl].get(128)
    a_rows.append('<tr>' + name_cell(impl) + ''.join(cell(r, k, sci) for k in ('recon_err', 'orth_u', 'orth_v', 'sv_err'))
                  + f'<td class="wrap">{esc(str(r["iterations"])) if r else ""}</td></tr>')
acc_table = table(['method', '‖A−UΣVᵀ‖/‖A‖', '‖UᵀU−I‖', '‖VVᵀ−I‖', 'max rel. σ error', 'iterations'], a_rows)

# float64 subset
f64 = defaultdict(dict)
for r in runs:
    if r['dtype'] == 'float64':
        f64[r['impl']][r['n']] = r
d_rows = ['<tr>' + name_cell(impl) + ''.join(cell(f64[impl].get(n), 'median_ms', ms) for n in SIZES)
          + cell(f64[impl].get(128), 'recon_err', sci) + '</tr>' for impl in sorted(f64, key=order.index)]
f64_table = table(['method', 'n=32', '64', '128', '256', 'recon. error at 128'], d_rows)

alg_rows = [f'<tr><td><i class="dot" style="background:{COLORS[FAM.index(a["family"])]}"></i>{esc(a["impl"])}</td>'
            f'<td class="wrap">{esc(a["steps"])}</td><td class="wrap">{esc(a["iterative"])}</td><td>{esc(a["rank"])}</td></tr>'
            for a in algos]
alg_table = table(['method', 'steps', 'iterations', 'rank'], alg_rows, 'text')

s_rows = [f'<tr><td>{esc(s["method"])}</td><td>{esc(s["case"])}</td><td>{esc(s["shape"])} {esc(s["dtype"])}</td>'
          f'<td class="num">{ms(s["time_ms"])}</td><td class="num">{sci(s["recon"])}</td><td class="num">{sci(s["sv_rel"])}</td>'
          f'<td class="num">{sci(s["orth"])}</td><td class="{"good" if s["ok"] == "ok" else "bad"}">{esc(s["ok"])}</td>'
          f'<td>{esc(s.get("gpu_check", ""))}</td></tr>' for s in score]
score_table = table(['method', 'case', 'shape / dtype', 'ms', 'recon', 'σ rel', 'orth', 'result', 'GPU check'], s_rows)


def svg(name):
    s = open(f'{d}/{name}.svg').read()
    return s[s.index('<svg'):]


legend = ''.join(f'<span><i class="dot" style="background:{c}"></i>{esc(f)}</span>' for f, c in zip(FAM, COLORS))
schemes = section('schemes').replace('<section', '<section class="page"')
for f, c in zip(FAM, COLORS):
    schemes = schemes.replace(f'class="famdot" data-fam="{f}"', f'class="dot" style="background:{c}"')
method = section('method')
torch_v, cuda_v = env.get('torch', 'n/a'), env.get('cuda', 'n/a')

page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>SVD GPU benchmark</title><style>
@page {{ size: A4 landscape; margin: 12mm; }}
:root {{ --fg:#1b1d22; --muted:#5d6370; --line:#d9dce2; --panel:#f4f5f7; --bad:#b42318; --ok:#1a7f37; }}
body {{ font: 10px/1.4 "DejaVu Sans", Helvetica, Arial, sans-serif; color: var(--fg); margin: 0; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} h2 {{ font-size: 15px; margin: 0 0 6px; }} h3 {{ font-size: 12px; margin: 0 0 4px; }}
p, li {{ max-width: 120ch; }} .muted {{ color: var(--muted); }}
.page {{ break-before: page; }} section {{ margin-bottom: 10px; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 2px 6px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
th {{ background: var(--panel); white-space: nowrap; }} tr {{ break-inside: avoid; }}
td.num, th.num {{ text-align: right; white-space: nowrap; }} td.bad {{ color: var(--bad); }} td.good {{ color: var(--ok); }}
table.text td {{ font-size: 9.5px; }} td.wrap {{ max-width: 60ch; }}
.dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 5px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 4px 14px; margin: 4px 0 8px; }}
.chart svg {{ width: 100%; height: auto; max-height: 165mm; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }}
.card {{ border: 1px solid var(--line); border-radius: 4px; padding: 8px; break-inside: avoid; }}
.flow {{ display: flex; flex-wrap: wrap; gap: 5px; align-items: stretch; margin: 4px 0; }}
.step {{ border: 1px solid var(--line); border-radius: 4px; padding: 4px 6px; display: grid; gap: 1px; font-size: 9.5px; }}
.step.loop {{ border-style: dashed; border-color: var(--muted); }} .step.loop::before {{ content: "↻ iterative"; font-size: 8.5px; color: var(--muted); }}
.step.in, .step.out {{ border-color: transparent; align-content: center; font-size: 12px; }}
.arr {{ align-self: center; color: var(--muted); }} .members {{ font-size: 9.5px; color: var(--muted); }}
.notes ul {{ padding-left: 16px; }}
</style></head><body>
<section>
<h1>SVD GPU benchmark: Pavel-s_SVD on NVIDIA GB10</h1>
<p class="muted">30 October 2026 · PyTorch {esc(str(torch_v))} · CUDA {esc(str(cuda_v))} · float32 square Gaussian matrices, n = 32, 64, 128, 256 · interactive version: report.html (claude.ai Dashboard runtime)</p>
<h2>Findings</h2>
<ul>
<li><b>Speed.</b> Every repository solver is limited by kernel launches and host synchronization, not by arithmetic: at most about 6 GFLOP/s against 49 GFLOP/s for cuSOLVER. Time grows roughly linearly in n instead of as n³. At n = 256 the fastest repository SVDs (polar_polynomial, polar_newton_schulz, polar_cans, about 0.69 s) are about 97× slower than torch.linalg.svd (7.2 ms).</li>
<li><b>Slowest.</b> bidiag_qr / recipes.golub_kahan / recipes.lanczos (bdsqr is a Python loop with Givens rotations) and block_jacobi (4960 inner Jacobi solves at n = 128) are an order of magnitude slower; their n = 256 cases were skipped.</li>
<li><b>Accuracy.</b> bidiag_dc reconstructs best (6e-7, float64 merges), then bidiag_qr (2e-6); the Jacobi family is at about 5e-5 and the polar family at about 1e-4.</li>
<li><b>Gram methods.</b> gram_jacobi and gram_dc square the condition number and lose the smallest singular values at n = 256 (relative σ error 1.0).</li>
<li><b>Failures.</b> None. The standard svdlab check on CUDA passes for my_svd, torch and jacobi.</li>
</ul>
</section>
<section class="page">
<h2>Median time per SVD, ms (float32)</h2>
<div class="legend">{legend}</div>
{time_table}
<p class="muted">GFLOP are executed operations (torch.linalg.svd: textbook estimate 21n³). “skipped”: not run at n = 256 because n = 128 took over 5 s.</p>
</section>
{schemes}
<section class="page chart"><h2>Median time vs n</h2>{svg('time_vs_n')}</section>
<section class="page chart"><h2>Executed FLOPs vs n</h2>{svg('flops_vs_n')}</section>
<section class="page chart"><h2>Latency vs reconstruction error</h2><p class="muted">Each method's points are joined from n = 32 to 256; marker size grows with n.</p>{svg('latency_vs_error')}</section>
<section class="page"><h2>Accuracy and iterations at n = 128 (float32)</h2>{acc_table}
<h2 style="margin-top:12px">float64 subset, median ms</h2>{f64_table}</section>
<section class="page"><h2>Algorithms</h2>{alg_table}</section>
<section class="page"><h2>Standard check: svdlab check on CUDA</h2>{score_table}</section>
{method}
</body></html>'''

out_html, out_pdf = f'{d}/report_static.html', f'{d}/report.pdf'
open(out_html, 'w').write(page)
chrome = os.environ.get('CHROME') or next(filter(None, map(shutil.which, ('chromium', 'chromium-browser', 'google-chrome'))), None)
if not chrome:
    sys.exit(f'wrote {out_html}; no Chromium found (set CHROME) to print {out_pdf}')
subprocess.run([chrome, '--headless', '--no-sandbox', '--disable-gpu', '--no-pdf-header-footer',
                f'--print-to-pdf={os.path.abspath(out_pdf)}', 'file://' + os.path.abspath(out_html)], check=True)
print('wrote', out_html, out_pdf)
