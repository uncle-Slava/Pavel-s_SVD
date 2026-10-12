"""Accuracy-gated report and scientific plots for the larger-matrix benchmark."""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


LABELS={
 'vendor_gesvd':'cuSOLVER QR','vendor_gesvdj':'cuSOLVER Jacobi','vendor_gesvda':'cuSOLVER gesvda',
 'native_Xgesvdp':'cuSOLVER Xgesvdp','native_Xgesvdr':'cuSOLVER Xgesvdr (legacy wrapper)',
 'gram_library':'Library Gram','qr_library':'Library QR + SVD',
 'randomized_library':'Library randomized q2','randomized_library_q4':'Library randomized q4',
 'sparse_randomized_library':'Sparse library randomized q2','sparse_randomized_library_q4':'Sparse library randomized q4',
 'jacobi':'Standalone Jacobi','gram_jacobi':'Standalone Gram/Jacobi','polar_newton_schulz':'Standalone Newton–Schulz',
 'bidiag_dc':'Standalone bidiagonal D&C','bidiag_qr':'Standalone bidiagonal QR',
 'block_jacobi':'Standalone block Jacobi','tsqr':'Standalone TSQR','qr':'Standalone QR',
 'randomized':'Standalone randomized','subspace':'Standalone subspace','block_krylov':'Standalone block Krylov',
 'lanczos':'Standalone dense Lanczos','sparse_lanczos':'Standalone sparse Lanczos',
}
def name(row):
    key=row['id']
    if key in LABELS:return LABELS[key]
    if '_q' in key:
        base,q=key.rsplit('_q',1)
        label=LABELS.get(base,base).replace(' q2','')
        return label+' q'+q
    return key
def wall(row):return row['resident_solver']['wall']['median_ms']
def peak(row):return row['memory']['allocator_peak_allocated_bytes']/2**20
def metric(row,key):return row['metrics']['worst'].get(key)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('--output',required=True)
    p.add_argument('--supplement',action='append',default=[])
    args=p.parse_args();root=Path(args.directory);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for line in (root/'measurements.jsonl').read_text().splitlines()]
    for directory in args.supplement:
        rows += [json.loads(line) for line in (Path(directory)/'measurements.jsonl').read_text().splitlines()]
    attempts=len(rows)
    # Latest attempt is authoritative for a retried setting; retain every attempt on disk.
    rows=list({(r['case']['name'],r['id']):r for r in rows}.values())
    config=json.loads((root/'config.json').read_text());env=json.loads((root/'environment.json').read_text())
    cases=[c['name'] for c in config['cases']];summaries=[]
    lines=['# Larger-matrix SVD benchmark — Hetzner, 12 October 2026','',
           f'{len(rows)} distinct method/matrix comparisons on {len(cases)} workload cases, from {attempts} total attempts. Hardware: {env["accelerator"]["name"]}; Torch {env["torch"]}; CUDA build {env["cuda_build"]}.',
           '',f'Execution statuses: {dict(Counter(r["status"] for r in rows))}. Accuracy gates passed: {sum(r["assessment"]["passed"] for r in rows)}.',
           '', '## Fastest and lowest-memory accepted results','',
           'Winners are the lowest observed medians or allocator peaks among methods that pass the declared accuracy gates. Full and top-k outputs are ranked separately by workload. Three timed calls follow one warmup; small timing differences are not statistically established.','',
           '| Workload | Fastest accepted (ms) | Lowest peak allocated GPU memory (MiB) | Accepted / attempted |',
           '|---|---|---|---|']
    for case in cases:
        subset=[r for r in rows if r['case']['name']==case];good=[r for r in subset if r['assessment']['passed']]
        best=min(good,key=wall) if good else None;mem=min(good,key=peak) if good else None
        spectrum=min(good,key=lambda r:metric(r,'resolved_spectrum_max_relative_error')) if good else None
        recon=min(good,key=lambda r:metric(r,'reconstruction_relative_fro')) if good else None
        stand=[r for r in good if not ('library' in r['id'] or r['id'].startswith(('vendor_','native_')))]
        fastest_standalone=min(stand,key=wall) if stand else None
        item=dict(case=case,attempted=len(subset),accepted=len(good),fastest=best,lowest_memory=mem,
                  best_spectrum=spectrum,best_reconstruction=recon,fastest_standalone=fastest_standalone)
        summaries.append(item)
        lines.append(f'| {case} | {name(best)} **{wall(best):.3f}** | {name(mem)} **{peak(mem):.2f}** | {len(good)}/{len(subset)} |' if good else f'| {case} | No accepted result | No accepted result | 0/{len(subset)} |')
    lines+=['','## Accuracy observations','',
            '| Workload | Lowest relative reconstruction error among accepted | Smallest maximum resolved singular-value relative error among accepted |','|---|---|---|']
    for item in summaries:
        a,b=item['best_reconstruction'],item['best_spectrum']
        if a:lines.append(f'| {item["case"]} | {name(a)}: {metric(a,"reconstruction_relative_fro"):.3e} | {name(b)}: {metric(b,"resolved_spectrum_max_relative_error"):.3e} |')
    lines+=['','These two accuracy metrics can select different methods. Top-k reconstruction includes the irreducible discarded tail; compare its ratio to the optimal reference tail. Tiny differences near the reference or precision floor should not be treated as meaningful accuracy wins.','',
            '## Standalone implementations','',
            '| Workload | Fastest accepted standalone | Wall ms | Times slower than fastest accepted overall |','|---|---|---:|---:|']
    for item in summaries:
        a=item['fastest_standalone'];best=item['fastest']
        if a:lines.append(f'| {item["case"]} | {name(a)} | {wall(a):.3f} | {wall(a)/wall(best):.1f}× |')
        else:lines.append(f'| {item["case"]} | No accepted standalone result | — | — |')
    lines+=['','## Sensitivity to the FP32 accuracy budget','',
            'This secondary view relaxes only the full-spectrum FP32 reconstruction, spectrum/sigma1 and orthogonality limits from 2e-4 to 1e-3. It does not alter the primary ranking or the raw outcomes.','',
            '| Full FP32 workload | Fastest under 2e-4 | Fastest under 1e-3 |','|---|---|---|']
    for item in summaries:
        subset=[r for r in rows if r['case']['name']==item['case']]
        first=subset[0] if subset else None
        if not first or first['case']['dtype']!='float32' or first['algorithm'].get('rank',min(first['case']['m'],first['case']['n']))<min(first['case']['m'],first['case']['n']):continue
        relaxed=[r for r in subset if r['status']=='ok' and metric(r,'singular_values_nonnegative_sorted') and
                 all(metric(r,k) is not None and metric(r,k)<=1e-3 for k in ('reconstruction_relative_fro','spectrum_error_over_sigma1','orthogonality_u_fro_over_sqrt_cols','orthogonality_v_fro_over_sqrt_rows'))]
        if not relaxed:continue
        alt=min(relaxed,key=wall);best=item['fastest']
        lines.append(f'| {item["case"]} | '+(f'{name(best)} {wall(best):.3f} ms' if best else 'None')+f' | {name(alt)} {wall(alt):.3f} ms |')
    lines+=['','## Failed accuracy gates and incomplete measurements','',
            'Timeouts are **whole-worker limits**, including startup, warmup, three timed solves, an additional validation solve and CPU validation. They are not per-SVD lower bounds. Failed/timeout rows do not enter speed or memory rankings.','']
    for row in rows:
        if not row['assessment']['passed']:
            reasons='; '.join(row['assessment']['reasons'])
            lines.append(f'- `{row["case"]["name"]}` / `{row["id"]}`: {row["status"]}; {reasons}')
    lines+=['','## Measurement contract and limits','',
            '- Full-spectrum FP32 acceptance: reconstruction, spectrum error/sigma1 and normalized U/V orthogonality ≤2e-4; FP64: ≤1e-9. Sorted nonnegative singular values are required.',
            '- Top-k acceptance: residual ≤1.05× the optimal reference tail, maximum relative error of the resolved leading k singular values ≤1%, and the same orthogonality limits. If the reference tail is zero, the full reconstruction tolerance replaces the tail-ratio check.',
            '- The independent reference is SciPy LAPACK gesvd in FP64 on the actual rounded input. Values below its dimension-scaled reference floor are excluded from relative spectrum errors. The reference is not arbitrary precision.',
            '- Timing uses resident GPU input, synchronized wall clocks and CUDA events. Generation, CPU reference, transfers/preparation and validation are excluded from solver medians. TF32 is disabled, four CPU threads are used, and no CUDA graphs are replayed.',
            '- Memory is the Torch allocator peak during timed solves, including prepared input, outputs and Torch-visible workspace. It excludes CUDA context/driver overhead and allocations outside the allocator. Reserved and incremental peaks are also preserved in the raw records.',
            '- Sparse methods use actual sparse COO products, but the harness also retains dense GPU input and dense CPU references. Their reported peaks do not establish the minimum memory needed by a pure sparse application. Sparse conversion is outside resident solve time.',
            '- Xgesvdr uses the previously inspected legacy wrapper, which allocates full square U/V buffers even for a truncated result. Its memory result describes this wrapper, not an intrinsic requirement of randomized cuSOLVER.',
            '- Explicit vendor top-k SVD controls compute a full decomposition and slice it. Randomized library controls use optimized QR/SVD primitives; they are not standalone decompositions. Native wrappers include allocation, workspace query and status synchronization inside the measured call.',
            '- One input seed is used per matrix. Randomized implementations use different sketches; the native routine seed is not controlled through the legacy wrapper. Timing repeats are not independent accuracy trials. These results support the listed workloads, not a universal ranking.',
            '- Matrices and references are preserved in the server experiment directory. The machine-readable summary retains the complete winning records; measurements.jsonl contains every attempt, including failures. Coarse device telemetry is observational, not per-call energy or occupancy.',
            '', '![Speed and memory](broad-winners.png)','']
    (out/'BROAD_RESULTS.md').write_text('\n'.join(lines))
    (out/'summary.json').write_text(json.dumps(dict(environment=env,attempts=attempts,comparisons=len(rows),results=summaries),indent=2,allow_nan=False))
    fig,axes=plt.subplots(1,2,figsize=(15,9),layout='constrained')
    y=np.arange(len(summaries));good=[x for x in summaries if x['fastest']]
    for ax in axes:ax.set_yticks(y,[s['case'] for s in summaries]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axes[0].barh(y,[wall(s['fastest']) if s['fastest'] else np.nan for s in summaries],color='#187f75',height=.35,label='Fastest accepted overall')
    stand=[wall(s['fastest_standalone']) if s['fastest_standalone'] else np.nan for s in summaries]
    axes[0].barh(y+.36,stand,color='#7385c5',height=.3,label='Fastest accepted standalone')
    axes[0].set_xscale('log');axes[0].set_xlabel('Synchronized wall time (ms, log scale)');axes[0].legend(loc='lower right')
    axes[1].barh(y,[peak(s['lowest_memory']) if s['lowest_memory'] else np.nan for s in summaries],color='#ce9153',label='Lowest accepted allocator peak')
    axes[1].set_xlabel('Peak allocated GPU memory (MiB)');axes[1].legend(loc='lower right')
    axes[0].set_title('Latency — quality-gated results');axes[1].set_title('Minimum memory — independently chosen per case')
    fig.suptitle('Hetzner RTX PRO 4000 Blackwell · larger SVD workloads',fontsize=15)
    fig.savefig(out/'broad-winners.png',dpi=180);fig.savefig(out/'broad-winners.svg');plt.close(fig)
    selected=['dense1024','cluster512_fp64','lowrank8192x2048_k64','sparse4096x2048_d001']
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for ax,case in zip(axes.flat,selected):
        subset=[r for r in rows if r['case']['name']==case and r['status']=='ok']
        for r in subset:
            value=metric(r,'resolved_spectrum_max_relative_error')
            ax.scatter(wall(r),max(value,1e-16),s=max(25,min(peak(r)*2,500)),alpha=.65,
                       color='#187f75' if r['assessment']['passed'] else '#bb5b55',edgecolors='white')
        accepted=[r for r in subset if r['assessment']['passed']]
        picks=[]
        if accepted:picks=[min(accepted,key=wall),min(accepted,key=lambda r:metric(r,'resolved_spectrum_max_relative_error'))]
        for j,r in enumerate({r['id']:r for r in picks}.values()):
            # Keep labels inside the axes when the selected method is near the right edge.
            right=wall(r)>np.sqrt(min(wall(x) for x in subset)*max(wall(x) for x in subset))
            ax.annotate(name(r),(wall(r),max(metric(r,'resolved_spectrum_max_relative_error'),1e-16)),
                        xytext=(-8 if right else 8,10+12*j),textcoords='offset points',
                        ha='right' if right else 'left',fontsize=8)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_title(case);ax.grid(alpha=.2)
        ax.set_xlabel('Wall time (ms)');ax.set_ylabel('Max resolved singular-value relative error')
    fig.suptitle('Speed / spectral accuracy · green passes all gates, red fails\nMarker area represents allocated GPU peak (capped for readability)',fontsize=13)
    fig.savefig(out/'broad-tradeoffs.png',dpi=180);fig.savefig(out/'broad-tradeoffs.svg');plt.close(fig)
    # Matplotlib leaves spaces in SVG path lines; normalize generated text for Git.
    for path in (out/'broad-winners.svg',out/'broad-tradeoffs.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    print('Wrote',out/'BROAD_RESULTS.md',flush=True)


if __name__=='__main__':main()
