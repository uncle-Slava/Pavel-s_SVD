"""Static research figure for an accuracy-annotated standalone SVD comparison."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rows=list(csv.DictReader(open('results/standalone-matrix-review.csv')))
rows=[r for r in rows if r['case']=='square32']
order=['gesvdj','gesvd','jacobi','block_jacobi','bidiag_qr','bidiag_dc','qr','tsqr','gram_jacobi','gram_dc','polar_qdwh','polar_newton_schulz','polar_polynomial','polar_halley','randomized','subspace','block_krylov','lanczos']
rows=sorted(rows,key=lambda r:order.index(r['algorithm']))
y=np.arange(len(rows))
colors=['#bd3f3f' if r['passed_pilot_budget']=='False' else ('#117b71' if r['algorithm'] in ('gesvd','gesvdj') else '#3566a2') for r in rows]
fig,axs=plt.subplots(1,3,figsize=(14,8.5),sharey=True,gridspec_kw={'width_ratios':[1.2,1,1]})
axs[0].barh(y,[float(r['wall_ms']) for r in rows],color=colors,height=.65)
axs[0].set_xscale('log');axs[0].set_xlabel('Median synchronized wall time (ms)')
axs[0].set_yticks(y,[r['algorithm'] for r in rows]);axs[0].invert_yaxis()
axs[1].scatter([max(float(r['residual']),1e-16) for r in rows],y,color=colors,marker='o',label='Reconstruction')
axs[1].scatter([max(float(r['orth_u']),1e-16) for r in rows],y,color=colors,marker='x',label='U orthogonality')
axs[1].axvline(2e-4,color='#bd3f3f',linestyle=':',lw=1,label='Pilot budget')
axs[1].set_xscale('log');axs[1].set_xlabel('Error (smaller is better)');axs[1].legend(loc='lower right',fontsize=8)
axs[2].barh(y,[float(r['peak_increment_mib']) for r in rows],color=colors,height=.65)
axs[2].set_xscale('log');axs[2].set_xlabel('Incremental allocator peak (MiB)')
for ax in axs:
    ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Standalone SVD: correctness baseline and dispatch cost',fontsize=17,x=.51,y=.97)
fig.text(.51,.925,'RTX PRO 4000 Blackwell SFF · 32 x 32 Gaussian FP32 · reduced full spectrum',ha='center',fontsize=11)
fig.text(.02,.025,'3 timed samples after warmup. Blue: standalone tensor arithmetic. Green: cuSOLVER. Red: fails the stated accuracy budget.\nNo graph capture; time includes Python control and synchronization. Memory is the PyTorch allocator increment, excluding resident input.',fontsize=9)
fig.subplots_adjust(left=.205,right=.98,bottom=.12,top=.89,wspace=.28)
for ext in ('png','svg'):
    fig.savefig(Path('results')/f'standalone-comparison.{ext}',dpi=170)
