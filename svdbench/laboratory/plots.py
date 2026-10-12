"""Scientific figures; each underlying number remains in events.jsonl."""
from collections import defaultdict
from pathlib import Path
import numpy as np


def plot_run(run, destination=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    checkpoints = [e for e in run.events if 'accuracy' in e]
    for key, label in [('reconstruction_relative_fro','Reconstruction'), ('spectrum_error_over_sigma1','Spectrum / sigma1')]:
        values = [(e['event_id'], e['accuracy']['worst'].get(key)) for e in checkpoints]
        values = [(x,y) for x,y in values if y is not None and np.isfinite(y)]
        if values:
            axes[0,0].semilogy([x for x,y in values],[max(y,1e-18) for x,y in values],'.-',label=label)
    for key, label in [('orthogonality_u_fro_over_sqrt_cols','U'), ('orthogonality_v_fro_over_sqrt_rows','V')]:
        values = [(e['event_id'],e['accuracy']['worst'].get(key)) for e in checkpoints]
        values = [(x,y) for x,y in values if y is not None and np.isfinite(y)]
        if values: axes[0,1].semilogy([x for x,y in values],[max(y,1e-18) for x,y in values],'.-',label=label)
    stages = defaultdict(list)
    for e in run.select(kind='stage'): stages[e['name']].append(e['wall_ms'])
    names = list(stages)
    axes[1,0].barh(names,[sum(stages[n]) for n in names],color='#317c83')
    axes[1,0].set_xlabel('Total observed stage wall time (ms)')
    axes[1,0].set_title('Inspection pass — not benchmark timings')
    memory = [(e['event_id'], e['allocator_peak_allocated_bytes']/2**20)
              for e in run.select(kind='stage') if e.get('allocator_peak_allocated_bytes') is not None]
    if memory:
        axes[1,1].plot([x for x,y in memory],[y for x,y in memory],'.-')
    else:
        axes[1,1].text(.5,.5,'GPU allocator memory unavailable on CPU',ha='center',transform=axes[1,1].transAxes)
    axes[1,1].set(xlabel='Event ID', ylabel='Torch allocated peak (MiB)', title='Memory by measured stage')
    for ax,title in zip(axes[0],('Accuracy against independent reference','Normalized orthogonality error')):
        ax.set(xlabel='Checkpoint event ID',title=title)
        if ax.lines: ax.legend()
    for ax in axes.flat: ax.grid(alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Algorithm laboratory · '+run.path.name)
    path = Path(destination) if destination else run.path/'trace.png'
    fig.savefig(path,dpi=150)
    plt.close(fig)
    return path
