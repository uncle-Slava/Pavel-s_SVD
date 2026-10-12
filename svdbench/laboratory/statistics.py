"""Uninstrumented timings and explicit descriptive statistics."""
import gc
from contextlib import nullcontext
import numpy as np
import torch
from svdbench.cli import measure, synchronize


def describe(samples, seed=7):
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Expected nonempty finite samples')
    median = float(np.median(x))
    ci = None
    if len(x) >= 5:
        rng = np.random.default_rng(seed)
        medians = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(2000)]
        ci = np.quantile(medians, [.025, .975]).tolist()
    return dict(n=len(x), samples=x.tolist(), mean=float(x.mean()), median=median,
                std_sample=float(x.std(ddof=1)) if len(x)>1 else None,
                mad=float(np.median(abs(x-median))), min=float(x.min()), max=float(x.max()),
                p10=float(np.quantile(x,.1)), p90=float(np.quantile(x,.9)),
                median_bootstrap_95_interval=ci,
                interval_note='Describes these samples only; autocorrelated timings and one seed do not establish general performance.')


def benchmark(fn, device='cpu', repeat=10, warmup=2):
    if not isinstance(repeat, int) or repeat<1 or not isinstance(warmup, int) or warmup<0:
        raise ValueError('repeat must be positive and warmup nonnegative')
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Laboratory timing currently supports CPU and CUDA/ROCm')
    device_context = torch.cuda.device(device) if device.type=='cuda' else nullcontext()
    with device_context, torch.inference_mode():
        for _ in range(warmup):
            result = fn(); synchronize(device); del result
        gc.collect()
        baseline = None
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
        timed = measure(fn, device, repeat, events=True)
        memory = dict(allocator_baseline_bytes=baseline, allocator_peak_allocated_bytes=None,
                      allocator_peak_increment_bytes=None, allocator_peak_reserved_bytes=None,
                      note='Torch GPU allocator only; CPU, driver and non-Torch allocations are not measured.')
        if device.type == 'cuda':
            peak = torch.cuda.max_memory_allocated(device)
            memory.update(allocator_peak_allocated_bytes=peak, allocator_peak_increment_bytes=peak-baseline,
                          allocator_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return dict(wall_ms=describe(timed['wall']['samples_ms']),
                device_event_ms=describe(timed['device_event']['samples_ms']) if timed['device_event'] else None,
                memory=memory, repeat=repeat, warmup=warmup, inspection_enabled=False)
