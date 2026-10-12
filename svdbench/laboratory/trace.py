"""Append-only observations, bounded snapshots and reloadable experiment runs."""
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from svdbench.cli import synchronize
from .diagnostics import accuracy, spectrum_errors, tensor_statistics, host


def clean(value):
    """Strict JSON: nonfinite scalar observations become null, never NaN literals."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError('Put array observations in tensors, not scalars')
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, complex):
        return {'real':clean(value.real), 'imag':clean(value.imag)}
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')


class NullTrace:
    enabled = False
    def wants(self, iteration=None): return False
    def stage(self, *args, **kwargs): return nullcontext()
    def checkpoint(self, *args, **kwargs): return None
    def record(self, *args, **kwargs): return None


class Trace:
    """Use with Trace(...). Every checkpoint is outside timed stages.

    Saved arrays are CPU copies. The snapshot budget excludes input/reference
    files and JSON statistics. No frames, hidden locals or autograd graphs persist.
    """
    enabled = True

    def __init__(self, directory, case, device='cpu', snapshots='full',
                 snapshot_budget_bytes=64*2**20, every=1, max_events=10000,
                 parameters=None, source_files=()):
        if snapshots not in ('full', 'stats') or every < 1 or max_events < 1 or snapshot_budget_bytes < 0:
            raise ValueError('Invalid trace capture settings')
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Laboratory tracing currently supports CPU and CUDA/ROCm')
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=False)
        self.case, self.snapshots = case, snapshots
        self.budget, self.every, self.max_events = snapshot_budget_bytes, every, max_events
        self.bytes_saved = self.count = self.dropped = 0
        self._active = False
        self._closed = False
        self.events = []
        (self.path/'snapshots').mkdir()
        case.save(self.path)
        sources = []
        (self.path/'source').mkdir()
        for i, filename in enumerate(source_files):
            path = Path(filename)
            data = path.read_bytes()
            dest = f'source/{i}-{path.name}'
            (self.path/dest).write_bytes(data)
            sources.append(dict(original=str(path), copy=dest, sha256=hashlib.sha256(data).hexdigest()))
        self.metadata = dict(format_version=1, started_utc=datetime.now(timezone.utc).isoformat(),
                             matrix=case.spec, input_sha256=case.sha256, parameters=parameters or {},
                             device=str(self.device), python=platform.python_version(), torch=torch.__version__,
                             numpy=np.__version__, cuda_build=torch.version.cuda, hip_build=torch.version.hip,
                             threads=torch.get_num_threads(), tf32_allowed=torch.backends.cuda.matmul.allow_tf32,
                             snapshots=snapshots, snapshot_budget_bytes=self.budget, checkpoint_every=every,
                             max_events=max_events, sources=sources, status='running')
        if self.device.type == 'cuda':
            self.metadata['gpu'] = torch.cuda.get_device_name(self.device)
        write_json(self.path/'run.json', self.metadata)
        self._file = (self.path/'events.jsonl').open('x')

    def __enter__(self): return self

    def __exit__(self, typ, exc, tb):
        self.close(error=str(exc) if typ else None)
        return False

    def close(self, error=None):
        if self._closed: return
        self._file.close()
        self.metadata.update(status='failed' if error else 'complete', error=error,
                             event_count=self.count, events_dropped=self.dropped,
                             snapshot_bytes_saved=self.bytes_saved,
                             finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(self.path/'run.json', self.metadata)
        self._closed = True

    def wants(self, iteration=None):
        return iteration is None or iteration % self.every == 0

    def _emit(self, event):
        if self._closed: raise RuntimeError('Trace is closed')
        if self.count >= self.max_events:
            self.dropped += 1
            return None
        event = clean(dict(event, event_id=self.count))
        self._file.write(json.dumps(event, allow_nan=False)+'\n')
        self._file.flush()
        self.events.append(event)
        self.count += 1
        return event

    def record(self, name, iteration=None, **scalars):
        if self.wants(iteration):
            return self._emit(dict(kind='observation', name=name, iteration=iteration, scalars=scalars))

    @contextmanager
    def stage(self, name, iteration=None, **details):
        if self._active: raise RuntimeError('Nested measured stages are ambiguous; use non-overlapping stages')
        self._active = True
        start_event = end_event = None
        baseline = None
        synchronize(self.device)
        if self.device.type == 'cuda':
            baseline = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            with torch.cuda.device(self.device):
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_event.record()
        started = time.perf_counter()
        status = 'ok'
        try:
            yield
        except BaseException:
            status = 'error'
            raise
        finally:
            if end_event:
                with torch.cuda.device(self.device): end_event.record()
            synchronize(self.device)
            wall = (time.perf_counter()-started)*1000
            peak = torch.cuda.max_memory_allocated(self.device) if baseline is not None else None
            self._active = False
            self._emit(dict(kind='stage', name=name, iteration=iteration, details=details, status=status,
                            wall_ms=wall, device_event_ms=start_event.elapsed_time(end_event) if start_event else None,
                            allocator_baseline_bytes=baseline, allocator_peak_allocated_bytes=peak,
                            allocator_peak_increment_bytes=peak-baseline if peak is not None else None))

    def checkpoint(self, name, iteration=None, tensors=None, scalars=None, factors=None):
        if self._active: raise RuntimeError('Place checkpoints after measured stages')
        if not self.wants(iteration): return None
        if self.count >= self.max_events:
            self.dropped += 1
            return None
        event = dict(kind='checkpoint', name=name, iteration=iteration, scalars=scalars or {}, tensors={})
        arrays = dict(tensors or {})
        if factors is not None:
            factors = tuple(host(x) for x in factors)
            arrays.update(zip(('U','s','Vh'), factors))
            try:
                event['accuracy'] = accuracy(self.case, factors)
                arrays.update(spectrum_errors(self.case, factors[1]))
            except ValueError as exc:
                # Preserve divergent intermediates so the failure can be investigated.
                event['validation_error'] = str(exc)
        for j, (key, value) in enumerate(arrays.items()):
            a = host(value)
            item = tensor_statistics(a)
            if self.snapshots == 'full' and self.bytes_saved+a.nbytes <= self.budget:
                dest = f'snapshots/{self.count:06d}-{j:03d}.npy'
                np.save(self.path/dest, a, allow_pickle=False)
                self.bytes_saved += a.nbytes
                item.update(snapshot=dest, snapshot_status='saved')
            else:
                item.update(snapshot=None, snapshot_status='budget_exceeded' if self.snapshots=='full' else 'stats_only')
            event['tensors'][key] = item
        return self._emit(event)


class Run:
    """Read a run without its algorithm, GPU or original interpreter session."""
    def __init__(self, directory):
        self.path = Path(directory)
        self.metadata = json.loads((self.path/'run.json').read_text())
        self.events = [json.loads(line) for line in (self.path/'events.jsonl').read_text().splitlines()]

    def select(self, kind=None, name=None, iteration=None):
        return [e for e in self.events if (kind is None or e['kind']==kind)
                and (name is None or e['name']==name) and (iteration is None or e.get('iteration')==iteration)]

    def tensor(self, event_id, name):
        event = next(e for e in self.events if e['event_id']==event_id)
        item = event['tensors'][name]
        if not item['snapshot']:
            raise ValueError(f'Array was not saved: {item["snapshot_status"]}')
        return np.load(self.path/item['snapshot'], allow_pickle=False)

    def dataframe(self, kind=None):
        """Optional pandas convenience; raw event dictionaries need no pandas."""
        import pandas as pd
        return pd.json_normalize(self.select(kind=kind))

    def stage_statistics(self, by_iteration=False):
        """Summarize observed calls, not independent trials of the same operation."""
        from collections import defaultdict
        from .statistics import describe
        groups = defaultdict(list)
        for event in self.select(kind='stage'):
            if event['status']=='ok':
                key=(event['name'],event.get('iteration') if by_iteration else None)
                groups[key].append(event['wall_ms'])
        return [dict(name=name,iteration=iteration,wall_ms=describe(samples),
                     total_wall_ms=sum(samples),note='Inspection-stage observations; calls may have different inputs.')
                for (name,iteration),samples in groups.items()]

    def plot(self, destination=None):
        from .plots import plot_run
        return plot_run(self, destination)
