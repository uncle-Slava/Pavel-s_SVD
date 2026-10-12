"""Optional coarse NVIDIA telemetry during a separate sustained, untraced load."""
import csv
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import torch
from svdbench.cli import synchronize


def summarize_power(rows, start, end):
    summaries=[]
    for uuid in sorted({r['uuid'] for r in rows}):
        samples=[r for r in rows if r['uuid']==uuid and start<=r['monotonic_s']<=end and r['power_w'] is not None]
        samples.sort(key=lambda r:r['monotonic_s'])
        duration=samples[-1]['monotonic_s']-samples[0]['monotonic_s'] if len(samples)>=2 else 0.
        energy=sum((b['monotonic_s']-a['monotonic_s'])*(a['power_w']+b['power_w'])/2 for a,b in zip(samples,samples[1:]))
        summaries.append(dict(uuid=uuid,samples=len(samples),covered_seconds=duration,
                              gross_energy_joules=energy if duration else None,
                              average_power_watts=energy/duration if duration else None))
    return summaries


def measure_power(fn, device, seconds, destination):
    device=torch.device(device)
    if device.type!='cuda' or torch.version.hip or seconds<=0:
        raise ValueError('Use NVIDIA CUDA with positive load seconds')
    path=Path(destination)
    rows=[];ready=threading.Event()
    # Physical UUIDs avoid silently confusing CUDA_VISIBLE_DEVICES with SMI indices.
    command=['nvidia-smi','--query-gpu=uuid,power.draw,utilization.gpu,memory.used',
             '--format=csv,noheader,nounits','--loop-ms=200']
    with path.open('x') as log:
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
        def collect():
            for line in proc.stdout:
                fields=next(csv.reader([line]))
                if len(fields)!=4:continue
                def number(value):
                    try:return float(value.strip())
                    except ValueError:return None
                row=dict(monotonic_s=time.monotonic(),uuid=fields[0].strip(),power_w=number(fields[1]),
                         utilization_percent=number(fields[2]),memory_used_mib=number(fields[3]))
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush();ready.set()
        thread=threading.Thread(target=collect,daemon=True);thread.start()
        try:
            if not ready.wait(5):raise RuntimeError('NVIDIA telemetry produced no samples')
            synchronize(device);start=time.monotonic();calls=0
            with torch.inference_mode():
                while time.monotonic()-start<seconds:
                    answer=fn();synchronize(device);del answer;calls+=1
            end=time.monotonic()
        finally:
            proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            thread.join(timeout=5)
            proc.stdout.close()
    return dict(load_seconds=end-start,solver_calls=calls,device=str(device),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                physical_gpus=summarize_power(rows,start,end),
                note='Coarse whole-device sensor integration over covered samples only; no idle subtraction or per-iteration/per-call energy attribution. All physical GPUs are recorded. Other processes and sampling overhead may contribute.')
