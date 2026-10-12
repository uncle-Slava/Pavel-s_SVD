"""Synchronized wall timing, CUDA/HIP events, explicit transfers and validation."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import warnings

import numpy as np
import scipy
import torch
from threadpoolctl import threadpool_limits, threadpool_info

from .algorithms import Unsupported, prepare
from .matrices import input_for
from .metrics import reference, validate


def command(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"unavailable": str(e)}


def inventory(device):
    packages = {}
    for name in ("nvidia-cusolver-cu12", "nvidia-cusolver-cu13", "torch-npu"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    sources = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        if path.name.startswith("."):
            continue
        sources.update(str(path.relative_to(Path(__file__).parent)).encode())
        sources.update(path.read_bytes())
    info = {
        "utc": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
        "cuda_build": torch.version.cuda, "hip_build": torch.version.hip,
        "packages": packages, "device": str(device), "source_sha256": sources.hexdigest(),
        "torch_threads": torch.get_num_threads(), "blas": threadpool_info(),
        "torch_build": torch.__config__.show(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_allowed": torch.backends.cuda.matmul.allow_tf32,
        "nvidia_smi": command(["nvidia-smi"]),
        "lscpu": command(["lscpu"]), "free": command(["free", "-b"]),
    }
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info["accelerator"] = {"name": p.name, "total_memory_bytes": p.total_memory,
                               "compute_capability": [p.major, p.minor]}
    elif device.type == "npu":
        info["npu_smi"] = command(["npu-smi", "info"])
    return info


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        torch.npu.synchronize(device)


def to_host(result):
    return tuple(None if x is None else x.detach().cpu().resolve_conj().resolve_neg().numpy().copy() for x in result)


def summary(samples):
    return {"samples_ms": samples, "median_ms": float(np.median(samples)),
            "p10_ms": float(np.percentile(samples, 10)),
            "p90_ms": float(np.percentile(samples, 90))}


def measure(fn, device, repeat, events=False):
    wall, dev = [], []
    for _ in range(repeat):
        start = end = None
        if events and device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        synchronize(device)
        t = time.perf_counter()
        if start:
            start.record()
        result = fn()
        if end:
            end.record()
        synchronize(device)
        wall.append((time.perf_counter() - t) * 1e3)
        if start:
            dev.append(start.elapsed_time(end))
        # Do not retain prior outputs across samples (avoids inflated peak memory).
        del result
    return {"wall": summary(wall), "device_event": summary(dev) if dev else None}


def run_one(a_cpu, case, spec, device, args, refs):
    row = {"type": "measurement", "case": case, "algorithm": spec,
           "device": str(device), "status": "error"}
    a = fn = result = None
    try:
        with warnings.catch_warnings(record=True) as caught, torch.inference_mode():
            warnings.simplefilter("always")
            synchronize(device)
            t = time.perf_counter()
            a = torch.from_numpy(a_cpu).to(device=device, copy=True)
            synchronize(device)
            row["initial_input_copy_ms"] = (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
            fn = prepare(a, spec, case.get("seed", 20261011) + spec.get("seed_offset", 0))
            if hasattr(fn, "metadata"):
                row["adapter"] = fn.metadata
            synchronize(device)
            row["prepare_ms"] = (time.perf_counter() - t) * 1e3
            for _ in range(args.warmup):
                result = fn()
                synchronize(device)
                del result
                result = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                baseline = torch.cuda.memory_allocated(device)
                reserved_baseline = torch.cuda.memory_reserved(device)
            row["resident_solver"] = measure(fn, device, args.repeat, events=True)
            if device.type == "cuda":
                row["memory"] = {
                    "input_bytes": a_cpu.nbytes,
                    "allocator_baseline_bytes": baseline,
                    "allocator_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "allocator_peak_increment_bytes": torch.cuda.max_memory_allocated(device) - baseline,
                    "allocator_reserved_baseline_bytes": reserved_baseline,
                    "allocator_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "native_workspace_bytes": None,
                    "note": "PyTorch allocator only; native allocations and other processes are excluded. Warm caches retained.",
                }
            else:
                row["memory"] = {"input_bytes": a_cpu.nbytes, "native_workspace_bytes": None}
            result = fn()
            synchronize(device)
            if hasattr(fn, "diagnostics"):
                row["native_diagnostics"] = fn.diagnostics
            t = time.perf_counter()
            host_result = to_host(result)
            synchronize(device)
            row["output_copy_ms"] = (time.perf_counter() - t) * 1e3
            row["memory"]["output_bytes"] = sum(x.nbytes for x in host_result if x is not None)
            row["metrics"] = validate(a_cpu, host_result, refs)
            del result
            result = None
            if args.end_to_end:
                def end_to_end():
                    x = torch.from_numpy(a_cpu).to(device=device, copy=True)
                    solve = prepare(x, spec, case.get("seed", 20261011) + spec.get("seed_offset", 0))
                    return to_host(solve())
                row["end_to_end"] = measure(end_to_end, device, args.repeat)
            row["warnings"] = sorted({str(w.message) for w in caught})
            row["status"] = "ok"
            # For NPU, output placement alone cannot rule out an internal CPU fallback.
            if device.type == "npu":
                row["placement_verified"] = False
                row["note"] = "Experimental: an msprof trace is required before treating this as NPU execution."
    except Unsupported as e:
        row.update(status="unsupported", error=str(e))
    except Exception as e:
        row.update(status="error", error=f"{type(e).__name__}: {e}")
    finally:
        del result, fn, a
        gc.collect()
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/smoke.json")
    p.add_argument("--device", default="cpu", help="cpu, cuda:0 (CUDA or ROCm), or experimental npu:0")
    p.add_argument("--output", default="results/run.jsonl")
    p.add_argument("--corpus", help="Read/write .npy corpus; copy it between hosts for byte-identical inputs")
    p.add_argument("--generate-only", action="store_true")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--end-to-end", action="store_true")
    p.add_argument("--allow-unverified-npu", action="store_true")
    p.add_argument("--only", nargs="*", help="Algorithm names to select from the config")
    args = p.parse_args()
    if args.repeat < 1 or args.warmup < 0 or args.threads < 1:
        p.error("repeat/threads must be positive; warmup nonnegative")
    if args.device.startswith("npu"):
        if not args.allow_unverified_npu:
            p.error("NPU requires --allow-unverified-npu and subsequent profiler placement verification")
        import torch_npu  # noqa: F401
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda", "npu"):
        p.error("Only CPU, CUDA/ROCm, and experimental NPU are supported")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    elif device.type == "npu":
        torch.npu.set_device(device)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads(Path(args.config).read_text())
    names = [c["name"] for c in config["cases"]]
    if len(names) != len(set(names)) or any(Path(n).name != n for n in names):
        p.error("Case names must be unique filenames")
    with threadpool_limits(args.threads), torch.inference_mode():
        if args.generate_only:
            if not args.corpus:
                p.error("--generate-only requires --corpus")
            for case in config["cases"]:
                input_for(case, args.corpus)
            return
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create prevents accidental overwriting of research results.
        with path.open("x") as f:
            meta = {"type": "metadata", "environment": inventory(device), "arguments": vars(args),
                    "config": config, "timing_contract": "Resident input; host dispatch + synchronization included in wall time. End-to-end includes copies and preparation; input generation and validation excluded."}
            f.write(json.dumps(meta, allow_nan=False) + "\n")
            f.flush()
            for case in config["cases"]:
                a_cpu, digest = input_for(case, args.corpus)
                refs = reference(a_cpu)
                methods = case.get("algorithms", config["algorithms"])
                for spec in methods:
                    if args.only and spec["name"] not in args.only:
                        continue
                    if spec.get("rank", 1) > min(case["m"], case["n"]):
                        continue
                    row = run_one(a_cpu, case, spec, device, args, refs)
                    row["input_sha256"] = digest
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                    f.flush()
                    wall = row.get("resident_solver", {}).get("wall", {}).get("median_ms")
                    print(case["name"], spec, row["status"], wall if wall is not None else row.get("error"), flush=True)
            f.write(json.dumps({"type": "environment_after", "nvidia_smi": command(["nvidia-smi"])}) + "\n")
