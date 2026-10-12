"""Run svdbench with coarse NVIDIA telemetry; this is not per-kernel energy."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--telemetry", required=True)
    args, benchmark_args = p.parse_known_args()
    path = Path(args.telemetry)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = "timestamp,index,name,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem"
    with path.open("x") as f:
        sampler = None
        try:
            sampler = subprocess.Popen(["nvidia-smi", f"--query-gpu={fields}", "--format=csv", "--loop-ms=200"], stdout=f, stderr=subprocess.STDOUT)
            result = subprocess.run([sys.executable, "-m", "svdbench", *benchmark_args])
        finally:
            if sampler is not None:
                sampler.terminate()
                sampler.wait(timeout=10)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
