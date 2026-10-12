"""Flatten benchmark JSONL into a compact review table (no ranking by speed)."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output", default="results/summary.csv")
    args = parser.parse_args()
    out = []
    for filename in args.files:
        for line in Path(filename).read_text().splitlines():
            row = json.loads(line)
            if row["type"] != "measurement":
                continue
            metric = row.get("metrics", {}).get("worst", {})
            timing = row.get("resident_solver", {})
            memory = row.get("memory", {})
            out.append({
                "run": Path(filename).stem, "case": row["case"]["name"],
                "algorithm": json.dumps(row["algorithm"], sort_keys=True),
                "status": row["status"], "error": row.get("error", ""),
                "device_event_median_ms": (timing.get("device_event") or {}).get("median_ms"),
                "wall_median_ms": timing.get("wall", {}).get("median_ms"),
                "end_to_end_median_ms": row.get("end_to_end", {}).get("wall", {}).get("median_ms"),
                "peak_allocator_increment_bytes": memory.get("allocator_peak_increment_bytes"),
                "native_workspace_bytes": memory.get("native_workspace_bytes"),
                "residual_relative_fro": metric.get("reconstruction_relative_fro"),
                "residual_over_optimal_tail": metric.get("residual_over_optimal_tail"),
                "spectrum_error_over_sigma1": metric.get("spectrum_error_over_sigma1"),
                "reference_resolved_spectrum_max_relative_error": metric.get("resolved_spectrum_max_relative_error"),
                "orthogonality_u": metric.get("orthogonality_u_fro_over_sqrt_cols"),
                "orthogonality_v": metric.get("orthogonality_v_fro_over_sqrt_rows"),
                "input_sha256": row["input_sha256"],
            })
    if not out:
        parser.error("No measurement rows")
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"Wrote {len(out)} measurements to {target}")


if __name__ == "__main__":
    main()
