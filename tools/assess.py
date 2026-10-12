"""Apply explicit application tolerances to saved results; never rank invalid rows."""
import argparse
import json
from pathlib import Path


def assess(row, limits):
    if row.get("status") != "ok":
        return [row.get("error", "execution did not succeed")]
    worst = row.get("metrics", {}).get("worst", {})
    reasons = []
    if not worst.get("singular_values_nonnegative_sorted", False):
        reasons.append("singular values are not nonnegative and sorted")
    for key, limit in limits.items():
        value = worst.get(key)
        if value is None:
            # values-only outputs have no reconstruction or basis checks.
            reasons.append(f"{key}: not assessed / unavailable")
        elif value > limit:
            reasons.append(f"{key}: {value:.6g} > {limit:.6g}")
    return reasons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--max-residual", type=float, required=True)
    p.add_argument("--max-orthogonality", type=float, required=True)
    p.add_argument("--max-spectrum-over-sigma1", type=float, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    limits = {
        "reconstruction_relative_fro": args.max_residual,
        "orthogonality_u_fro_over_sqrt_cols": args.max_orthogonality,
        "orthogonality_v_fro_over_sqrt_rows": args.max_orthogonality,
        "spectrum_error_over_sigma1": args.max_spectrum_over_sigma1,
    }
    if any(v < 0 for v in limits.values()):
        p.error("Tolerances must be nonnegative")
    rows = []
    for filename in args.files:
        for line in Path(filename).read_text().splitlines():
            row = json.loads(line)
            if row.get("type") != "measurement":
                continue
            failures = assess(row, limits)
            rows.append({"run": filename, "case": row["case"], "algorithm": row["algorithm"],
                         "passed_supplied_limits": not failures, "reasons": failures})
    Path(args.output).write_text(json.dumps({"limits": limits, "assessments": rows}, indent=2))
    print(f"{sum(r['passed_supplied_limits'] for r in rows)}/{len(rows)} passed the supplied limits; see {args.output}")


if __name__ == "__main__":
    main()
