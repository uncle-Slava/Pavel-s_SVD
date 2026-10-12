"""Create an accuracy-gated CSV from standalone benchmark rows."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def grade(row):
    if row.get('status') != 'ok':
        return [row.get('error','execution failure')]
    m = row['metrics']['worst']
    partial = row['algorithm'].get('rank',min(row['case']['m'],row['case']['n'])) < min(row['case']['m'],row['case']['n'])
    double = row['case'].get('dtype') in ('float64','complex128')
    tol = 1e-9 if double else 2e-4
    checks = {'orthogonality_u_fro_over_sqrt_cols':tol,
              'orthogonality_v_fro_over_sqrt_rows':tol}
    if partial:
        checks['residual_over_optimal_tail'] = 1.05
    else:
        checks['reconstruction_relative_fro'] = tol
        checks['spectrum_error_over_sigma1'] = tol
    fail = []
    if not m.get('singular_values_nonnegative_sorted'):
        fail.append('unsorted/negative singular values')
    for key, limit in checks.items():
        if m.get(key) is None or m[key] > limit:
            fail.append(f'{key}={m.get(key)} exceeds {limit}')
    return fail


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('files',nargs='+')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    rows=[]
    for fn in args.files:
        rows += [json.loads(x) for x in Path(fn).read_text().splitlines() if x.strip()]
    rows=[r for r in rows if r.get('type')=='measurement']
    out=[]
    for r in rows:
        m=r.get('metrics',{}).get('worst',{})
        t=r.get('resident_solver',{})
        reasons=grade(r)
        out.append(dict(case=r['case']['name'],algorithm=r['algorithm'].get('method',r['algorithm']['name']),
                        seed=r['algorithm'].get('seed',''),rank=r['algorithm'].get('rank','full'),status=r['status'],
                        passed_pilot_budget=not reasons,reasons='; '.join(reasons),
                        wall_ms=t.get('wall',{}).get('median_ms'),
                        event_ms=(t.get('device_event') or {}).get('median_ms'),
                        end_to_end_ms=r.get('end_to_end',{}).get('wall',{}).get('median_ms'),
                        peak_increment_mib=r.get('memory',{}).get('allocator_peak_increment_bytes',0)/2**20,
                        residual=m.get('reconstruction_relative_fro'),orth_u=m.get('orthogonality_u_fro_over_sqrt_cols'),
                        orth_v=m.get('orthogonality_v_fro_over_sqrt_rows'),sigma_over_sigma1=m.get('spectrum_error_over_sigma1'),
                        resolved_sigma_relative=m.get('resolved_spectrum_max_relative_error'),tail_ratio=m.get('residual_over_optimal_tail'),
                        input_sha256=r['input_sha256']))
    target=Path(args.output)
    with target.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    print('Rows:',len(out),'status:',dict(Counter(r['status'] for r in out)),
          'pilot budget passes:',sum(r['passed_pilot_budget'] for r in out))
    for r in out:
        if not r['passed_pilot_budget']:
            print(r['case'],r['algorithm'],r['reasons'])

if __name__=='__main__': main()
