"""Retry setup failures or tune randomized controls on exact cached benchmark inputs.

All original attempts remain untouched. Method requests are JSON objects containing
case (name), id, method, and optional iterations/oversample/steps/seed parameters.
"""
import argparse
import json
import os
from pathlib import Path
from svdbench.broad import child,grade,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline');p.add_argument('--requests',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();base=Path(args.baseline).resolve();root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    requests=json.loads(Path(args.requests).read_text());settings=json.loads((base/'config.json').read_text())
    write_json(root/'requests.json',requests)
    with (root/'measurements.jsonl').open('x') as saved:
        for i,request in enumerate(requests):
            source=base/request['case'];path=root/str(i);path.mkdir()
            cfg=json.loads((source/'config.json').read_text())
            cfg['methods']=[{k:v for k,v in request.items() if k!='case'}]
            write_json(path/'config.json',cfg)
            for name in ('matrix.npy','reference.npz','input.json'):os.link(source/name,path/name)
            status,elapsed=child('solve',path,0,settings['worker_timeout_s'])
            result=path/'result-0.json'
            if status=='ok' and result.exists():row=json.loads(result.read_text())
            else:
                row=dict(type='measurement',case=cfg['case'],id=request['id'],algorithm={**cfg['methods'][0],'rank':cfg['target_rank']},status=status,
                         input_sha256=json.loads((path/'input.json').read_text())['input_sha256'],error=f'Supplement worker {status}',
                         worker_log=(path/'solve-0.log').read_text(errors='replace')[-4000:])
                row['assessment']=grade(row,cfg)
            row.update(worker_elapsed_s=elapsed,supplement=True)
            saved.write(json.dumps(row,allow_nan=False)+'\n');saved.flush()
            print(request['case'],request['id'],row['status'],row['assessment']['passed'],row.get('resident_solver',{}).get('wall',{}).get('median_ms'),flush=True)
    write_json(root/'complete.json',dict(comparisons=len(requests)))


if __name__=='__main__':main()
