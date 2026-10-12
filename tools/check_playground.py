"""Exercise a running playground through its public API and save full experiment records."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8765')
    parser.add_argument('--device',default='cuda:0',choices=['cpu','cuda:0'])
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():parser.error('Choose a new output filename; existing measurements are preserved')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    token=''
    def request(path,body=None):
        data=None if body is None else json.dumps(body).encode()
        headers={'Content-Type':'application/json','X-Playground-Token':token}
        with opener.open(urllib.request.Request(args.url+path,data=data,headers=headers),timeout=30) as response:
            return json.load(response)
    info=request('/api/info');token=info['token']
    methods=[a for a in info['algorithms'] if a!='sparse_lanczos' and (args.device!='cpu' or a=='vendor_default' or not a.startswith('vendor_'))]
    cases=[
        ('all-methods',dict(m=20,n=12,dtype='float64',family='gaussian',output='reduced',algorithms=methods)),
        ('sparse-complex-batch',dict(m=20,n=12,batch=2,dtype='complex128',family='sparse',density=.25,output='topk',target_rank=4,steps=12,algorithms=['sparse_lanczos','lanczos','vendor_default'])),
        ('size-sweep',dict(m=64,n=64,family='low_rank',matrix_rank=4,tail=1e-5,output='topk',target_rank=4,sweep_sizes=[16,32,64],algorithms=['jacobi','randomized','vendor_default'])),
        ('clusters',dict(m=64,n=48,dtype='float64',family='clustered',cluster_size=6,cluster_spread=0,condition=1e5,output='reduced',algorithms=['jacobi','bidiag_qr','gram_jacobi','vendor_default'])),
        ('full-wide-batch',dict(m=12,n=20,batch=2,dtype='complex64',family='gaussian',output='full',algorithms=['jacobi','vendor_default'])),
        ('values-only',dict(m=32,n=20,dtype='float64',family='geometric',condition=1e5,output='values',algorithms=['jacobi','vendor_default'])),
    ]
    records=[]
    output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create also prevents a second client from overwriting this run.
    with output.open('x') as saved:
        for label,case in cases:
            cfg={**case,'device':args.device,'seed':20261012,'repeat':3,'warmup':1,'timeout':120}
            state=request('/api/jobs',cfg)
            deadline=time.monotonic()+120*(state['total']+len(state['configs']))+30
            while state['status'] in ('queued','running'):
                if time.monotonic()>deadline:
                    request('/api/jobs/'+state['id']+'/cancel',{})
                    raise RuntimeError('Experiment exceeded client deadline')
                time.sleep(.5)
                state=request('/api/jobs/'+state['id'])
            records.append({'label':label,'experiment':state})
            saved.seek(0);json.dump(records,saved,allow_nan=False);saved.truncate();saved.flush()
            failures=[{'id':r['id'],'status':r['status'],'assessment':r['assessment']} for r in state['rows'] if r['status']!='ok' or not r['assessment']['passed']]
            print(label,state['status'],f'{len(state["rows"])} rows',json.dumps(failures),flush=True)
            if state['status']!='complete':raise RuntimeError(state['message'])
    completed=sum(len(r['experiment']['rows']) for r in records)
    print(f'Saved {completed} solver comparisons to {output}',flush=True)


if __name__=='__main__':main()
