"""One bounded child process per preparation or algorithm, controlled by server.py."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from svdbench.cli import run_one
from svdbench.matrices import make_matrix
from svdbench.metrics import reference
from .engine import algorithm_spec, assess, plot_data


def main():
    stage,directory=sys.argv[1:3]
    path=Path(directory);cfg=json.loads((path/'config.json').read_text())
    torch.set_num_threads(cfg['threads'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with threadpool_limits(cfg['threads']),torch.inference_mode():
        if stage=='prepare':
            a=make_matrix(cfg['case'])
            np.save(path/'matrix.npy',a,allow_pickle=False)
            refs=reference(a)
            np.savez(path/'reference.npz',u=np.stack([r[0] for r in refs]),s=np.stack([r[1] for r in refs]),vh=np.stack([r[2] for r in refs]))
            preview=plot_data(a,refs[0][1])
            preview['sha256']=hashlib.sha256(a.tobytes()).hexdigest()
            (path/'preview.json').write_text(json.dumps(preview,allow_nan=False))
        else:
            name=sys.argv[3]
            a=np.load(path/'matrix.npy',allow_pickle=False)
            with np.load(path/'reference.npz',allow_pickle=False) as f:
                refs=list(zip(f['u'],f['s'],f['vh']))
            args=SimpleNamespace(warmup=cfg['warmup'],repeat=cfg['repeat'],end_to_end=False,
                result_callback=lambda result:plot_data(a,result[1],result[0],result[2]))
            row=run_one(a,cfg['case'],algorithm_spec(cfg,name),torch.device(cfg['device']),args,refs)
            row['id']=name;row['assessment']=assess(row,cfg)
            row['input_sha256']=hashlib.sha256(a.tobytes()).hexdigest()
            source=hashlib.sha256()
            package=Path(__file__).resolve().parents[1]
            for file in sorted(package.rglob('*.py')):
                if not file.name.startswith('.'):
                    source.update(str(file.relative_to(package)).encode())
                    source.update(file.read_bytes())
            row['source_sha256']=source.hexdigest()
            (path/f'{name}.json').write_text(json.dumps(row,allow_nan=False))

if __name__=='__main__': main()
