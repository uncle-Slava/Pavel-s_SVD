"""One-call launch/dispatch diagnostic; never use profiled timings for rankings."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from svdbench.standalone import svd

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',required=True)
args=p.parse_args()
torch.set_num_threads(4)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32=False
a=torch.randn(32,32,device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(31))
rows=[]
for name in ('jacobi','bidiag_qr','polar_newton_schulz','vendor_gesvdj'):
    def fn():
        return torch.linalg.svd(a,full_matrices=False,driver='gesvdj') if name.startswith('vendor') else svd(a,name)
    try:
        fn();torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn();torch.cuda.synchronize()
        events=prof.events()
        gpu=[e for e in events if str(e.device_type).endswith('CUDA')]
        names=Counter(e.name for e in gpu)
        rows.append({'algorithm':name,'status':'ok','cuda_activity_events':len(gpu),
                     'cuda_compute_events':sum(v for k,v in names.items() if not k.lower().startswith(('memcpy','memset'))),
                     'host_events':len(events)-len(gpu),'top_cuda_activities':names.most_common(12),
                     'note':'Profiled single call, 32x32 FP32, includes allocation, setup, synchronization; event counts are not timings.'})
    except Exception as e:
        rows.append({'algorithm':name,'status':'unavailable','error':f'{type(e).__name__}: {e}'})
Path(args.output).write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
