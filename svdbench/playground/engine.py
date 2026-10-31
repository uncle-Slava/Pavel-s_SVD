"""Validated experiment specifications, plots, and benchmark adapters."""
import math
import numpy as np
import torch
from svdbench.standalone import ALGORITHMS, svd
from svdbench.algorithms import Unsupported

EXTRA = ('sparse_lanczos', 'vendor_default', 'vendor_gesvd', 'vendor_gesvdj', 'vendor_gesvda')
FAMILIES = ('gaussian', 'low_rank', 'geometric', 'clustered', 'sparse', 'zero')
DTYPES = ('float32', 'float64', 'complex64', 'complex128')


def normalize(raw):
    if not isinstance(raw, dict):
        raise ValueError('Experiment must be a JSON object')
    def number(key, default, low, high, integer=False):
        val = raw.get(key, default)
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise ValueError(f'{key} must be a finite number')
        if not low <= val <= high or (integer and int(val) != val):
            raise ValueError(f'{key} must be {"an integer " if integer else ""}between {low} and {high}')
        return int(val) if integer else float(val)
    m, n = number('m',64,1,4096,True), number('n',32,1,4096,True)
    batch = number('batch',1,1,64,True)
    if min(m,n) > 512 or batch*m*n > 4_194_304:
        raise ValueError('Interactive limit: min(m,n) <= 512 and at most 4,194,304 total matrix elements. Use the CLI for larger sweeps.')
    dtype, family = raw.get('dtype','float32'), raw.get('family','gaussian')
    if dtype not in DTYPES or family not in FAMILIES:
        raise ValueError('Unknown matrix family or dtype')
    algs = raw.get('algorithms',['jacobi','randomized','vendor_default'])
    if not isinstance(algs,list) or not algs or len(algs)>21 or any(a not in (*ALGORITHMS,*EXTRA) for a in algs) or len(set(algs))!=len(algs):
        raise ValueError('Select distinct algorithms from the supplied list')
    output = raw.get('output','topk')
    if output not in ('reduced','full','topk','values'):
        raise ValueError('Unknown output mode')
    device = raw.get('device','cpu')
    if device not in ('cpu','cuda:0'):
        raise ValueError('Device must be cpu or cuda:0')
    if device=='cuda:0' and not torch.cuda.is_available():
        raise ValueError('CUDA/ROCm is unavailable on this server; choose CPU or connect to a GPU server')
    if output=='full' and batch*(m*m+n*n)*np.dtype(dtype).itemsize > 128*2**20:
        raise ValueError('Full output bases exceed the 128 MiB interactive output limit; choose reduced or top-k')
    r=min(m,n)
    case=dict(name='playground',m=m,n=n,batch=batch,dtype=dtype,family=family,
              rank=number('matrix_rank',min(8,r),1,r,True),tail=number('tail',1e-5,0,1),
              condition=number('condition',1e3,1,1e16),density=number('density',0.05,0,1),
              cluster_size=number('cluster_size',4,1,512,True),cluster_spread=number('cluster_spread',0.001,0,0.9),
              scale=number('scale',1,1e-12,1e12),seed=number('seed',20261012,0,2**32-1,True))
    return dict(case=case,algorithms=algs,device=device,output=output,
                rank=number('target_rank',min(8,r),1,r,True),
                oversample=number('oversample',8,0,128,True),iterations=number('iterations',2,0,8,True),
                max_sweeps=number('max_sweeps',60,1,200,True),
                steps=number('steps',min(r,32),1,r,True),
                repeat=number('repeat',3,1,10,True),warmup=number('warmup',1,0,3,True),
                threads=number('threads',4,1,16,True),timeout=number('timeout',120,5,600),
                tolerance=number('tolerance',1e-4,1e-14,0.5))


def request_spec(cfg):
    c=cfg['case']
    return {**{k:v for k,v in c.items() if k not in ('name','rank')},
            'matrix_rank':c['rank'], **{k:v for k,v in cfg.items() if k not in ('case','rank')},
            'target_rank':cfg['rank']}


def plot_data(a, s, u=None, vh=None):
    x=a if a.ndim==2 else a[0]
    spectra=s if s.ndim==1 else s[0]
    rows=np.linspace(0,x.shape[0]-1,min(64,x.shape[0]),dtype=int)
    cols=np.linspace(0,x.shape[1]-1,min(64,x.shape[1]),dtype=int)
    sample=x[np.ix_(rows,cols)]
    shown=abs(sample) if np.iscomplexobj(x) else sample
    limit=float(abs(shown).max()) or 1.
    squares=np.square(spectra.astype(float))
    energy=np.cumsum(squares)/(squares.sum() or 1.)
    result={'matrix':shown.tolist(),'sample_rows':rows.tolist(),'sample_cols':cols.tolist(),
            'complex_magnitude':bool(np.iscomplexobj(x)),'color_limit':limit,
            'singular_values':spectra.tolist(),'energy':energy.tolist(),
            'actual_density':float(np.count_nonzero(a)/a.size),'input_bytes':a.nbytes}
    if u is not None:
        uu=u if u.ndim==2 else u[0]; vv=vh if vh.ndim==2 else vh[0]
        k=len(spectra)
        rec=(uu[rows,:k]*spectra)@vv[:k,cols]
        result['reconstruction']=(abs(rec) if np.iscomplexobj(rec) else rec).tolist()
        result['error']=abs(sample-rec).tolist()
    return result


def assess(row, cfg):
    if row.get('status')!='ok':
        return {'passed':False,'reasons':[row.get('error','Solver failed')],'partial':cfg['output']=='topk'}
    w=row['metrics']['worst']; tol=cfg['tolerance']; checks={}
    if cfg['output']=='values':
        checks['spectrum_error_over_sigma1']=tol
    else:
        checks.update(orthogonality_u_fro_over_sqrt_cols=tol,orthogonality_v_fro_over_sqrt_rows=tol)
        partial=cfg['output']=='topk' and cfg['rank']<min(cfg['case']['m'],cfg['case']['n'])
        if partial:
            if w.get('residual_over_optimal_tail') is not None:
                checks['residual_over_optimal_tail']=1.05
            else:
                checks['reconstruction_relative_fro']=tol
        else:
            checks.update(reconstruction_relative_fro=tol,spectrum_error_over_sigma1=tol)
    reasons=[f'{key}: {w.get(key)} exceeds {limit:g}' for key,limit in checks.items() if w.get(key) is None or w[key]>limit]
    if not w.get('singular_values_nonnegative_sorted'):
        reasons.append('Singular values are not sorted and nonnegative')
    return {'passed':not reasons,'reasons':reasons,'limits':checks,
            'partial':cfg['output']=='topk','note':'Top-k gate checks tail ratio and orthogonality, not individual singular-value accuracy.'}


def factory(a, spec):
    name=spec['algorithm']; output=spec['output']; rank=spec['rank']
    opts={k:spec[k] for k in ('oversample','iterations','max_sweeps','seed')}
    opts['steps']=max(rank,spec['steps'])
    if name=='sparse_lanczos':
        if spec['family']!='sparse':
            raise Unsupported('Sparse Lanczos requires the sparse matrix family')
        from svdbench.standalone.iterative import lanczos_svd
        from svdbench.standalone.primitives import complete_basis
        mats=a[None] if a.ndim==2 else a
        class SparseOperator:
            def __init__(self,x):
                self.matrix=x.to_sparse().coalesce()
                self.shape,self.dtype,self.device=x.shape,x.dtype,x.device
            def matmat(self,x): return torch.sparse.mm(self.matrix,x)
            def rmatmat(self,x): return torch.sparse.mm(self.matrix.mH,x)
        ops=[SparseOperator(x) for x in mats]
        def run():
            answers=[lanczos_svd(op,rank,steps=opts['steps'],seed=opts['seed']) for op in ops]
            u,s,v=(torch.stack([r[j] for r in answers]) for j in range(3))
            if output=='full':
                u=complete_basis(u,a.shape[-2]);v=complete_basis(v,a.shape[-1])
            if a.ndim==2: u,s,v=u[0],s[0],v[0]
            run.diagnostics={'iterations':[r[3] for r in answers],'sparse_storage':True}
            return (None,s,None) if output=='values' else (u,s,v.mH)
        run.metadata={'implementation':'standalone sparse callback Lanczos','sparse_storage':True,
                      'sparse_bytes':sum(op.matrix.values().numel()*op.matrix.values().element_size()+op.matrix.indices().numel()*8 for op in ops),
                      'note':'Sparse conversion in preparation; dense corpus/reference retained outside solver.'}
        return run
    if name.startswith('vendor_'):
        driver=name.removeprefix('vendor_')
        if driver!='default' and (a.device.type!='cuda' or torch.version.hip):
            raise Unsupported('Explicit cuSOLVER drivers require an NVIDIA GPU')
        kw={} if driver=='default' else {'driver':driver}
        def run():
            if output=='values': return None,torch.linalg.svdvals(a,**kw),None
            u,s,vh=torch.linalg.svd(a,full_matrices=output=='full',**kw)
            if output=='topk': u,s,vh=u[...,:rank],s[...,:rank],vh[...,:rank,:]
            return u,s,vh
        run.metadata={'implementation':'vendor control','full_solve_then_slice':output=='topk'}
        return run
    def run():
        u,s,vh,info=svd(a,name,rank=rank,full_matrices=output=='full',values_only=output=='values',return_info=True,**opts)
        run.diagnostics=info
        return u,s,vh
    run.metadata={'implementation':'standalone tensor arithmetic','values_only_forms_vectors':True}
    return run


def algorithm_spec(cfg, name):
    rank=cfg['rank'] if cfg['output']=='topk' else min(cfg['case']['m'],cfg['case']['n'])
    return dict(name='svdbench.playground.engine:factory',algorithm=name,output=cfg['output'],
                rank=rank,family=cfg['case']['family'],seed=cfg['case']['seed'],
                **{k:cfg[k] for k in ('oversample','iterations','max_sweeps','steps')})
