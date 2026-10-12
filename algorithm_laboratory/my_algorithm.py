"""YOUR BENCH: replace any line here. This working seed has zero power iterations.

Run with: python -m algorithm_laboratory.experiment --algorithm mine
Add trace.stage around a computational block, then checkpoint its outputs.
You can put breakpoint() anywhere and examine a, omega, y, q, b, u, s, vh.
"""
import torch
from svdbench.laboratory import NullTrace
from .algorithms import orthogonalize, small_svd, product


def solve(a, trace=None, rank=8, oversample=8, seed=7, primitives='standalone', **options):
    trace = trace if trace is not None else NullTrace()
    ell = min(rank+oversample,min(a.shape))
    gen = torch.Generator(device=a.device).manual_seed(seed)
    with trace.stage('make_sketch'):
        omega = torch.randn(a.shape[1],ell,dtype=a.dtype,device=a.device,generator=gen)

    # START CHANGING THE ALGORITHM HERE.
    with trace.stage('range_finder',iteration=0):
        y = product(a,omega)
        q = orthogonalize(y,primitives)
    trace.checkpoint('range',iteration=0,tensors={'omega':omega,'Y':y,'Q':q})

    ah = a.mH if a.layout==torch.strided else a.mH.coalesce()
    with trace.stage('projected_problem'):
        b = product(ah,q).mH
        u,s,vh = small_svd(b,primitives)
    trace.checkpoint('small_problem',tensors={'B':b,'small_U':u,'small_s':s,'small_Vh':vh})
    with trace.stage('return_to_original_space'):
        answer = q@u[:,:rank],s[:rank],vh[:rank]
    trace.checkpoint('final',factors=answer)
    return answer
