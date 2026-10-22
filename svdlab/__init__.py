"""svdlab: write an SVD (or one step of one), run it, see its accuracy and speed.

    def svd(a, rank):                      # a: torch tensor (m, n)
        return U, s, Vh                    # (m, k), (k,), (k, n)

    python -m svdlab check my_svd.py [--device cuda] [--compare torch]

or from Python:

    from svdlab import check, compare, table
    rows = check(svd, device='cuda')

`svdlab.blocks` has the steps (QR, bidiagonalization, bidiagonal SVD, Jacobi, ...),
`svdlab.recipes` full SVDs whose steps you can replace, `svdlab.native` builds a C++/CUDA file.
"""
from . import blocks, recipes
from .scorecard import (DEFAULT_SUITE, QUICK_SUITE, SUITES, check, compare, gpu_check, load_file, save,
                    summary, table)
from svdbench.laboratory import MatrixCase

__all__ = ['blocks', 'recipes', 'check', 'compare', 'gpu_check', 'table', 'summary', 'save', 'load_file',
           'MatrixCase', 'DEFAULT_SUITE', 'QUICK_SUITE', 'SUITES']
