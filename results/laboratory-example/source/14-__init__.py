"""Independent SVD implementations over tensor arithmetic and matrix products.

No vendor factorization is called. PyTorch supplies arrays, reductions and GEMM.
This is an executable research specification, not a replacement for tuned kernels.
"""
from .solvers import ALGORITHMS, svd, factory

__all__ = ["ALGORITHMS", "svd", "factory"]
