"""The same checks from Python: several methods, your own matrices, results to a file.

    python examples/from_python.py
"""
import torch
from svdlab import MatrixCase, compare, recipes, save, summary
from my_svd import svd as my_svd                  # examples/my_svd.py

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# A matrix is a dict of MatrixCase.create arguments (family: gaussian, low_rank, clustered,
# geometric, sparse, zero) or a MatrixCase: from_spectrum([...]) or load('matrix.npy').
cases = [
    dict(name='tall_64x40', m=64, n=40, family='gaussian'),
    dict(name='low_rank_k5', m=120, n=80, family='low_rank', rank=5, tail=1e-8, output_rank=5),
    MatrixCase.from_spectrum([5, 5, 1, 1e-3, 1e-9, 0], m=10, n=8, seed=1),   # repeated and zero values
]

methods = {
    'my_svd': my_svd,                             # a function
    'golub_kahan': recipes.golub_kahan,           # a recipe from svdlab
    'jacobi': 'jacobi',                           # a builtin name
    'torch': 'torch',
}
rows = compare(methods, cases, device=device, repeat=3)

print()
for s in summary(rows):
    print(f"{s['method']:<12} {s['ok']}/{s['cases']} ok   worst recon {s['worst_reconstruction']:.1e}   "
          f"{s['total_median_ms']:.2f} ms")
print('saved', save(rows, 'laboratory_runs/from_python.csv'))   # laboratory_runs/ is not tracked by git
