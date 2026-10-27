import torch
from svdbench.standalone import svd

a = torch.randn(15,15, device='cpu', dtype=torch.float32)

U, s, Vh = svd(a, 'polar_cans')
u, S, vh = svd(a, 'polar_newton_schulz')
print(U)
print(s)
print(Vh)
print('=================================================')
print(u)
print(S)
print(vh)