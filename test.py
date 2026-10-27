import torch
from svdbench.standalone import svd

a = torch.randn(256,32, device='cpu', dtype=torch.float32)
U, s, Vh = svd(a, 'polar_cans')
print(U)
print(s)
print(Vh)