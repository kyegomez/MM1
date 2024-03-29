import torch
from mm1_torch.main import CAbstractor

# Tensors
x = torch.randn(1, 100, 512)

# Create a model
model = CAbstractor(
    dim=512,
    depth=12,
    heads=8,
)


# Forward
out = model(x)
print(out.shape)
