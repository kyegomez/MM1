import torch 
from torch import nn
from mm1_torch.main import DAbstractor

# Random tensors for img
img = torch.randn(1, 3, 224, 224)

# Define the model
model = DAbstractor(
    dim = 64,
    depth = 3,
    heads = 4,
    dropout = 0.1   
)

# Forward
out = model(img)
print(out)  # torch.Size([1, 3, 64])
