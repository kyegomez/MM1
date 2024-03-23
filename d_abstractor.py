import torch
from mm1_torch.main import DAbstractor

# Text tensor with shape (batch_size, seq_len, dimension)
text = torch.randn(2, 10, 768)

# Initialize the abstractor
abstractor = DAbstractor(
    dim=768,
    num_heads=8,
    heads=8,
    depth=8,
)

# Forward pass
out = abstractor(text)

# Output shape: (batch_size, seq_len, dimension)
print(out)  # torch.Size([2, 10, 768])
