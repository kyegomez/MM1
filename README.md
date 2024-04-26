[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# MM1 
PyTorch Implementation of the paper "MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training".

`img -> encoder -> connector -> llm -> tokens` 

## install
`pip3 install mm1-torch`

## usage
```python
import torch
from mm1_torch.main import MM1

# Tensors
x = torch.randint(0, 100, (1, 512))  # Create a random tensor of shape (1, 512)
img = torch.randn(1, 3, 224, 224)  # Create a random image tensor of shape (1, 3, 224, 224)

# Create a model
model = MM1(
    dim=512,  # Dimension of the input tensor
    depth=12,  # Number of transformer layers
    heads=8,  # Number of attention heads
    dim_head=64,  # Dimension of each attention head
    dropout=0.1,  # Dropout rate
    num_experts=4,  # Number of experts in mixture-of-experts
    num_experts_per_tok=2,  # Number of experts per token in mixture-of-experts
    encoder_dim=512,  # Dimension of the encoder output
    encoder_depth=12,  # Number of encoder transformer layers
    encoder_heads=8,  # Number of encoder attention heads
    use_moe=True,  # Whether to use mixture-of-experts
    return_logits=True  # Whether to return logits or probabilities
)

# Forward
out = model(x, img)  # Forward pass through the model
print(out.shape)  # Print the shape of the output tensor (torch.Size([2, 3, 512]))
print(out)  # Print the output tensor
```

### `CAbstractor`

```python
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

```


# License
MIT


## Todo

- [x] Implement the deformable attention
- [ ] Create a training script for Huggingface datasets
- [ ] Create unit tests for every module