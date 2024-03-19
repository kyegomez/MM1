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
x = torch.randint(0, 100, (1, 512))
img = torch.randn(1, 3, 224, 224)

# Create a model
model = MM1(
    dim=512,
    depth=12,
    heads=8,
    dim_head=64,
    dropout=0.1,
    num_experts=4,
    num_experts_per_tok=2,
)


# Forward
out = model(x, img)
print(out.shape)  # torch.Size([2, 3, 512])
```

### `CAbstractor`

```python
import torch 
from mm1_torch.main import CAbstractor

# Tensors
x = torch.randn(1, 3, 224, 224)

# Create a model
model = CAbstractor(
    dim=512,
    depth=12,
    heads=8,
)


# Forward
out = model(x)
print(out.shape)  # torch.Size([2, 3, 512])


```


# License
MIT
