import torch
from mm1_torch.main import CAbstractor


def run_model(x):
    """
    Runs the CAbstractor model on the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_size).
    """
    # Create a model
    model = CAbstractor(
        dim=512,
        depth=12,
        heads=8,
    )

    # Forward
    out = model(x)
    return out


# Tensors
x = torch.randn(1, 100, 512)

# Run the model
output = run_model(x)
print(output.shape)
