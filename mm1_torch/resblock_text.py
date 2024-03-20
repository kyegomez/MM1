from torch import nn, Tensor


class TextResBlock1d(nn.Module):
    """
    1D Residual Block for Text Processing.

    Args:
        channels (int): Number of input and output channels.

    Attributes:
        conv1 (nn.Conv1d): 1D convolutional layer with kernel size 3 and padding 1.
        bn1 (nn.BatchNorm1d): Batch normalization layer.
        conv2 (nn.Conv1d): 1D convolutional layer with kernel size 3 and padding 1.
        bn2 (nn.BatchNorm1d): Batch normalization layer.

    """

    def __init__(
        self,
        channels: int,
    ):
        super(TextResBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the 1D Residual Block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.

        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = nn.ReLU()(out)
        out += residual
        out = nn.ReLU()(out)
        return out
