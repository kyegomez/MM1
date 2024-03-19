import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.nn import Module
from zeta.nn import img_to_text
from zeta.nn.attention import Attention
from mm1_torch.moe import MoELayer
from zeta.structs import ViTransformerWrapper, Encoder


# constants
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


# small helper modules


class Residual(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


# building block modules


class Block(Module):
    """
    A block module that performs convolution, normalization, activation, and scaling/shift operations.

    Args:
        dim (int): The number of input channels.
        dim_out (int): The number of output channels.
        groups (int, optional): The number of groups to separate the channels into. Defaults to 1.

    Attributes:
        proj (nn.Conv2d): The convolutional layer.
        norm (nn.GroupNorm): The group normalization layer.
        act (nn.SiLU): The activation function.

    Methods:
        forward(x, scale_shift=None): Performs the forward pass of the block module.

    """

    def __init__(self, dim, dim_out, groups=1):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        """
        Performs the forward pass of the block module.

        Args:
            x (torch.Tensor): The input tensor.
            scale_shift (tuple, optional): A tuple containing the scale and shift values for scaling/shift operations. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after passing through the block module.

        """
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(Module):
    """
    Residual block for a ResNet architecture.

    Args:
        dim (int): Input dimension.
        dim_out (int): Output dimension.
        time_emb_dim (int, optional): Dimension of the time embedding. Defaults to None.
        groups (int, optional): Number of groups for grouped convolution. Defaults to 8.
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=1):
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)
            )
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = (
            nn.Conv2d(dim, dim_out, 1)
            if dim != dim_out
            else nn.Identity()
        )

    def forward(self, x, time_emb=None):
        """
        Forward pass of the ResnetBlock.

        Args:
            x (torch.Tensor): Input tensor.
            time_emb (torch.Tensor, optional): Time embedding tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


def posemb_sincos_2d(
    h: int,
    w: int,
    dim: int,
    temperature: int = 10000,
    dtype=torch.float32,
):
    """
    Generates positional embeddings using sine and cosine functions for a 2D grid.

    Args:
        h (int): Height of the grid.
        w (int): Width of the grid.
        dim (int): Feature dimension. Must be a multiple of 4 for sincos embedding.
        temperature (int, optional): Temperature parameter for the embedding. Defaults to 10000.
        dtype (torch.dtype, optional): Data type of the output tensor. Defaults to torch.float32.

    Returns:
        torch.Tensor: Positional embeddings of shape (h * w, dim).

    Raises:
        AssertionError: If the feature dimension is not a multiple of 4.

    Example:
        pe = posemb_sincos_2d(10, 10, 32)
    """
    y, x = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing="ij"
    )
    assert (
        dim % 4
    ) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# D Abstractor
class DAbstractor(nn.Module):
    """
    DAbstractor is a module that performs abstract reasoning on input data.

    Args:
        dim (int): The dimension of the input data.
        depth (int): The depth of the abstract reasoning process.
        heads (int): The number of attention heads.
        dropout (int): The dropout rate.
        mlp_dim (int): The dimension of the MLP layers.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The dimension of the input data.
        depth (int): The depth of the abstract reasoning process.
        heads (int): The number of attention heads.
        dropout (int): The dropout rate.
        mlp_dim (int): The dimension of the MLP layers.
        avg_pool (nn.AdaptiveAvgPool2d): The adaptive average pooling layer.
        attn (MultiQueryAttention): The multi-query attention module.

    Methods:
        forward(x: Tensor) -> Tensor:
            Performs the forward pass of the DAbstractor module.

    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dropout: float = 0.1,
        dim_head: int = 32,
        seq_len: int = 2048,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dropout = dropout
        self.dim_head = dim_head
        self.seq_len = seq_len

        # Positional Embedding
        # TODO: Implement

        # Adaptive pool

        # Attention
        # self.attn = MultiQueryAttention(dim, heads, *args, **kwargs)
        self.attn = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            causal=True,
            qk_norm=True,
        )

        # Deformable Attention
        # TODO: Implement

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Performs the forward pass of the DAbstractor module.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the DAbstractor module.

        """
        b, c, h, w = x.shape

        # Positional Embedding
        position_embeds = posemb_sincos_2d(h, w, self.dim)
        print(position_embeds.shape)

        # Adaptive pool
        # x = self.avg_pool(x)
        x = nn.AdaptiveAvgPool2d((h, w))(x)
        print(x.shape)

        # Reshape to 3d
        x = img_to_text(x, self.seq_len, dim=self.dim, norm=True)
        print(x.shape)

        # Attention
        x = self.attn(x)

        # Deformable Attention
        # TODO : Implement

        return x


class CAbstractor(nn.Module):
    """
    CBastractor is a class that represents a custom module for abstracting features in a neural network.

    Args:
        dim (int): The input dimension of the module.
        depth (int): The number of residual blocks in the module.
        heads (int): The number of attention heads in the module.
        mlp_dim (int): The dimension of the MLP layers in the module.
        dropout (float): The dropout rate to be applied in the module.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads

        # Res Blocks
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.res_block = ResnetBlock(dim, dim, *args, **kwargs)
        self.layers.append(self.res_block)

        # Average pooling

    def forward(self, x: Tensor):
        # B, C, H ,W
        B, C, H, W = x.shape

        # Res Blocks
        for layer in self.layers:
            x = layer(x)

        # Average pooling
        x = nn.AdaptiveAvgPool2d((H, W))(x)

        # ResnetBlocks
        for layer in self.layers:
            x = layer(x)

        return x


# x = torch.rand(1, 3, 224, 224)
# # d_abstractor = DAbstractor(dim=224, depth=3, heads=3, dropout=0.1)

# # print(d_abstractor(x))

# c_abstractor = CBastractor(3, 3, 3)
# print(c_abstractor(x))


class DecoderLLM(nn.Module):
    """
    DecoderLLM is a class that represents the decoder module of a language model.

    Args:
        dim (int): The dimension of the input tensor.
        depth (int): The number of attention and expert layers.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        num_experts (int): The number of experts in the MoE layer.
        dropout (float): The dropout rate.
        num_experts_per_tok (int, optional): The number of experts per token. Defaults to 4.

    Attributes:
        dim (int): The dimension of the input tensor.
        depth (int): The number of attention and expert layers.
        heads (int): The number of attention heads.
        dim_head (int): The dimension of each attention head.
        num_experts (int): The number of experts in the MoE layer.
        dropout (float): The dropout rate.
        num_experts_per_tok (int): The number of experts per token.
        attn_layers (nn.ModuleList): A list of attention layers.
        expert_layers (nn.ModuleList): A list of MoE layers.

    Methods:
        forward(x: Tensor) -> Tensor:
            Performs the forward pass of the decoder module.

    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        num_experts: int,
        dropout: float,
        num_experts_per_tok: int = 4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.num_experts = num_experts
        self.dropout = dropout
        self.num_experts_per_tok = num_experts_per_tok

        # Attention layers
        self.attn_layers = nn.ModuleList(
            [
                Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    causal=True,
                    qk_norm=True,
                    *args,
                    **kwargs,
                )
                for _ in range(self.depth)
            ]
        )

        # Expert layers
        self.expert_layers = nn.ModuleList(
            [
                MoELayer(dim, num_experts, num_experts_per_tok)
                for _ in range(self.depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the decoder module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.

        """
        for attn_layer, expert_layer in zip(
            self.attn_layers, self.expert_layers
        ):
            attn, _ = attn_layer(x)
            attn = attn + x
            expert = expert_layer(x)
            x = attn + expert

        return x


class MM1(nn.Module):
    """
    MM1 class represents the MM1 model architecture.

    Args:
        dim (int): The dimension of the model.
        depth (int): The depth of the model.
        heads (int): The number of attention heads in the model.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        num_experts (int, optional): The number of experts in the model. Defaults to 8.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        num_experts_per_tok (int, optional): The number of experts per token. Defaults to 4.
        image_size (int, optional): The size of the input image. Defaults to 224.
        patch_size (int, optional): The size of each image patch. Defaults to 16.
        encoder_dim (int, optional): The dimension of the encoder. Defaults to 256.
        encoder_depth (int, optional): The depth of the encoder. Defaults to 3.
        encoder_heads (int, optional): The number of attention heads in the encoder. Defaults to 4.
        num_tokens (int, optional): The number of tokens in the embedding layer. Defaults to 20000.

    Attributes:
        dim (int): The dimension of the model.
        depth (int): The depth of the model.
        heads (int): The number of attention heads in the model.
        dim_head (int): The dimension of each attention head.
        num_experts (int): The number of experts in the model.
        dropout (float): The dropout rate.
        num_experts_per_tok (int): The number of experts per token.
        image_size (int): The size of the input image.
        patch_size (int): The size of each image patch.
        encoder (CAbstractor): The encoder module.
        decoder (DecoderLLM): The decoder module.
        vit (ViTransformerWrapper): The vision encoder module.
        c_abstractor (CAbstractor): The C abstractor module.
        embedding (nn.Embedding): The embedding layer.

    Methods:
        forward(text, image): Performs forward pass of the MM1 model.

    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int = 64,
        num_experts: int = 8,
        dropout: float = 0.1,
        num_experts_per_tok: int = 4,
        image_size: int = 224,
        patch_size: int = 16,
        encoder_dim: int = 256,
        encoder_depth: int = 3,
        encoder_heads: int = 4,
        num_tokens: int = 20000,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.num_experts = num_experts
        self.dropout = dropout
        self.num_experts_per_tok = num_experts_per_tok
        self.image_size = image_size
        self.patch_size = patch_size

        # Encoder
        self.encoder = CAbstractor(dim, depth, heads)

        # Decoder
        self.decoder = DecoderLLM(
            dim,
            depth,
            heads,
            dim_head,
            num_experts,
            dropout,
            num_experts_per_tok,
        )

        # Vision Encoder
        self.vit = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            post_emb_norm=True,
            attn_layers=Encoder(
                dim=encoder_dim,
                depth=encoder_depth,
                heads=encoder_heads,
            ),
        )

        # C Abstractor
        self.c_abstractor = CAbstractor(
            dim,
            depth,
            heads,
        )

        # Embed the tokens
        self.embedding = nn.Embedding(num_tokens, dim)

    def forward(self, text: Tensor, image: Tensor):
        """
        Performs forward pass of the MM1 model.

        Args:
            text (Tensor): The input text tensor.
            image (Tensor): The input image tensor.

        Returns:
            Tensor: The output tensor of the model.

        """
        # Embed tokens
        x = self.embedding(text)

        t_b, t_s, t_d = x.shape
        i_b, i_c, i_h, i_w = image.shape

        print(f"Text: {x.shape}")
        print(f"Image: {image.shape}")

        # Pass tokens through decoder
        # print(type(x), x)
        x = self.decoder(x)

        # Vision Encoder
        image = self.vit(image, return_embeddings=True)

        # Connector
        # image = self.c_abstractor(image)
        image = nn.AdaptiveAvgPool1d((t_s, t_d))(image)

        # Decoder
        x = self.decoder(x + image)

        return x
