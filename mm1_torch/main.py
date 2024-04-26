import torch
from torch import Tensor, nn
from zeta.nn import FeedForward, OutputHead, threed_to_text
from zeta.nn.attention import Attention
from zeta.structs import Encoder, ViTransformerWrapper

from mm1_torch.resblock_text import TextResBlock1d
from zeta.nn.modules import NormalSparseMoE


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
def posemb_sincos_1d(patches, temperature=10000, dtype=torch.float32):
    """
    Compute positional embeddings using sine and cosine functions for 1D patches.

    Args:
        patches (torch.Tensor): Input patches of shape (batch_size, n, dim).
        temperature (float, optional): Temperature parameter for the positional embeddings. Default is 10000.
        dtype (torch.dtype, optional): Data type of the output tensor. Default is torch.float32.

    Returns:
        torch.Tensor: Positional embeddings of shape (batch_size, n, dim).

    Raises:
        AssertionError: If the feature dimension is not a multiple of 2.

    """
    _, n, dim, device, dtype = (
        *patches.shape,
        patches.device,
        patches.dtype,
    )

    n = torch.arange(n, device=device)
    assert (
        dim % 2
    ) == 0, "feature dimension must be multiple of 2 for sincos emb"
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim=1)
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

        # Attention
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
        # b, c, h, w = x.shape
        b, s, d = x.shape

        # Positional Embedding
        position_tokens = posemb_sincos_1d(x)
        print(f"Positional Tokens: {position_tokens.shape}")

        # Add positional embeddings
        x += position_tokens

        # Adaptive pool
        x = nn.AdaptiveAvgPool1d(d)(x)

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

    def forward(self, x: Tensor):
        # 3d -- B, S, D
        b, s, d = x.shape

        # Res Blocks
        self.layers = nn.ModuleList([])

        for _ in range(self.depth):
            self.res_block = TextResBlock1d(s)
        self.layers.append(self.res_block)

        # Res Blocks
        for layer in self.layers:
            x = layer(x)

        # Average pooling
        x = nn.AdaptiveAvgPool1d(d)(x)

        # ResnetBlocks
        for layer in self.layers:
            x = layer(x)

        return nn.LayerNorm(self.dim)(x)


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
        num_experts_per_tok: int = 1,
        use_feedforward: bool = True,
        use_moe: bool = True,
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
        self.use_feedforward = use_feedforward
        self.use_moe = use_moe

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
                NormalSparseMoE(dim, num_experts, hidden_dim=dim)
                for _ in range(self.depth)
            ]
        )

        # Expert layers
        self.ffn_layers = nn.ModuleList(
            [
                FeedForward(dim, dim, 4, post_act_ln=True, swish=True)
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
        if self.use_feedforward:
            for attn_layer, ffn in zip(
                self.attn_layers, self.ffn_layers
            ):
                attn, _ = attn_layer(x)
                attn = attn + x
                expert = ffn(attn)
                x = attn + expert

        else:
            for attn_layer, expert_layer in zip(
                self.attn_layers, self.expert_layers
            ):
                attn, _ = attn_layer(x)
                attn = attn + x
                expert = expert_layer(attn)
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
        return_logits: bool = True,
        return_embeddings: bool = False,
        use_moe: bool = True,
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
        self.return_logits = return_logits
        self.return_embeddings = return_embeddings

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
            use_moe=use_moe,
            *args,
            **kwargs,
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

        # Embed the tokens
        self.embedding = nn.Embedding(num_tokens, dim)

    def forward(self, text: Tensor, image: Tensor, *args, **kwargs):
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

        # Get shapes
        t_b, t_s, t_d = x.shape
        i_b, i_c, i_h, i_w = image.shape

        print(f"Text: {x.shape}")
        print(f"Image: {image.shape}")

        # Vision Encoder
        image = self.vit(
            image, return_embeddings=True, *args, **kwargs
        )
        print(f"Image Embedding: {image.shape}")
        image = threed_to_text(image, t_s, t_d)
        print(f"Image reshape: {image.shape}")

        # Connector
        # image = nn.AdaptiveAvgPool1d((t_s, t_d))(image) # 2nd option
        image = CAbstractor(
            self.dim,
            self.depth,
            self.heads,
        )(image)
        print(f"Image Connector: {image.shape}")

        # Decoder
        x = self.decoder(x + image)

        # Return logits
        if self.return_logits:
            return OutputHead(self.dim, -1)(x)

        else:
            return x
