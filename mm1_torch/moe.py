import torch
import torch.nn.functional as F
from torch import nn, Tensor
from zeta.nn import FeedForward


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer.

    This layer combines the outputs of multiple experts based on the gate logits.
    The gate logits determine the weights assigned to each expert's output.

    Args:
        experts (List[nn.Module]): List of expert modules.
        gate (nn.Module): Gate module that produces gate logits.
        num_experts (int): Total number of experts.
        num_experts_per_tok (int): Number of experts to select per token.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        experts (nn.ModuleList): List of expert modules.
        gate (nn.Module): Gate module that produces gate logits.
        num_experts (int): Total number of experts.
        num_experts_per_tok (int): Number of experts to select per token.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        num_experts_per_tok: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        assert dim > 0, "Dimension must be greater than 0"
        assert (
            num_experts > 0
        ), "Number of experts must be greater than 0"
        assert (
            num_experts_per_tok > 0
        ), "Number of experts per token must be greater than 0"

        # Experts
        self.experts = nn.ModuleList([])

        for _ in range(self.num_experts):
            self.experts.append(
                FeedForward(
                    dim,
                    dim,
                    4,
                    swish=True,
                    post_act_ln=True,
                    dropout=0.1,
                    *args,
                    **kwargs,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MoELayer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after combining the outputs of the selected experts.
        """
        gate_logits = self.gate(x)

        weights, selected_experts = torch.topk(
            gate_logits, self.num_experts_per_tok
        )

        weights = F.softmax(weights, dim=1, dtype=torch.float).to(
            x.dtype
        )

        results = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert, _ = torch.where(
                selected_experts == i
            )
            print(batch_idx.shape)
            print(nth_expert.shape)
            print(weights.shape)

            results[batch_idx] += weights[
                batch_idx, nth_expert, None
            ] * expert(x[batch_idx])
            return results


# # Forward
# x = torch.rand(2, 3, 4)


# # Experts
# feedforward = FeedForward(4, 4, 4)

# # Gate
# # MoE
# moe = MoELayer(
#     dim=4, num_experts_per_tok=4, num_experts=8
# )

# # Forward
# print(moe(x))
