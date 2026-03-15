import torch
import torch.nn as nn
import torch.nn.functional as F

def print_grad(grad):
    print(f"Number of zeros in the gradient: {(grad == 0).sum().item()} out of {grad.numel()}")

class LinearMatryoshka(nn.Module):

    def __init__(self, in_size: int, out_size: int, matryoshka_depth: int = 1, matryoshka_axis: int = 0, device="cuda"):
        super().__init__()
        self.grans = [i for i in range(matryoshka_depth)]
        matryoshka_len = in_size if matryoshka_axis == 0 else out_size
        self.matryoshka_sizes = [matryoshka_len // (2 ** g) for g in self.grans[1:]]
        self.submodels = nn.ModuleList()
        self.main_model = nn.Linear(in_size, out_size).to(device)
        self.matryoshka_axis = matryoshka_axis
        for i in self.matryoshka_sizes:
            if matryoshka_axis == 0:
                submodel = nn.Linear(i, out_size)
            elif matryoshka_axis == 1:
                submodel = nn.Linear(in_size, i)
            else:
                raise ValueError(f"Invalid matryoshka_axis {matryoshka_axis}. Must be 0 or 1.")

            del submodel._parameters['weight']
            del submodel._parameters['bias']
            if matryoshka_axis == 0:
                submodel.weight = self.main_model.weight[:, :i]
                submodel.bias = self.main_model.bias
            elif matryoshka_axis == 1:
                submodel.weight = self.main_model.weight[:i, :]
                submodel.bias = self.main_model.bias[:i]

            self.submodels.append(submodel)
        # self.main_model.weight.register_hook(print_grad)

    def forward(self, x: torch.Tensor, granularity: int) -> torch.Tensor:
        if granularity not in self.grans:
            raise ValueError(f"Invalid granularity {granularity}. Must be one of {self.grans}.")
        if granularity == 0:
            return self.main_model(x)
        else:
            submodel = self.submodels[granularity - 1]
            if self.matryoshka_axis == 0:
                return submodel(x[:, :self.matryoshka_sizes[granularity - 1]])
            else:
                return submodel(x)

class MultiHeadSelfAttentionMatryoshka(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.
    Args:
        E_q (int): Size of embedding dim for query, key and value (self-attention)
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias is added by default
    """

    def __init__(
        self,
        E_q: int,
        E_total: int,
        nheads: int,
        matryoshka_depth: int = 1,
        dropout: float = 0.0,
        device=None,
    ):
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self.packed_proj = LinearMatryoshka(E_q, E_total * 3, matryoshka_depth, 1, device=device)
        E_out = E_q
        self.out_proj = LinearMatryoshka(E_total, E_out, matryoshka_depth, 0, device=device)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = True

    def forward(
        self,
        query: torch.Tensor,
        matryoshka_granularity,
        is_causal=False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``) - key and values are the same as this is self-attention
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        nheads = self.nheads // (2 ** matryoshka_granularity)
        # Step 1. Apply input projection
        result = self.packed_proj(query, matryoshka_granularity)
        query, key, value = torch.chunk(result, 3, dim=-1)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output, matryoshka_granularity)

        return attn_output