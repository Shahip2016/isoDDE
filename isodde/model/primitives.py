"""Primitive neural network building blocks for IsoDDE.

Provides Linear, LayerNorm, and attention modules used throughout the
model architecture.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class IsoLinear(nn.Module):
    """Linear layer with optional bias and specific initialisation.

    Uses the truncated-normal init scheme common in structure prediction
    models for stable training at depth.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_scale: float = 1.0,
        init_zeros: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if init_zeros:
            nn.init.zeros_(self.linear.weight)
            if bias:
                nn.init.zeros_(self.linear.bias)
        else:
            std = init_scale * math.sqrt(2.0 / (in_features + out_features))
            nn.init.trunc_normal_(self.linear.weight, std=std)
            if bias:
                nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class Transition(nn.Module):
    """Two-layer feed-forward transition block with SiLU activation.

    Expands the representation by a multiplicative factor, applies SiLU,
    then projects back to the original dimension.

    Parameters
    ----------
    dim : int
        Input and output dimension.
    multiplier : float
        Hidden dimension expansion factor.
    """

    def __init__(self, dim: int, multiplier: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * multiplier)
        self.norm = nn.LayerNorm(dim)
        self.linear1 = IsoLinear(dim, hidden)
        self.linear2 = IsoLinear(hidden, dim, init_zeros=True)

    def forward(self, x: Tensor) -> Tensor:
        x_norm = self.norm(x)
        return x + self.linear2(F.silu(self.linear1(x_norm)))


class Attention(nn.Module):
    """Multi-head attention with optional gating and pair bias.

    Parameters
    ----------
    q_dim : int
        Query input dimension.
    kv_dim : int
        Key/value input dimension.
    output_dim : int
        Output dimension.
    num_heads : int
        Number of attention heads.
    gating : bool
        Whether to apply sigmoid gating to the output.
    pair_bias : bool
        Whether to add a learned bias from pair representation.
    pair_dim : int
        Dimension of pair representation (required if pair_bias=True).
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        output_dim: int,
        num_heads: int,
        gating: bool = True,
        pair_bias: bool = False,
        pair_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        assert output_dim % num_heads == 0

        self.q_norm = nn.LayerNorm(q_dim)
        self.kv_norm = nn.LayerNorm(kv_dim) if kv_dim != q_dim else self.q_norm

        self.q_proj = IsoLinear(q_dim, output_dim, bias=False)
        self.k_proj = IsoLinear(kv_dim, output_dim, bias=False)
        self.v_proj = IsoLinear(kv_dim, output_dim, bias=False)
        self.o_proj = IsoLinear(output_dim, q_dim, init_zeros=True)

        self.gating = gating
        if gating:
            self.gate_proj = IsoLinear(q_dim, output_dim)

        self.pair_bias = pair_bias
        if pair_bias:
            self.pair_bias_norm = nn.LayerNorm(pair_dim)
            self.pair_bias_proj = IsoLinear(pair_dim, num_heads, bias=False)

    def forward(
        self,
        q_input: Tensor,
        kv_input: Optional[Tensor] = None,
        pair: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        q_input : Tensor (..., N_q, q_dim)
        kv_input : Tensor (..., N_kv, kv_dim), optional
            Defaults to q_input (self-attention).
        pair : Tensor (..., N_q, N_kv, pair_dim), optional
        mask : Tensor (..., N_q, N_kv), optional

        Returns
        -------
        Tensor (..., N_q, q_dim)
        """
        if kv_input is None:
            kv_input = q_input

        q = self.q_norm(q_input)
        kv = self.kv_norm(kv_input)

        # Project and reshape to (... , N, H, D)
        batch_shape = q.shape[:-1]
        q = self.q_proj(q).unflatten(-1, (self.num_heads, self.head_dim))
        k = self.k_proj(kv).unflatten(-1, (self.num_heads, self.head_dim))
        v = self.v_proj(kv).unflatten(-1, (self.num_heads, self.head_dim))

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        # (..., H, N_q, N_kv)
        attn = torch.einsum("...qhd,...khd->...hqk", q, k) / scale

        # Add pair bias
        if self.pair_bias and pair is not None:
            bias = self.pair_bias_proj(self.pair_bias_norm(pair))
            # (..., N_q, N_kv, H) -> (..., H, N_q, N_kv)
            attn = attn + bias.movedim(-1, -3)

        # Apply mask
        if mask is not None:
            if mask.dim() < attn.dim():
                mask = mask.unsqueeze(-3)  # add head dimension
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        out = torch.einsum("...hqk,...khd->...qhd", attn, v)
        out = out.flatten(-2)  # merge heads

        # Gating
        if self.gating:
            gate = torch.sigmoid(self.gate_proj(self.q_norm(q_input)))
            out = out * gate

        return self.o_proj(out)
