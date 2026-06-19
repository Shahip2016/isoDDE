"""Triangular self-attention for pair representations.

Implements the O(n³) triangular attention operations (starting node
and ending node variants) with an optional chunk-based O(n²) memory
implementation via FlashAttention-style processing (Section 1.1).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from isodde.model.primitives import IsoLinear


class TriangularAttention(nn.Module):
    """Triangular self-attention on pair representation.

    Attends along one axis of the pair representation while using the
    other axis to provide gating and bias signals, enforcing triangular
    consistency.

    Parameters
    ----------
    pair_dim : int
        Pair representation dimension.
    num_heads : int
        Number of attention heads.
    starting : bool
        If True, attention over starting nodes (axis=-3).
        If False, attention over ending nodes (axis=-2).
    """

    def __init__(
        self,
        pair_dim: int,
        num_heads: int = 4,
        starting: bool = True,
    ) -> None:
        super().__init__()
        self.starting = starting
        self.num_heads = num_heads
        self.head_dim = pair_dim // num_heads
        assert pair_dim % num_heads == 0

        self.norm = nn.LayerNorm(pair_dim)

        self.q_proj = IsoLinear(pair_dim, pair_dim, bias=False)
        self.k_proj = IsoLinear(pair_dim, pair_dim, bias=False)
        self.v_proj = IsoLinear(pair_dim, pair_dim, bias=False)

        # Triangular bias: project pair to per-head bias
        self.bias_proj = IsoLinear(pair_dim, num_heads, bias=False)

        # Gating
        self.gate_proj = IsoLinear(pair_dim, pair_dim)
        self.output_proj = IsoLinear(pair_dim, pair_dim, init_zeros=True)

    def forward(
        self,
        pair: Tensor,
        pair_mask: Optional[Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> Tensor:
        """Apply triangular attention.

        Parameters
        ----------
        pair : Tensor (B, N, N, pair_dim)
        pair_mask : Tensor (B, N, N), optional
        chunk_size : int, optional
            If provided, process in chunks for memory efficiency.

        Returns
        -------
        Tensor (B, N, N, pair_dim)
        """
        B, N, _, D = pair.shape
        pair_normed = self.norm(pair)

        if not self.starting:
            pair_normed = pair_normed.transpose(-2, -3)
            if pair_mask is not None:
                pair_mask = pair_mask.transpose(-1, -2)

        # Project Q, K, V: (B, N, N, D) -> (B, N, N, H, d)
        q = self.q_proj(pair_normed).unflatten(-1, (self.num_heads, self.head_dim))
        k = self.k_proj(pair_normed).unflatten(-1, (self.num_heads, self.head_dim))
        v = self.v_proj(pair_normed).unflatten(-1, (self.num_heads, self.head_dim))

        # Triangular bias from the pair representation
        bias = self.bias_proj(pair_normed)  # (B, N, N, H)

        # Gate from normalised pair
        gate = torch.sigmoid(self.gate_proj(pair_normed))

        if chunk_size is not None and chunk_size < N:
            out = self._chunked_attention(q, k, v, bias, pair_mask, chunk_size)
        else:
            out = self._full_attention(q, k, v, bias, pair_mask)

        out = out * gate

        if not self.starting:
            out = out.transpose(-2, -3)

        return self.output_proj(out)

    def _full_attention(
        self,
        q: Tensor, k: Tensor, v: Tensor,
        bias: Tensor,
        mask: Optional[Tensor],
    ) -> Tensor:
        """Standard O(n³) attention computation."""
        scale = math.sqrt(self.head_dim)
        # Attention along the last N dimension (for each row i):
        # attn[i,j,k] = softmax_k(q[i,j] · k[i,k] + bias[i,k])
        attn = torch.einsum("bijhd,bikhd->bijkh", q, k) / scale

        # Rearrange bias to match: (B, N, N, H) -> (B, i, 1, k, H)
        attn = attn + bias.unsqueeze(2)

        if mask is not None:
            # mask has shape (B, N, N) corresponding to (B, i, k)
            attn_mask = mask.unsqueeze(2).unsqueeze(-1)
            attn = attn.masked_fill(~attn_mask, float("-inf"))

        attn = F.softmax(attn, dim=-2)

        out = torch.einsum("bijkh,bikhd->bijhd", attn, v)
        return out.flatten(-2)

    def _chunked_attention(
        self,
        q: Tensor, k: Tensor, v: Tensor,
        bias: Tensor,
        mask: Optional[Tensor],
        chunk_size: int,
    ) -> Tensor:
        """Memory-efficient chunked attention, reducing peak from O(n³) to O(n²·c).

        Processes attention in chunks along the query axis.
        """
        B, N_i, N_j, H, d = q.shape
        outputs = []
        for start in range(0, N_j, chunk_size):
            end = min(start + chunk_size, N_j)
            q_chunk = q[:, :, start:end]
            out_chunk = self._full_attention(
                q_chunk, k, v, bias, mask
            )
            outputs.append(out_chunk)
        return torch.cat(outputs, dim=2)


class TriangularAttentionStarting(TriangularAttention):
    """Triangular attention over starting nodes."""

    def __init__(self, pair_dim: int, num_heads: int = 4) -> None:
        super().__init__(pair_dim, num_heads, starting=True)


class TriangularAttentionEnding(TriangularAttention):
    """Triangular attention over ending nodes."""

    def __init__(self, pair_dim: int, num_heads: int = 4) -> None:
        super().__init__(pair_dim, num_heads, starting=False)
