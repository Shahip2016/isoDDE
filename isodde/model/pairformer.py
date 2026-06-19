"""Pairformer stack for IsoDDE.

The Pairformer processes pair and single representations through
alternating triangular operations, single-track attention, and
transition blocks. Uses wider pair representations (384-dim) following
improvements noted in Zhou et al. (2025) / SeedFold (Section 1.1).
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torch import Tensor

from isodde.model.primitives import Attention, Transition
from isodde.model.triangular_attention import (
    TriangularAttentionStarting,
    TriangularAttentionEnding,
)
from isodde.model.triangular_multiplication import (
    TriangularMultiplicationOutgoing,
    TriangularMultiplicationIncoming,
)


class PairformerBlock(nn.Module):
    """Single Pairformer block.

    Applies the following operations in sequence:
    1. Triangular multiplicative update (outgoing)
    2. Triangular multiplicative update (incoming)
    3. Triangular self-attention (starting node)
    4. Triangular self-attention (ending node)
    5. Pair transition
    6. Single-track attention with pair bias
    7. Single transition

    Parameters
    ----------
    pair_dim : int
        Pair representation dimension.
    single_dim : int
        Single representation dimension.
    num_heads_pair : int
        Attention heads for triangular attention.
    num_heads_single : int
        Attention heads for single-track attention.
    transition_multiplier : float
        FFN expansion factor.
    dropout : float
        Dropout rate.
    chunk_size : int, optional
        Chunk size for memory-efficient triangular attention.
    """

    def __init__(
        self,
        pair_dim: int = 384,
        single_dim: int = 256,
        num_heads_pair: int = 16,
        num_heads_single: int = 16,
        transition_multiplier: float = 4.0,
        dropout: float = 0.0,
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size

        # Pair track
        self.tri_mul_out = TriangularMultiplicationOutgoing(pair_dim)
        self.tri_mul_in = TriangularMultiplicationIncoming(pair_dim)
        self.tri_att_start = TriangularAttentionStarting(pair_dim, num_heads_pair)
        self.tri_att_end = TriangularAttentionEnding(pair_dim, num_heads_pair)
        self.pair_transition = Transition(pair_dim, transition_multiplier)

        # Single track with pair bias
        self.single_attention = Attention(
            q_dim=single_dim,
            kv_dim=single_dim,
            output_dim=single_dim,
            num_heads=num_heads_single,
            gating=True,
            pair_bias=True,
            pair_dim=pair_dim,
        )
        self.single_transition = Transition(single_dim, transition_multiplier)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        pair: Tensor,
        single: Tensor,
        pair_mask: Optional[Tensor] = None,
        single_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply one Pairformer block.

        Parameters
        ----------
        pair : Tensor (B, N, N, pair_dim)
        single : Tensor (B, N, single_dim)
        pair_mask : Tensor (B, N, N), optional
        single_mask : Tensor (B, N), optional

        Returns
        -------
        pair : Tensor (B, N, N, pair_dim)
        single : Tensor (B, N, single_dim)
        """
        # Pair track updates
        pair = pair + self.dropout(self.tri_mul_out(pair, pair_mask))
        pair = pair + self.dropout(self.tri_mul_in(pair, pair_mask))
        pair = pair + self.dropout(
            self.tri_att_start(pair, pair_mask, self.chunk_size)
        )
        pair = pair + self.dropout(
            self.tri_att_end(pair, pair_mask, self.chunk_size)
        )
        pair = self.pair_transition(pair)

        # Single track update with pair bias
        attn_mask = None
        if single_mask is not None:
            attn_mask = single_mask.unsqueeze(1) & single_mask.unsqueeze(2)
        single = single + self.dropout(
            self.single_attention(single, pair=pair, mask=attn_mask)
        )
        single = self.single_transition(single)

        return pair, single


class Pairformer(nn.Module):
    """Full Pairformer stack.

    Applies multiple PairformerBlocks in sequence to iteratively refine
    pair and single representations. The 48-block default follows the
    scale described in the paper.

    Parameters
    ----------
    pair_dim : int
        Pair representation dimension (384, wider than AF3's 128).
    single_dim : int
        Single representation dimension.
    num_blocks : int
        Number of Pairformer blocks.
    num_heads_pair : int
        Attention heads for triangular attention.
    num_heads_single : int
        Attention heads for single-track attention.
    transition_multiplier : float
        FFN expansion factor.
    dropout : float
        Dropout rate.
    chunk_size : int, optional
        Chunk size for memory-efficient triangular operations.
    """

    def __init__(
        self,
        pair_dim: int = 384,
        single_dim: int = 256,
        num_blocks: int = 48,
        num_heads_pair: int = 16,
        num_heads_single: int = 16,
        transition_multiplier: float = 4.0,
        dropout: float = 0.0,
        chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            PairformerBlock(
                pair_dim=pair_dim,
                single_dim=single_dim,
                num_heads_pair=num_heads_pair,
                num_heads_single=num_heads_single,
                transition_multiplier=transition_multiplier,
                dropout=dropout,
                chunk_size=chunk_size,
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        pair: Tensor,
        single: Tensor,
        pair_mask: Optional[Tensor] = None,
        single_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply all Pairformer blocks.

        Parameters
        ----------
        pair : Tensor (B, N, N, pair_dim)
        single : Tensor (B, N, single_dim)
        pair_mask : Tensor (B, N, N), optional
        single_mask : Tensor (B, N), optional

        Returns
        -------
        pair : Tensor (B, N, N, pair_dim)
        single : Tensor (B, N, single_dim)
        """
        for block in self.blocks:
            pair, single = block(pair, single, pair_mask, single_mask)
        return pair, single
