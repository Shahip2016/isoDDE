"""Triangular multiplicative updates for pair representations.

Implements outgoing and incoming triangular multiplication operations
that enforce consistency in the pair representation by aggregating
information along the edges of a triangle.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from isodde.model.primitives import IsoLinear


class TriangularMultiplication(nn.Module):
    """Triangular multiplicative update.

    Updates pair[i,j] by aggregating products of pair entries along a
    shared intermediate node k, enforcing triangular consistency.

    Parameters
    ----------
    pair_dim : int
        Pair representation dimension.
    inner_dim : int
        Intermediate projection dimension.
    outgoing : bool
        If True, outgoing edges (pair[i,k] * pair[j,k]).
        If False, incoming edges (pair[k,i] * pair[k,j]).
    """

    def __init__(
        self,
        pair_dim: int,
        inner_dim: Optional[int] = None,
        outgoing: bool = True,
    ) -> None:
        super().__init__()
        self.outgoing = outgoing
        if inner_dim is None:
            inner_dim = pair_dim

        self.norm = nn.LayerNorm(pair_dim)

        self.left_proj = IsoLinear(pair_dim, inner_dim)
        self.right_proj = IsoLinear(pair_dim, inner_dim)
        self.left_gate = IsoLinear(pair_dim, inner_dim)
        self.right_gate = IsoLinear(pair_dim, inner_dim)

        self.output_gate = IsoLinear(pair_dim, pair_dim)
        self.center_norm = nn.LayerNorm(inner_dim)
        self.output_proj = IsoLinear(inner_dim, pair_dim, init_zeros=True)

    def forward(
        self, pair: Tensor, pair_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Apply triangular multiplicative update.

        Parameters
        ----------
        pair : Tensor (B, N, N, pair_dim)
        pair_mask : Tensor (B, N, N), optional

        Returns
        -------
        Tensor (B, N, N, pair_dim)
        """
        pair_normed = self.norm(pair)

        if pair_mask is not None:
            pair_normed = pair_normed * pair_mask.unsqueeze(-1)

        # Project with gating
        left = self.left_proj(pair_normed) * torch.sigmoid(
            self.left_gate(pair_normed)
        )
        right = self.right_proj(pair_normed) * torch.sigmoid(
            self.right_gate(pair_normed)
        )

        # Triangular aggregation
        if self.outgoing:
            # out[i,j] = sum_k left[i,k] * right[j,k]
            result = torch.einsum("bikc,bjkc->bijc", left, right)
        else:
            # out[i,j] = sum_k left[k,i] * right[k,j]
            result = torch.einsum("bkic,bkjc->bijc", left, right)

        result = self.center_norm(result)

        # Output gating
        gate = torch.sigmoid(self.output_gate(pair_normed))

        return self.output_proj(result) * gate


class TriangularMultiplicationOutgoing(TriangularMultiplication):
    """Triangular multiplicative update with outgoing edges."""

    def __init__(self, pair_dim: int, inner_dim: Optional[int] = None) -> None:
        super().__init__(pair_dim, inner_dim, outgoing=True)


class TriangularMultiplicationIncoming(TriangularMultiplication):
    """Triangular multiplicative update with incoming edges."""

    def __init__(self, pair_dim: int, inner_dim: Optional[int] = None) -> None:
        super().__init__(pair_dim, inner_dim, outgoing=False)
