"""MSA processing module for IsoDDE.

Implements row-wise and column-wise attention over multiple sequence
alignments, with improved information flow between single and pair
representations. The reordered operations follow improvements described
in Section 1.1 (Wohlwend et al. 2024, ByteDance AML 2025).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from isodde.model.primitives import Attention, Transition, IsoLinear


class OuterProductMean(nn.Module):
    """Compute outer product mean from MSA to update pair representation.

    For each pair (i, j), computes the mean outer product of the MSA
    column representations at positions i and j.

    Parameters
    ----------
    msa_dim : int
        MSA embedding dimension.
    pair_dim : int
        Pair representation dimension.
    inner_dim : int
        Intermediate projection dimension.
    """

    def __init__(
        self, msa_dim: int, pair_dim: int, inner_dim: int = 32
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(msa_dim)
        self.left_proj = IsoLinear(msa_dim, inner_dim)
        self.right_proj = IsoLinear(msa_dim, inner_dim)
        self.output_proj = IsoLinear(inner_dim * inner_dim, pair_dim, init_zeros=True)

    def forward(self, msa: Tensor, msa_mask: Optional[Tensor] = None) -> Tensor:
        """Compute outer product mean.

        Parameters
        ----------
        msa : Tensor (B, N_msa, N_res, msa_dim)
        msa_mask : Tensor (B, N_msa, N_res), optional

        Returns
        -------
        Tensor (B, N_res, N_res, pair_dim)
        """
        msa = self.norm(msa)
        left = self.left_proj(msa)   # (B, N_msa, N_res, inner)
        right = self.right_proj(msa)  # (B, N_msa, N_res, inner)

        if msa_mask is not None:
            left = left * msa_mask.unsqueeze(-1)
            right = right * msa_mask.unsqueeze(-1)

        # Outer product: (B, N_msa, N_res_i, inner) x (B, N_msa, N_res_j, inner)
        # -> (B, N_res_i, N_res_j, inner*inner)
        outer = torch.einsum("bsic,bsjd->bijcd", left, right)
        outer = outer.flatten(-2)  # (B, N_res, N_res, inner*inner)

        # Mean over MSA sequences
        if msa_mask is not None:
            n_seqs = msa_mask.sum(dim=1).clamp(min=1)
            # n_seqs: (B, N_res) — but we averaged in einsum, so just divide
            outer = outer / msa.shape[1]
        else:
            outer = outer / msa.shape[1]

        return self.output_proj(outer)


class MSARowAttention(nn.Module):
    """Row-wise attention over MSA with pair bias.

    Each MSA sequence attends to all residue positions, with attention
    biases derived from the pair representation.

    Parameters
    ----------
    msa_dim : int
        MSA embedding dimension.
    pair_dim : int
        Pair representation dimension (for bias).
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, msa_dim: int, pair_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.attention = Attention(
            q_dim=msa_dim,
            kv_dim=msa_dim,
            output_dim=msa_dim,
            num_heads=num_heads,
            gating=True,
            pair_bias=True,
            pair_dim=pair_dim,
        )

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        msa_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply row-wise attention.

        Parameters
        ----------
        msa : Tensor (B, N_msa, N_res, msa_dim)
        pair : Tensor (B, N_res, N_res, pair_dim)
        msa_mask : Tensor (B, N_msa, N_res), optional

        Returns
        -------
        Tensor (B, N_msa, N_res, msa_dim)
        """
        B, S, N, D = msa.shape

        # Reshape: treat each MSA row independently
        msa_flat = msa.reshape(B * S, N, D)

        # Pair bias is shared across MSA rows
        pair_expanded = pair.unsqueeze(1).expand(-1, S, -1, -1, -1)
        pair_flat = pair_expanded.reshape(B * S, N, N, -1)

        mask = None
        if msa_mask is not None:
            mask = msa_mask.reshape(B * S, N)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)

        out = self.attention(msa_flat, pair=pair_flat, mask=mask)
        return msa + out.reshape(B, S, N, D)


class MSAColumnAttention(nn.Module):
    """Column-wise attention over MSA.

    Each residue position attends across all MSA sequences, enabling
    the model to extract co-evolutionary information.

    Parameters
    ----------
    msa_dim : int
        MSA embedding dimension.
    num_heads : int
        Number of attention heads.
    """

    def __init__(self, msa_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.attention = Attention(
            q_dim=msa_dim,
            kv_dim=msa_dim,
            output_dim=msa_dim,
            num_heads=num_heads,
            gating=True,
        )

    def forward(
        self, msa: Tensor, msa_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Apply column-wise attention.

        Parameters
        ----------
        msa : Tensor (B, N_msa, N_res, msa_dim)
        msa_mask : Tensor (B, N_msa, N_res), optional

        Returns
        -------
        Tensor (B, N_msa, N_res, msa_dim)
        """
        B, S, N, D = msa.shape

        # Transpose to (B, N_res, N_msa, msa_dim) for column attention
        msa_t = msa.permute(0, 2, 1, 3).reshape(B * N, S, D)

        mask = None
        if msa_mask is not None:
            mask_t = msa_mask.permute(0, 2, 1).reshape(B * N, S)
            mask = mask_t.unsqueeze(1) & mask_t.unsqueeze(2)

        out = self.attention(msa_t, mask=mask)
        out = out.reshape(B, N, S, D).permute(0, 2, 1, 3)
        return msa + out


class MSAModule(nn.Module):
    """Full MSA processing module.

    Applies alternating row and column attention blocks with outer
    product mean updates to the pair representation. Uses the
    improved operation ordering from Section 1.1: outer product mean
    is computed BEFORE row attention (improved single↔pair flow).

    Parameters
    ----------
    msa_dim : int
        MSA embedding dimension.
    pair_dim : int
        Pair representation dimension.
    num_blocks : int
        Number of MSA processing blocks.
    num_heads : int
        Number of attention heads.
    opm_dim : int
        Inner dimension for outer product mean.
    """

    def __init__(
        self,
        msa_dim: int = 64,
        pair_dim: int = 384,
        num_blocks: int = 4,
        num_heads: int = 8,
        opm_dim: int = 32,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                # Reordered: OPM first, then row attention (improved flow)
                "outer_product_mean": OuterProductMean(msa_dim, pair_dim, opm_dim),
                "row_attention": MSARowAttention(msa_dim, pair_dim, num_heads),
                "column_attention": MSAColumnAttention(msa_dim, num_heads),
                "msa_transition": Transition(msa_dim),
            }))

    def forward(
        self,
        msa: Tensor,
        pair: Tensor,
        msa_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Process MSA and update pair representation.

        Parameters
        ----------
        msa : Tensor (B, N_msa, N_res, msa_dim)
        pair : Tensor (B, N_res, N_res, pair_dim)
        msa_mask : Tensor (B, N_msa, N_res), optional

        Returns
        -------
        msa : Tensor (B, N_msa, N_res, msa_dim)
        pair : Tensor (B, N_res, N_res, pair_dim)
        """
        for block in self.blocks:
            # Improved ordering: OPM first to update pair before row attention
            pair = pair + block["outer_product_mean"](msa, msa_mask)
            msa = block["row_attention"](msa, pair, msa_mask)
            msa = block["column_attention"](msa, msa_mask)
            msa = block["msa_transition"](msa)

        return msa, pair
