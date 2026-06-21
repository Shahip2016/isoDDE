"""Auxiliary prediction heads for IsoDDE.

Includes distogram prediction, experimentally resolved head, and
masked MSA prediction — auxiliary objectives that support training.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from isodde.model.primitives import IsoLinear


class DistogramHead(nn.Module):
    """Predict pairwise distance distribution from pair representation.

    Bins inter-residue distances into discrete categories for use as
    an auxiliary training objective and for quality assessment.

    Parameters
    ----------
    pair_dim : int
        Pair representation dimension.
    num_bins : int
        Number of distance bins.
    min_dist : float
        Minimum distance (Å).
    max_dist : float
        Maximum distance (Å).
    """

    def __init__(
        self,
        pair_dim: int = 384,
        num_bins: int = 64,
        min_dist: float = 2.3125,
        max_dist: float = 21.6875,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.net = nn.Sequential(
            nn.LayerNorm(pair_dim),
            IsoLinear(pair_dim, pair_dim),
            nn.SiLU(),
            IsoLinear(pair_dim, num_bins),
        )

    def forward(self, pair: Tensor) -> Tensor:
        """Predict distogram logits.

        Parameters
        ----------
        pair : Tensor (B, N, N, pair_dim)

        Returns
        -------
        Tensor (B, N, N, num_bins)
        """
        # Symmetrise pair
        pair_sym = 0.5 * (pair + pair.transpose(-2, -3))
        return self.net(pair_sym)


class ExperimentallyResolvedHead(nn.Module):
    """Predict whether each atom is experimentally resolved.

    Used as an auxiliary training signal and for filtering
    predictions at inference time.

    Parameters
    ----------
    single_dim : int
        Single representation dimension.
    max_atoms : int
        Maximum atoms per token.
    """

    def __init__(self, single_dim: int = 256, max_atoms: int = 14) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(single_dim),
            IsoLinear(single_dim, single_dim),
            nn.SiLU(),
            IsoLinear(single_dim, max_atoms),
        )

    def forward(self, single: Tensor) -> Tensor:
        """Predict resolution logits.

        Parameters
        ----------
        single : Tensor (B, N, single_dim)

        Returns
        -------
        Tensor (B, N, max_atoms)
            Logits for whether each atom is resolved.
        """
        return self.net(single)


class MaskedMSAHead(nn.Module):
    """Predict masked MSA tokens.

    Auxiliary objective: predict the identity of randomly masked
    MSA residues from the MSA representation.

    Parameters
    ----------
    msa_dim : int
        MSA representation dimension.
    num_tokens : int
        Number of MSA token types.
    """

    def __init__(self, msa_dim: int = 64, num_tokens: int = 23) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(msa_dim),
            IsoLinear(msa_dim, num_tokens),
        )

    def forward(self, msa: Tensor) -> Tensor:
        """Predict masked MSA logits.

        Parameters
        ----------
        msa : Tensor (B, N_msa, N_res, msa_dim)

        Returns
        -------
        Tensor (B, N_msa, N_res, num_tokens)
        """
        return self.net(msa)


class SecondaryStructureHead(nn.Module):
    """Predict 3-state secondary structure categories for each residue.

    States: Helix (0), Sheet (1), Coil/Loop (2).
    """

    def __init__(self, single_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(single_dim),
            IsoLinear(single_dim, single_dim),
            nn.SiLU(),
            IsoLinear(single_dim, 3),
        )

    def forward(self, single: Tensor) -> Tensor:
        """Predict secondary structure logits.

        Parameters
        ----------
        single : Tensor (B, N, single_dim)

        Returns
        -------
        Tensor (B, N, 3)
        """
        return self.net(single)


class SolventAccessibilityHead(nn.Module):
    """Predict relative solvent accessibility value for each residue."""

    def __init__(self, single_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(single_dim),
            IsoLinear(single_dim, single_dim),
            nn.SiLU(),
            IsoLinear(single_dim, 1),
        )

    def forward(self, single: Tensor) -> Tensor:
        """Predict solvent accessibility values.

        Parameters
        ----------
        single : Tensor (B, N, single_dim)

        Returns
        -------
        Tensor (B, N)
        """
        return self.net(single).squeeze(-1)
