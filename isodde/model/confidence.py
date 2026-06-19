"""Confidence prediction head for IsoDDE.

Predicts confidence metrics for ranking multi-seed predictions:
pLDDT (per-residue), pTM (predicted TM-score), ipTM (interface pTM),
and a composite ranking score. These metrics are critical for the
top-1 selection strategy used throughout the paper's benchmarks
(Figures 2, 4, 5, 6, 15).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from isodde.model.primitives import IsoLinear, Transition


class PLDDTHead(nn.Module):
    """Predicted LDDT (pLDDT) per-residue confidence.

    Bins the predicted LDDT into discrete categories and outputs
    a probability distribution, from which the expected pLDDT can
    be computed.

    Parameters
    ----------
    single_dim : int
        Single representation dimension.
    num_bins : int
        Number of LDDT bins.
    """

    def __init__(self, single_dim: int = 256, num_bins: int = 50) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.LayerNorm(single_dim),
            IsoLinear(single_dim, single_dim),
            nn.SiLU(),
            IsoLinear(single_dim, num_bins),
        )

    def forward(self, single: Tensor) -> Tensor:
        """Predict pLDDT logits.

        Parameters
        ----------
        single : Tensor (B, N, single_dim)

        Returns
        -------
        Tensor (B, N, num_bins)
            Logits over LDDT bins.
        """
        return self.net(single)

    def compute_plddt(self, logits: Tensor) -> Tensor:
        """Compute expected pLDDT from logits.

        Parameters
        ----------
        logits : Tensor (B, N, num_bins)

        Returns
        -------
        Tensor (B, N)
            pLDDT scores in [0, 1].
        """
        probs = F.softmax(logits, dim=-1)
        bin_centers = torch.linspace(
            1 / (2 * self.num_bins),
            1 - 1 / (2 * self.num_bins),
            self.num_bins,
            device=logits.device,
        )
        return (probs * bin_centers).sum(dim=-1)


class PTMHead(nn.Module):
    """Predicted TM-score (pTM) head.

    Predicts pairwise distance error distributions from the pair
    representation, from which pTM and ipTM can be computed.

    Parameters
    ----------
    pair_dim : int
        Pair representation dimension.
    num_bins : int
        Number of distance error bins.
    max_dist : float
        Maximum distance for binning.
    """

    def __init__(
        self,
        pair_dim: int = 384,
        num_bins: int = 64,
        max_dist: float = 22.0,
    ) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.max_dist = max_dist
        self.net = nn.Sequential(
            nn.LayerNorm(pair_dim),
            IsoLinear(pair_dim, pair_dim),
            nn.SiLU(),
            IsoLinear(pair_dim, num_bins),
        )

    def forward(self, pair: Tensor) -> Tensor:
        """Predict pairwise distance error logits.

        Parameters
        ----------
        pair : Tensor (B, N, N, pair_dim)

        Returns
        -------
        Tensor (B, N, N, num_bins)
        """
        return self.net(pair)

    def compute_ptm(
        self,
        logits: Tensor,
        mask: Optional[Tensor] = None,
        interface_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute pTM or ipTM from logits.

        Parameters
        ----------
        logits : Tensor (B, N, N, num_bins)
        mask : Tensor (B, N), optional
            Residue mask.
        interface_mask : Tensor (B, N, N), optional
            If provided, computes ipTM over interface pairs only.

        Returns
        -------
        Tensor (B,)
            pTM or ipTM scores.
        """
        probs = F.softmax(logits, dim=-1)

        # Bin edges for distance errors
        bin_centers = torch.linspace(
            0, self.max_dist, self.num_bins, device=logits.device
        )

        # Expected distance error per pair
        expected_error = (probs * bin_centers).sum(dim=-1)  # (B, N, N)

        B, N, _ = expected_error.shape

        # TM-score scaling: d0 depends on sequence length
        d0 = 1.24 * (max(N, 19) - 15) ** (1.0 / 3.0) - 1.8
        d0 = max(d0, 0.5)

        # TM-score contribution per pair
        tm_per_pair = 1.0 / (1.0 + (expected_error / d0) ** 2)

        # Apply masks
        if interface_mask is not None:
            tm_per_pair = tm_per_pair * interface_mask
            count = interface_mask.sum(dim=(-1, -2)).clamp(min=1)
        elif mask is not None:
            pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            tm_per_pair = tm_per_pair * pair_mask
            count = pair_mask.sum(dim=(-1, -2)).clamp(min=1)
        else:
            count = N * N

        return tm_per_pair.sum(dim=(-1, -2)) / count


class RankingScore(nn.Module):
    """Composite ranking score for multi-seed selection.

    Combines pLDDT, pTM, and ipTM into a single scalar ranking score
    used to select the best prediction from multiple seeds.

    Parameters
    ----------
    weights : dict, optional
        Weights for combining confidence metrics.
    """

    def __init__(
        self,
        plddt_weight: float = 0.2,
        ptm_weight: float = 0.8,
        iptm_weight: float = 0.8,
    ) -> None:
        super().__init__()
        self.plddt_weight = plddt_weight
        self.ptm_weight = ptm_weight
        self.iptm_weight = iptm_weight

    def forward(
        self,
        plddt: Tensor,
        ptm: Tensor,
        iptm: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute ranking score.

        Parameters
        ----------
        plddt : Tensor (B, N)
            Per-residue pLDDT.
        ptm : Tensor (B,)
            Predicted TM-score.
        iptm : Tensor (B,), optional
            Interface predicted TM-score.

        Returns
        -------
        Tensor (B,)
            Ranking score (higher is better).
        """
        mean_plddt = plddt.mean(dim=-1)
        score = self.plddt_weight * mean_plddt + self.ptm_weight * ptm

        if iptm is not None:
            # For interfaces, replace ptm component with iptm
            score = (
                self.plddt_weight * mean_plddt
                + (1 - self.plddt_weight) * (
                    0.2 * ptm + 0.8 * iptm
                )
            )

        return score


class ConfidenceHead(nn.Module):
    """Full confidence prediction module.

    Combines pLDDT, pTM, ipTM heads with a ranking score for
    multi-seed selection.

    Parameters
    ----------
    single_dim : int
    pair_dim : int
    plddt_bins : int
    ptm_bins : int
    max_dist : float
    """

    def __init__(
        self,
        single_dim: int = 256,
        pair_dim: int = 384,
        plddt_bins: int = 50,
        ptm_bins: int = 64,
        max_dist: float = 22.0,
    ) -> None:
        super().__init__()
        self.plddt_head = PLDDTHead(single_dim, plddt_bins)
        self.ptm_head = PTMHead(pair_dim, ptm_bins, max_dist)
        self.ranking = RankingScore()

    def forward(
        self,
        single: Tensor,
        pair: Tensor,
        mask: Optional[Tensor] = None,
        interface_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Predict all confidence metrics.

        Parameters
        ----------
        single : Tensor (B, N, single_dim)
        pair : Tensor (B, N, N, pair_dim)
        mask : Tensor (B, N), optional
        interface_mask : Tensor (B, N, N), optional

        Returns
        -------
        dict with keys: plddt, ptm, iptm, ranking_score,
                        plddt_logits, ptm_logits
        """
        plddt_logits = self.plddt_head(single)
        ptm_logits = self.ptm_head(pair)

        plddt = self.plddt_head.compute_plddt(plddt_logits)
        ptm = self.ptm_head.compute_ptm(ptm_logits, mask)

        iptm = None
        if interface_mask is not None:
            iptm = self.ptm_head.compute_ptm(ptm_logits, mask, interface_mask)

        ranking_score = self.ranking(plddt, ptm, iptm)

        result = {
            "plddt": plddt,
            "ptm": ptm,
            "ranking_score": ranking_score,
            "plddt_logits": plddt_logits,
            "ptm_logits": ptm_logits,
        }
        if iptm is not None:
            result["iptm"] = iptm

        return result
