"""Interface contact loss module for IsoDDE."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InterfaceContactLoss(nn.Module):
    """Computes binary cross entropy loss for predicted interface contact maps.

    Compares predicted interface contact logits with contact labels derived from
    ground truth 3D coordinates, restricted to inter-chain residue pairs.
    """

    def __init__(self, contact_threshold_angstrom: float = 8.0) -> None:
        super().__init__()
        self.contact_threshold = contact_threshold_angstrom

    def forward(
        self,
        logits: Tensor,
        coords_true: Tensor,
        chain_index: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the interface contact loss.

        Parameters
        ----------
        logits : Tensor (B, N, N)
            Predicted contact logits.
        coords_true : Tensor (B, N, 3)
            Ground truth 3D coordinates.
        chain_index : Tensor (B, N)
            Chain identifier for each residue/token.
        mask : Tensor (B, N), optional
            Residue validity mask.

        Returns
        -------
        Tensor (1,)
            Scalar loss.
        """
        # Compute true pairwise distances: (B, N, N)
        diffs = coords_true.unsqueeze(2) - coords_true.unsqueeze(1)
        dists = torch.linalg.norm(diffs, dim=-1)

        # Contact labels: (B, N, N)
        labels = (dists < self.contact_threshold).float()

        # Inter-chain mask: (B, N, N)
        inter_chain_mask = (chain_index.unsqueeze(-1) != chain_index.unsqueeze(-2))

        # Combined loss mask
        loss_mask = inter_chain_mask
        if mask is not None:
            pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            loss_mask = loss_mask & pair_mask

        # Compute element-wise BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

        # Mask loss
        masked_loss = bce_loss * loss_mask.float()

        # Average over mask
        num_valid_pairs = loss_mask.sum()
        if num_valid_pairs > 0:
            return masked_loss.sum() / num_valid_pairs
        else:
            return torch.tensor(0.0, device=logits.device)
