"""Protein-ligand contact loss module for IsoDDE."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ProteinLigandContactLoss(nn.Module):
    """Computes binary cross entropy loss for predicted protein-ligand contact maps.

    Compares predicted contact logits with contact labels derived from ground truth
    3D coordinates, restricted to protein residue and ligand atom pairs.
    """

    def __init__(self, contact_threshold_angstrom: float = 4.5) -> None:
        super().__init__()
        self.contact_threshold = contact_threshold_angstrom

    def forward(
        self,
        logits: Tensor,
        coords_true: Tensor,
        token_type: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute the protein-ligand contact loss.

        Parameters
        ----------
        logits : Tensor (B, N, N)
            Predicted contact logits.
        coords_true : Tensor (B, N, 3)
            Ground truth 3D coordinates.
        token_type : Tensor (B, N)
            Token types (0 for PROTEIN, 3 for LIGAND).
        mask : Tensor (B, N), optional
            Token validity mask.

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

        # Identify protein and ligand tokens
        # TokenType.PROTEIN is 0, TokenType.LIGAND is 3
        is_protein = (token_type == 0)
        is_ligand = (token_type == 3)

        # Protein-ligand pair mask: (B, N, N)
        # Matches if token i is protein and token j is ligand, or vice versa
        pl_pair_mask = (is_protein.unsqueeze(-1) & is_ligand.unsqueeze(-2)) | (
            is_ligand.unsqueeze(-1) & is_protein.unsqueeze(-2)
        )

        # Combined loss mask
        loss_mask = pl_pair_mask
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
