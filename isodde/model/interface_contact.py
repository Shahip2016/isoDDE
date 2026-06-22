"""Interface contact prediction head for IsoDDE.

Predicts residue-level inter-chain contact probabilities from the pair representation.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

from isodde.config import InterfaceContactConfig
from isodde.model.primitives import IsoLinear


class InterfaceContactHead(nn.Module):
    """Predicts inter-chain residue contacts from pair representations."""

    def __init__(self, config: InterfaceContactConfig, pair_dim: int = 384) -> None:
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.LayerNorm(pair_dim),
            IsoLinear(pair_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            IsoLinear(config.hidden_dim, 1),  # Logits for contact probability
        )

    def forward(self, pair: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Predict interface contact logits.

        Parameters
        ----------
        pair : Tensor (B, N, N, pair_dim)
            Pair representations.
        mask : Tensor (B, N), optional
            Residue validity mask.

        Returns
        -------
        Tensor (B, N, N)
            Contact logits.
        """
        # Symmetrize the pair representation as contacts are symmetric
        pair_sym = 0.5 * (pair + pair.transpose(-2, -3))

        logits = self.net(pair_sym).squeeze(-1)  # (B, N, N)

        if mask is not None:
            # Mask out invalid residue pairs
            pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            logits = logits.masked_fill(~pair_mask, -1e9)

        return logits
