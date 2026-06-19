"""Binding affinity prediction head for IsoDDE.

Predicts binding free energy (delta G) / pKd / pIC50 from structure-conditioned
pair representations and spatial pocket-ligand interaction features.
"""

from __future__ import annotations

from typing import Optional, Dict
import torch
import torch.nn as nn
from torch import Tensor

from isodde.config import AffinityConfig
from isodde.model.primitives import IsoLinear, Attention, Transition
from isodde.utils.tensor_utils import rbf_encoding


class BindingAffinityHead(nn.Module):
    """Structure-conditioned binding affinity prediction head.

    Extracts spatial interaction features between ligand atoms and pocket
    residues, combines them with the trunk pair representation, and
    aggregates them via cross-attention to predict scalar binding affinity.
    """

    def __init__(self, config: AffinityConfig, pair_dim: int = 384, single_dim: int = 256) -> None:
        super().__init__()
        self.config = config

        # Spatial interaction encoder
        self.dist_projection = nn.Sequential(
            IsoLinear(32, config.hidden_dim),  # 32 bins of RBF
            nn.SiLU(),
            IsoLinear(config.hidden_dim, config.hidden_dim),
        )

        self.pair_proj = IsoLinear(pair_dim, config.hidden_dim)
        
        # Interaction layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attention": Attention(
                    q_dim=config.hidden_dim,
                    kv_dim=config.hidden_dim,
                    output_dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    gating=True,
                ),
                "transition": Transition(config.hidden_dim),
            })
            for _ in range(config.num_layers)
        ])

        # Global pooling and prediction MLP
        self.pool_attention = Attention(
            q_dim=config.hidden_dim,
            kv_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_heads=config.num_heads,
            gating=True,
        )
        
        self.pool_query = nn.Parameter(torch.randn(1, 1, config.hidden_dim))

        self.predict_head = nn.Sequential(
            IsoLinear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            IsoLinear(config.hidden_dim, config.output_dim),
        )

    def forward(
        self,
        pair: Tensor,
        coords: Tensor,
        is_ligand: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict binding affinity.

        Parameters
        ----------
        pair : Tensor (B, N, N, pair_dim)
            Refined pair representations.
        coords : Tensor (B, N, 3)
            Predicted 3D coordinates.
        is_ligand : Tensor (B, N)
            Boolean mask indicating ligand tokens.
        mask : Tensor (B, N), optional
            Residue / token validity mask.

        Returns
        -------
        Tensor (B, 1)
            Predicted binding affinity (e.g. pKd / pIC50).
        """
        B, N, _, _ = pair.shape
        device = pair.device

        # Compute pairwise distance matrix
        dist_matrix = torch.cdist(coords, coords)  # (B, N, N)
        
        # RBF encode distances
        rbf_dist = rbf_encoding(dist_matrix, num_bins=32, max_dist=20.0)  # (B, N, N, 32)
        spatial_features = self.dist_projection(rbf_dist)  # (B, N, N, hidden_dim)

        # Combine spatial features with trunk pair representations
        h_pair = self.pair_proj(pair) + spatial_features  # (B, N, N, hidden_dim)

        # Identify pocket-ligand interactions
        # If no ligand tokens found, fallback to full pair interaction
        ligand_mask = is_ligand.unsqueeze(-1) & (~is_ligand).unsqueeze(-2)  # (B, N, N)
        if ligand_mask.sum() == 0:
            ligand_mask = torch.ones(B, N, N, dtype=torch.bool, device=device)

        if mask is not None:
            ligand_mask = ligand_mask & mask.unsqueeze(-1) & mask.unsqueeze(-2)

        # Flatten interaction edges
        # We collect all candidate interactions
        # For simplicity and training speed, we average the representations of interaction edges
        # and pass them through self-attention
        interaction_mask = ligand_mask.flatten(1, 2)  # (B, N*N)
        h_flat = h_pair.flatten(1, 2)  # (B, N*N, hidden_dim)

        # Attention blocks over interactions
        for block in self.layers:
            attn_mask = interaction_mask.unsqueeze(1) & interaction_mask.unsqueeze(2)
            h_flat = h_flat + block["attention"](h_flat, mask=attn_mask)
            h_flat = block["transition"](h_flat)

        # Pool interaction features into a global binding pocket representation
        # Query with pool_query parameter
        q = self.pool_query.expand(B, 1, -1)
        pool_mask = interaction_mask.unsqueeze(1)
        pooled = self.pool_attention(q, kv=h_flat, mask=pool_mask)  # (B, 1, hidden_dim)

        # Predict final scalar binding free energy
        affinity = self.predict_head(pooled.squeeze(1))  # (B, 1)

        return affinity
