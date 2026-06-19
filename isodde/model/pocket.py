"""Pocket identification head for IsoDDE.

Predicts residue-level ligand-binding probability and performs spatial pocket
clustering using single-linkage clustering at a 5 Å threshold (Section 3).
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from isodde.config import PocketConfig
from isodde.model.primitives import IsoLinear


class PocketIdentificationHead(nn.Module):
    """Predicts ligand-binding pockets from single representation and coordinates.

    Uses a residue-level classification head to predict binding probability,
    then clusters pocket residues using spatial single-linkage clustering.
    """

    def __init__(self, config: PocketConfig, single_dim: int = 256) -> None:
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.LayerNorm(single_dim),
            IsoLinear(single_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            IsoLinear(config.hidden_dim, 1),  # Sigmoid probability output
        )

    def forward(
        self,
        single: Tensor,
        coords: Tensor,
        mask: Optional[Tensor] = None,
        prob_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        """Predict pocket residues and cluster them into distinct pockets.

        Parameters
        ----------
        single : Tensor (B, N, single_dim)
            Residue single representations.
        coords : Tensor (B, N, 3)
            Predicted 3D coordinates.
        mask : Tensor (B, N), optional
            Residue validity mask.
        prob_threshold : float
            Probability threshold to classify a residue as part of a pocket.

        Returns
        -------
        Dict containing:
            - logits: Tensor (B, N)
            - probabilities: Tensor (B, N)
            - pockets: List of List of Dict (for each batch element, list of pockets,
                       each pocket is a dict with residue indices and center coords)
        """
        B, N, _ = single.shape
        logits = self.net(single).squeeze(-1)  # (B, N)
        probabilities = torch.sigmoid(logits)  # (B, N)

        if mask is not None:
            probabilities = probabilities * mask.float()

        batch_pockets = []

        # Perform clustering per batch element
        for b in range(B):
            prob_b = probabilities[b]
            coords_b = coords[b]
            
            # Find candidate pocket residues
            candidate_indices = torch.where(prob_b >= prob_threshold)[0].tolist()
            
            if not candidate_indices:
                batch_pockets.append([])
                continue

            # Spatial single-linkage clustering (5 Å threshold)
            pockets = self._single_linkage_cluster(
                candidate_indices,
                coords_b,
                threshold=self.config.cluster_threshold_angstrom,
                min_residues=self.config.min_pocket_residues,
            )
            batch_pockets.append(pockets)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "pockets": batch_pockets,
        }

    def _single_linkage_cluster(
        self,
        indices: List[int],
        coords: Tensor,
        threshold: float = 5.0,
        min_residues: int = 10,
    ) -> List[Dict[str, Any]]:
        """Perform spatial single-linkage clustering of candidate residues."""
        # Standard disjoint-set / BFS grouping based on spatial distance
        n = len(indices)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            
            # Start new cluster
            current_cluster = []
            queue = [i]
            visited[i] = True

            while queue:
                curr = queue.pop(0)
                current_cluster.append(indices[curr])
                curr_coord = coords[indices[curr]]

                for neighbor in range(n):
                    if not visited[neighbor]:
                        neigh_coord = coords[indices[neighbor]]
                        dist = float(torch.norm(curr_coord - neigh_coord).item())
                        if dist <= threshold:
                            visited[neighbor] = True
                            queue.append(neighbor)

            if len(current_cluster) >= min_residues:
                # Compute center of pocket
                cluster_coords = coords[current_cluster]
                center = cluster_coords.mean(dim=0).tolist()
                clusters.append({
                    "residue_indices": current_cluster,
                    "center": center,
                    "size": len(current_cluster),
                })

        return sorted(clusters, key=lambda p: p["size"], reverse=True)
