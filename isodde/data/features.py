"""Feature computation for IsoDDE model inputs.

Computes single representations, pair representations, MSA features,
and template features from tokenized inputs and external data sources.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from isodde.utils.tensor_utils import one_hot, rbf_encoding
from isodde.data.chemical import ELEMENT_PROPERTIES
from isodde.data.tokenizer import ELEMENTS


class LigandPhysicalEmbedding(nn.Module):
    """Compute physical property embeddings for ligand atoms.

    Uses element features (atomic number, mass, covalent radius, VdW radius)
    and projects them to the single representation dimension.
    """

    def __init__(self, single_dim: int = 256) -> None:
        super().__init__()
        self.single_proj = nn.Linear(4, single_dim)

        # Precompute property table for the elements
        # 13 standard elements + 1 unknown
        num_elements = len(ELEMENTS) + 1
        properties = torch.zeros(num_elements, 4, dtype=torch.float32)

        for idx, symbol in enumerate(ELEMENTS):
            props = ELEMENT_PROPERTIES[symbol]
            properties[idx, 0] = float(props.atomic_number)
            properties[idx, 1] = float(props.mass)
            properties[idx, 2] = float(props.van_der_waals_radius)
            properties[idx, 3] = float(props.covalent_radius)

        # For unknown/fallback element, use mean properties of all elements
        properties[-1] = properties[:-1].mean(dim=0)

        # Register properties as a non-trainable buffer
        self.register_buffer("properties", properties)

    def forward(self, token_ids: Tensor, is_ligand: Tensor) -> Tensor:
        """Compute physical property embeddings.

        Parameters
        ----------
        token_ids : Tensor (B, N)
            Token indices (should represent element indices for ligand).
        is_ligand : Tensor (B, N)
            Boolean mask indicating which tokens are ligand atoms.

        Returns
        -------
        Tensor (B, N, single_dim)
            Physical embeddings, zeroed out for non-ligand tokens.
        """
        # Clamp token IDs to safety range
        clamped_ids = token_ids.clamp(0, self.properties.size(0) - 1)

        # Look up properties: (B, N, 4)
        feat = self.properties[clamped_ids]

        # Project: (B, N, single_dim)
        embedded = self.single_proj(feat)

        # Mask out non-ligand tokens
        embedded = embedded * is_ligand.unsqueeze(-1).float()

        return embedded


class InputEmbedding(nn.Module):
    """Compute initial single and pair representations from tokenized inputs.

    Combines token type embeddings, residue index features, relative
    position encoding, and chain-relative features into dense
    representations consumed by the MSA module and Pairformer.

    Parameters
    ----------
    num_token_types : int
        Number of distinct token types.
    num_residue_types : int
        Vocabulary size for residue-level tokens.
    single_dim : int
        Dimension of single representation.
    pair_dim : int
        Dimension of pair representation.
    max_relative_position : int
        Maximum relative residue index for positional encoding.
    """

    def __init__(
        self,
        num_token_types: int = 6,
        num_residue_types: int = 21,
        single_dim: int = 256,
        pair_dim: int = 384,
        max_relative_position: int = 32,
    ) -> None:
        super().__init__()
        self.single_dim = single_dim
        self.pair_dim = pair_dim
        self.max_relative_position = max_relative_position

        # Token embeddings
        self.token_embedding = nn.Embedding(num_residue_types, single_dim)
        self.token_type_embedding = nn.Embedding(num_token_types, single_dim)

        # Physical embedding for ligand atoms
        self.ligand_physical_embedding = LigandPhysicalEmbedding(single_dim)

        # Relative position encoding for pair representation
        self.relpos_embedding = nn.Embedding(
            2 * max_relative_position + 1, pair_dim
        )

        # Project single representations to pair axes
        self.left_proj = nn.Linear(single_dim, pair_dim)
        self.right_proj = nn.Linear(single_dim, pair_dim)

        # Chain embedding for inter-chain pair features
        self.chain_same_embedding = nn.Embedding(2, pair_dim)

    def forward(
        self,
        token_ids: Tensor,
        token_type: Tensor,
        residue_index: Tensor,
        chain_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute initial single and pair representations.

        Parameters
        ----------
        token_ids : Tensor (B, N)
        token_type : Tensor (B, N)
        residue_index : Tensor (B, N)
        chain_index : Tensor (B, N)

        Returns
        -------
        single : Tensor (B, N, single_dim)
        pair : Tensor (B, N, N, pair_dim)
        """
        # Single representation
        single = self.token_embedding(token_ids) + self.token_type_embedding(token_type)

        # Add physical property embeddings for ligand tokens (type = 3)
        is_ligand = (token_type == 3)
        single = single + self.ligand_physical_embedding(token_ids, is_ligand)

        # Pair representation
        left = self.left_proj(single)
        right = self.right_proj(single)
        pair = left.unsqueeze(2) + right.unsqueeze(1)

        # Relative position encoding
        rel_pos = residue_index.unsqueeze(2) - residue_index.unsqueeze(1)
        rel_pos = rel_pos.clamp(
            -self.max_relative_position, self.max_relative_position
        ) + self.max_relative_position
        pair = pair + self.relpos_embedding(rel_pos)

        # Same-chain indicator
        same_chain = (chain_index.unsqueeze(2) == chain_index.unsqueeze(1)).long()
        pair = pair + self.chain_same_embedding(same_chain)

        return single, pair


class MSAFeatures(nn.Module):
    """Compute MSA features for the MSA processing module.

    Embeds MSA rows and produces row-wise and column-wise features
    from the multiple sequence alignment.

    Parameters
    ----------
    msa_dim : int
        Embedding dimension for MSA rows.
    num_msa_types : int
        Vocabulary size for MSA tokens (AA + gap + unknown).
    pair_dim : int
        Pair representation dimension (for outer product mean).
    """

    def __init__(
        self,
        msa_dim: int = 64,
        num_msa_types: int = 23,  # 20 AA + gap + mask + unknown
        pair_dim: int = 384,
    ) -> None:
        super().__init__()
        self.msa_embedding = nn.Embedding(num_msa_types, msa_dim)
        self.has_deletion = nn.Linear(1, msa_dim)
        self.deletion_value = nn.Linear(1, msa_dim)

    def forward(
        self,
        msa_tokens: Tensor,
        has_deletion: Tensor,
        deletion_value: Tensor,
    ) -> Tensor:
        """Compute MSA features.

        Parameters
        ----------
        msa_tokens : Tensor (B, N_msa, N_res)
            MSA token indices.
        has_deletion : Tensor (B, N_msa, N_res)
            Binary indicator for deletions.
        deletion_value : Tensor (B, N_msa, N_res)
            Deletion count (normalised).

        Returns
        -------
        Tensor (B, N_msa, N_res, msa_dim)
        """
        msa_feat = self.msa_embedding(msa_tokens)
        msa_feat = msa_feat + self.has_deletion(has_deletion.unsqueeze(-1))
        msa_feat = msa_feat + self.deletion_value(deletion_value.unsqueeze(-1))
        return msa_feat


class TemplateFeatures(nn.Module):
    """Compute pairwise template features.

    Projects template structural information into the pair
    representation space.

    Parameters
    ----------
    pair_dim : int
        Pair representation dimension.
    num_template_features : int
        Number of input template features per pair.
    """

    def __init__(
        self,
        pair_dim: int = 384,
        num_template_features: int = 88,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(num_template_features, pair_dim)
        self.norm = nn.LayerNorm(pair_dim)

    def forward(self, template_pair_feat: Tensor) -> Tensor:
        """Project template features to pair space.

        Parameters
        ----------
        template_pair_feat : Tensor (B, N_templates, N, N, num_features)

        Returns
        -------
        Tensor (B, N, N, pair_dim) — averaged over templates.
        """
        projected = self.proj(template_pair_feat)
        return self.norm(projected.mean(dim=1))
