"""Structure prediction model for IsoDDE.

Assembles the input embedding, MSA module, Pairformer, score-based diffusion
module, confidence heads, and auxiliary prediction heads into an end-to-end
trainable PyTorch model.
"""

from __future__ import annotations

from typing import Optional, Union, Dict
import torch
import torch.nn as nn
from torch import Tensor

from isodde.config import IsoDDEConfig
from isodde.data.features import InputEmbedding, MSAFeatures, TemplateFeatures
from isodde.model.msa_module import MSAModule
from isodde.model.pairformer import Pairformer
from isodde.model.diffusion import DiffusionModule
from isodde.model.confidence import ConfidenceHead
from isodde.model.heads import (
    DistogramHead,
    ExperimentallyResolvedHead,
    MaskedMSAHead,
    SecondaryStructureHead,
    SolventAccessibilityHead,
)


class IsoDDEStructurePrediction(nn.Module):
    """End-to-end structure prediction model for IsoDDE.

    Assembles:
    1. InputEmbedding (tokens, token_types, residue_index, chain_index)
    2. TemplateFeatures (optional)
    3. MSAFeatures + MSAModule
    4. Pairformer (refining single and pair tracks)
    5. DiffusionModule (coordinate generation)
    6. ConfidenceHead (pLDDT, pTM, ipTM)
    7. Auxiliary heads (distogram, experimental resolution, masked MSA)
    """

    def __init__(self, config: IsoDDEConfig) -> None:
        super().__init__()
        self.config = config

        # Input embeds
        self.embedding = InputEmbedding(
            num_token_types=config.data.num_bond_types + 5, # safety margin
            num_residue_types=config.data.num_atom_types + config.data.num_residue_types + 10,
            single_dim=config.pairformer.single_dim,
            pair_dim=config.pairformer.pair_dim,
        )

        # Template features
        self.template_features = TemplateFeatures(
            pair_dim=config.pairformer.pair_dim
        )

        # MSA features and trunk module
        self.msa_features = MSAFeatures(
            msa_dim=config.msa.msa_embedding_dim,
            pair_dim=config.pairformer.pair_dim,
        )
        self.msa_module = MSAModule(
            msa_dim=config.msa.msa_embedding_dim,
            pair_dim=config.pairformer.pair_dim,
            num_blocks=config.msa.num_msa_blocks,
            num_heads=config.msa.num_heads,
            opm_dim=config.msa.outer_product_mean_dim,
        )

        # Pairformer
        self.pairformer = Pairformer(
            pair_dim=config.pairformer.pair_dim,
            single_dim=config.pairformer.single_dim,
            num_blocks=config.pairformer.num_blocks,
            num_heads_pair=config.pairformer.num_heads_pair,
            num_heads_single=config.pairformer.num_heads_single,
            transition_multiplier=config.pairformer.transition_multiplier,
            dropout=config.pairformer.dropout,
            chunk_size=config.pairformer.chunk_size,
        )

        # Structure prediction (diffusion)
        self.diffusion = DiffusionModule(
            single_dim=config.pairformer.single_dim,
            pair_dim=config.pairformer.pair_dim,
            atom_dim=config.diffusion.atom_dim,
            num_steps=config.diffusion.num_diffusion_steps,
            sigma_min=config.diffusion.sigma_min,
            sigma_max=config.diffusion.sigma_max,
            num_layers=config.diffusion.num_denoising_layers,
            num_heads=config.diffusion.num_heads,
            time_dim=config.diffusion.time_embedding_dim,
        )

        # Confidence assessment
        self.confidence_head = ConfidenceHead(
            single_dim=config.pairformer.single_dim,
            pair_dim=config.pairformer.pair_dim,
            plddt_bins=config.confidence.plddt_bins,
            ptm_bins=config.confidence.num_bins_distogram,
            max_dist=config.confidence.max_dist,
        )

        # Auxiliary heads
        self.distogram_head = DistogramHead(
            pair_dim=config.pairformer.pair_dim,
            num_bins=config.confidence.num_bins_distogram,
            max_dist=config.confidence.max_dist,
        )
        self.experimentally_resolved_head = ExperimentallyResolvedHead(
            single_dim=config.pairformer.single_dim
        )
        self.masked_msa_head = MaskedMSAHead(
            msa_dim=config.msa.msa_embedding_dim
        )
        self.secondary_structure_head = SecondaryStructureHead(
            single_dim=config.pairformer.single_dim
        )
        self.solvent_accessibility_head = SolventAccessibilityHead(
            single_dim=config.pairformer.single_dim
        )

    def forward(
        self,
        token_ids: Tensor,
        token_type: Tensor,
        residue_index: Tensor,
        chain_index: Tensor,
        msa_tokens: Optional[Tensor] = None,
        has_deletion: Optional[Tensor] = None,
        deletion_value: Optional[Tensor] = None,
        msa_mask: Optional[Tensor] = None,
        template_pair_feat: Optional[Tensor] = None,
        coords_noisy: Optional[Tensor] = None,
        sigma: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        interface_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Forward pass for training or inference.

        If coords_noisy and sigma are provided, returns diffusion denoising prediction.
        Otherwise, performs full iterative sampling to generate 3D coordinates.
        """
        B, N = token_ids.shape
        device = token_ids.device

        # 1. Base input embedding
        single, pair = self.embedding(token_ids, token_type, residue_index, chain_index)

        # 2. Add template features if available
        if template_pair_feat is not None:
            pair = pair + self.template_features(template_pair_feat)

        # 3. Setup MSA representations
        if msa_tokens is None:
            # Fallback default: single sequence msa from token_ids
            msa_tokens = token_ids.unsqueeze(1)  # (B, 1, N)
            has_deletion = torch.zeros(B, 1, N, device=device)
            deletion_value = torch.zeros(B, 1, N, device=device)
            msa_mask = torch.ones(B, 1, N, dtype=torch.bool, device=device)

        if has_deletion is None:
            has_deletion = torch.zeros_like(msa_tokens, dtype=torch.float)
        if deletion_value is None:
            deletion_value = torch.zeros_like(msa_tokens, dtype=torch.float)

        msa_feat = self.msa_features(msa_tokens, has_deletion, deletion_value)

        # 4. MSA module (alternating MSA row/col attention + outer product mean updates)
        msa_feat, pair = self.msa_module(msa_feat, pair, msa_mask)

        # 5. Pairformer (triangular updates + single track attention)
        pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2) if mask is not None else None
        pair, single = self.pairformer(pair, single, pair_mask=pair_mask, single_mask=mask)

        # 6. Diffusion structure prediction
        outputs = {}
        if coords_noisy is not None and sigma is not None:
            # Training mode: single step score matching (denoising prediction)
            pred_coords = self.diffusion(single, pair, coords_noisy, sigma, mask)
            outputs["pred_coords"] = pred_coords
        else:
            # Inference mode: full iterative sampling
            sampled_coords = self.diffusion.sample(single, pair, mask)
            outputs["pred_coords"] = sampled_coords

        # 7. Confidence scores prediction
        conf_outputs = self.confidence_head(single, pair, mask, interface_mask)
        outputs.update(conf_outputs)

        # 8. Auxiliary heads prediction
        outputs["distogram_logits"] = self.distogram_head(pair)
        outputs["resolved_logits"] = self.experimentally_resolved_head(single)
        outputs["masked_msa_logits"] = self.masked_msa_head(msa_feat)
        outputs["secondary_structure_logits"] = self.secondary_structure_head(single)
        outputs["solvent_accessibility"] = self.solvent_accessibility_head(single)

        return outputs
