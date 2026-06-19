"""Unit and integration tests for IsoDDE structure prediction model components."""

from __future__ import annotations

import torch
import pytest

from isodde.config import IsoDDEConfig
from isodde.model.isodde_structure import IsoDDEStructurePrediction


def test_model_forward():
    # 1. Setup small testing configuration
    config = IsoDDEConfig.small()
    
    # 2. Instantiate model
    model = IsoDDEStructurePrediction(config)

    # 3. Create mock input features
    B, N = 2, 16
    token_ids = torch.randint(0, 10, (B, N))
    token_type = torch.randint(0, 4, (B, N))
    residue_index = torch.arange(N).unsqueeze(0).expand(B, -1)
    chain_index = torch.zeros(B, N, dtype=torch.long)
    mask = torch.ones(B, N, dtype=torch.bool)

    # 4. Forward pass in training (denoising score matching) mode
    coords_noisy = torch.randn(B, N, 3)
    sigma = torch.tensor([0.5, 1.2])

    outputs = model(
        token_ids=token_ids,
        token_type=token_type,
        residue_index=residue_index,
        chain_index=chain_index,
        coords_noisy=coords_noisy,
        sigma=sigma,
        mask=mask,
    )

    # Validate output dictionary contents
    assert "pred_coords" in outputs
    assert "plddt" in outputs
    assert "ptm" in outputs
    assert "ranking_score" in outputs
    assert "distogram_logits" in outputs
    
    assert outputs["pred_coords"].shape == (B, N, 3)
    assert outputs["plddt"].shape == (B, N)
    assert outputs["ptm"].shape == (B,)
    assert outputs["distogram_logits"].shape == (B, N, N, config.confidence.num_bins_distogram)


def test_model_sampling():
    config = IsoDDEConfig.small()
    model = IsoDDEStructurePrediction(config)

    B, N = 1, 8
    token_ids = torch.randint(0, 10, (B, N))
    token_type = torch.randint(0, 4, (B, N))
    residue_index = torch.arange(N).unsqueeze(0).expand(B, -1)
    chain_index = torch.zeros(B, N, dtype=torch.long)
    mask = torch.ones(B, N, dtype=torch.bool)

    # Forward pass in inference/sampling mode (no coords_noisy or sigma)
    outputs = model(
        token_ids=token_ids,
        token_type=token_type,
        residue_index=residue_index,
        chain_index=chain_index,
        mask=mask,
    )

    assert "pred_coords" in outputs
    assert outputs["pred_coords"].shape == (B, N, 3)
