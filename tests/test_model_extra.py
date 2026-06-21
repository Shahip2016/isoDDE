"""Extra unit tests for model components."""

from __future__ import annotations

import torch
from isodde.model.triangular_multiplication import DualTriangularMultiplication
from isodde.model.heads import SecondaryStructureHead, SolventAccessibilityHead
from isodde.config import IsoDDEConfig
from isodde.model.isodde_structure import IsoDDEStructurePrediction


def test_dual_triangular_multiplication():
    pair_dim = 64
    inner_dim = 32
    module = DualTriangularMultiplication(pair_dim, inner_dim)

    B, N = 2, 8
    pair = torch.randn(B, N, N, pair_dim)
    pair_mask = torch.ones(B, N, N, dtype=torch.float32)

    out = module(pair, pair_mask)
    assert out.shape == (B, N, N, pair_dim)


def test_secondary_structure_head():
    single_dim = 128
    head = SecondaryStructureHead(single_dim)

    B, N = 2, 16
    single = torch.randn(B, N, single_dim)
    logits = head(single)

    assert logits.shape == (B, N, 3)


def test_solvent_accessibility_head():
    single_dim = 128
    head = SolventAccessibilityHead(single_dim)

    B, N = 2, 16
    single = torch.randn(B, N, single_dim)
    rsa = head(single)

    assert rsa.shape == (B, N)


def test_model_forward_with_new_heads():
    config = IsoDDEConfig.small()
    model = IsoDDEStructurePrediction(config)

    B, N = 2, 12
    token_ids = torch.randint(0, 10, (B, N))
    token_type = torch.randint(0, 4, (B, N))
    residue_index = torch.arange(N).unsqueeze(0).expand(B, -1)
    chain_index = torch.zeros(B, N, dtype=torch.long)
    mask = torch.ones(B, N, dtype=torch.bool)
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

    assert "secondary_structure_logits" in outputs
    assert "solvent_accessibility" in outputs
    assert outputs["secondary_structure_logits"].shape == (B, N, 3)
    assert outputs["solvent_accessibility"].shape == (B, N)
