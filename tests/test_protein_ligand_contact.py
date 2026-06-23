"""Unit tests for Protein-Ligand Contact Predictor Head and Loss."""

from __future__ import annotations

import os
import tempfile
import torch
import pytest

from isodde.config import IsoDDEConfig, ProteinLigandContactConfig
from isodde.model.protein_ligand_contact import ProteinLigandContactHead
from isodde.model.protein_ligand_loss import ProteinLigandContactLoss
from isodde.model.isodde_structure import IsoDDEStructurePrediction
from isodde.pipeline import IsoDDEPipeline
from isodde.cli import main as cli_main


def test_protein_ligand_contact_config():
    config = ProteinLigandContactConfig(hidden_dim=128, num_layers=2)
    assert config.hidden_dim == 128
    assert config.num_layers == 2
    assert config.contact_threshold_angstrom == 4.5


def test_protein_ligand_contact_head():
    config = ProteinLigandContactConfig(hidden_dim=64, num_layers=2)
    pair_dim = 128
    head = ProteinLigandContactHead(config, pair_dim=pair_dim)

    B, N = 2, 10
    pair = torch.randn(B, N, N, pair_dim)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[0, 8:] = False

    head.eval()
    logits = head(pair, mask)
    assert logits.shape == (B, N, N)

    # Symmetry check: logits should be symmetric
    assert torch.allclose(logits, logits.transpose(-1, -2), atol=1e-5)

    # Masking check: elements corresponding to masked tokens should be -1e9
    assert torch.all(logits[0, 8:, :] < -1e8)
    assert torch.all(logits[0, :, 8:] < -1e8)


def test_protein_ligand_contact_loss():
    loss_fn = ProteinLigandContactLoss(contact_threshold_angstrom=4.5)

    B, N = 2, 8
    # token_type: 0-3 are protein (0), 4-7 are ligand (3)
    token_type = torch.tensor([[0, 0, 0, 0, 3, 3, 3, 3],
                               [0, 0, 0, 0, 3, 3, 3, 3]], dtype=torch.long)

    # Ground truth coords: let protein residue 0 be close to ligand atom 4 in batch 0, and far in batch 1
    coords_true = torch.zeros(B, N, 3)
    # protein residue 0
    coords_true[:, 0, :] = torch.tensor([0.0, 0.0, 0.0])
    # ligand atom 4
    coords_true[0, 4, :] = torch.tensor([1.0, 0.0, 0.0])  # contact: dist=1.0 < 4.5
    coords_true[1, 4, :] = torch.tensor([15.0, 0.0, 0.0]) # no contact: dist=15.0 > 4.5

    # Remaining coords spread out so no other contacts exist
    for i in range(N):
        if i != 0 and i != 4:
            coords_true[:, i, :] = torch.tensor([float(i) * 20.0, 0.0, 0.0])

    mask = torch.ones(B, N, dtype=torch.bool)

    # logits: batch 0 predicts contact between 0 and 4 (high logit), batch 1 predicts no contact (low logit)
    logits_good = torch.zeros(B, N, N)
    logits_good[0, 0, 4] = 5.0
    logits_good[0, 4, 0] = 5.0
    logits_good[1, 0, 4] = -5.0
    logits_good[1, 4, 0] = -5.0

    loss_good = loss_fn(logits_good, coords_true, token_type, mask)

    # bad logits (wrong prediction)
    logits_bad = torch.zeros(B, N, N)
    logits_bad[0, 0, 4] = -5.0
    logits_bad[0, 4, 0] = -5.0
    logits_bad[1, 0, 4] = 5.0
    logits_bad[1, 4, 0] = 5.0

    loss_bad = loss_fn(logits_bad, coords_true, token_type, mask)

    assert loss_good < loss_bad
    assert not torch.isnan(loss_good)


def test_model_forward_with_protein_ligand_contact():
    config = IsoDDEConfig.small()
    model = IsoDDEStructurePrediction(config)

    B, N = 2, 12
    token_ids = torch.randint(0, 10, (B, N))
    token_type = torch.randint(0, 4, (B, N))
    # make sure we have some protein (0) and ligand (3) tokens
    token_type[0, :6] = 0
    token_type[0, 6:] = 3
    token_type[1, :6] = 0
    token_type[1, 6:] = 3

    residue_index = torch.arange(N).unsqueeze(0).expand(B, -1)
    chain_index = torch.zeros(B, N, dtype=torch.long)
    mask = torch.ones(B, N, dtype=torch.bool)

    outputs = model(
        token_ids=token_ids,
        token_type=token_type,
        residue_index=residue_index,
        chain_index=chain_index,
        mask=mask,
    )

    assert "protein_ligand_contact_logits" in outputs
    assert outputs["protein_ligand_contact_logits"].shape == (B, N, N)


def test_pipeline_protein_ligand_contact():
    config = IsoDDEConfig.small()
    pipeline = IsoDDEPipeline(config)

    results = pipeline.run_cofolding(
        protein_sequence="MTEYKLVVVG",
        ligand_elements=["C", "C", "O"],
        num_seeds=2,
    )

    assert "protein_ligand_contact_probs" in results
    N_total = 10 + 3
    assert len(results["protein_ligand_contact_probs"]) == N_total
    assert len(results["protein_ligand_contact_probs"][0]) == N_total


def test_cli_protein_ligand_contact():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Run CLI in small config mode
        exit_code = cli_main([
            "--sequence", "MTEYKLVVVG",
            "--ligand", "C,C,O",
            "--output-dir", tmp_dir,
            "--num-seeds", "1",
            "--small"
        ])
        assert exit_code == 0

        # Check files were created
        pdb_path = os.path.join(tmp_dir, "predicted_complex.pdb")
        json_path = os.path.join(tmp_dir, "prediction_metrics.json")
        assert os.path.exists(pdb_path)
        assert os.path.exists(json_path)

        # Check JSON contains the new metrics
        import json
        with open(json_path, "r") as f:
            data = json.load(f)
        assert "protein_ligand_contact_probs" in data
        assert len(data["protein_ligand_contact_probs"]) == 13
        assert len(data["protein_ligand_contact_probs"][0]) == 13
