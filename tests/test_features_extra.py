"""Extra unit tests for features module."""

from __future__ import annotations

import torch
from isodde.data.features import LigandPhysicalEmbedding, InputEmbedding


def test_ligand_physical_embedding():
    # Instantiate embedding
    single_dim = 64
    embed = LigandPhysicalEmbedding(single_dim)

    # Mock inputs
    B, N = 2, 8
    # 3 = "N", 2 = "C", 4 = "O", 13 = UNK_AA_INDEX
    token_ids = torch.tensor([
        [3, 2, 4, 13, 0, 0, 0, 0],
        [2, 2, 2, 2, 0, 0, 0, 0]
    ], dtype=torch.long)
    
    is_ligand = torch.tensor([
        [True, True, True, False, False, False, False, False],
        [True, True, False, False, False, False, False, False]
    ], dtype=torch.bool)

    feat = embed(token_ids, is_ligand)

    # Assert shape: (B, N, single_dim)
    assert feat.shape == (B, N, single_dim)

    # Assert non-ligand tokens are zeroed out
    assert torch.all(feat[0, 3:] == 0)
    assert torch.all(feat[1, 2:] == 0)

    # Assert ligand tokens are non-zero (or at least calculated)
    assert torch.any(feat[0, :3] != 0)


def test_input_embedding_with_physical():
    single_dim = 128
    pair_dim = 192
    
    embed = InputEmbedding(
        num_token_types=6,
        num_residue_types=30,
        single_dim=single_dim,
        pair_dim=pair_dim
    )

    B, N = 2, 10
    token_ids = torch.randint(0, 20, (B, N))
    token_type = torch.zeros(B, N, dtype=torch.long)
    token_type[0, 5:] = 3  # Mark some as ligands
    token_type[1, 7:] = 3
    
    residue_index = torch.arange(N).unsqueeze(0).expand(B, -1)
    chain_index = torch.zeros(B, N, dtype=torch.long)

    single, pair = embed(token_ids, token_type, residue_index, chain_index)

    assert single.shape == (B, N, single_dim)
    assert pair.shape == (B, N, N, pair_dim)
