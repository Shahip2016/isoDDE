"""Unit tests for IsoDDE evaluation metrics."""

from __future__ import annotations

import torch
import pytest

from isodde.evaluation.metrics import (
    compute_rmsd,
    compute_aligned_rmsd,
    compute_lddt,
    compute_pocket_aligned_rmsd,
)


def test_rmsd_identical():
    coords1 = torch.randn(10, 3)
    # Identical coords should yield 0 RMSD
    val = compute_rmsd(coords1, coords1)
    assert pytest.approx(val, abs=1e-5) == 0.0


def test_rmsd_translated():
    coords1 = torch.zeros(5, 3)
    coords2 = torch.ones(5, 3)  # Shifted by 1 in all axes
    # distance between point is sqrt(3). RMSD should be sqrt(3) ~ 1.73205
    val = compute_rmsd(coords1, coords2)
    assert pytest.approx(val, rel=1e-3) == 1.73205


def test_aligned_rmsd():
    coords1 = torch.randn(8, 3)
    # Rotate and translate coords1
    R = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])  # 90 deg rotation around Z
    t = torch.tensor([[5.0, -3.0, 2.0]])
    coords2 = torch.matmul(coords1, R.t()) + t

    # Unaligned RMSD will be large
    unaligned = compute_rmsd(coords1, coords2)
    assert unaligned > 1.0

    # Aligned RMSD should be 0.0
    aligned = compute_aligned_rmsd(coords1, coords2)
    assert pytest.approx(aligned, abs=1e-4) == 0.0


def test_lddt():
    coords1 = torch.randn(15, 3)
    # LDDT with itself should be 1.0
    val = compute_lddt(coords1, coords1)
    assert pytest.approx(val, abs=1e-5) == 1.0


def test_pocket_aligned_rmsd():
    # Coords of size 10: indices 0-4 are pocket, 5-9 are ligand
    coords1 = torch.randn(10, 3)
    
    # Translate ligand portion only
    coords2 = coords1.clone()
    coords2[5:] = coords2[5:] + 2.5

    pocket_indices = list(range(5))
    ligand_indices = list(range(5, 10))

    # Pocket-aligned ligand RMSD should be exactly 2.5 * sqrt(3) ~ 4.33013
    val = compute_pocket_aligned_rmsd(coords1, coords2, pocket_indices, ligand_indices)
    assert pytest.approx(val, rel=1e-3) == 4.33013
