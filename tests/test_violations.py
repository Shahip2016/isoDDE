"""Unit tests for stereochemical violation detection functions."""

from __future__ import annotations

import torch
import pytest

from isodde.evaluation.violations import check_bond_lengths, check_clashes, check_chirality


def test_check_bond_lengths():
    # Two carbons
    coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.54, 0.0, 0.0],  # Single bond C-C ideal is 1.54
    ])
    elements = ["C", "C"]
    bonds = [(0, 1, "SINGLE")]

    # 1. No violation
    res = check_bond_lengths(coords, bonds, elements, threshold=0.1)
    assert res["num_violations"] == 0
    assert res["max_deviation"] < 0.01

    # 2. Significant violation
    coords_violated = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.10, 0.0, 0.0],
    ])
    res2 = check_bond_lengths(coords_violated, bonds, elements, threshold=0.1)
    assert res2["num_violations"] == 1
    assert pytest.approx(res2["max_deviation"], abs=1e-2) == 0.56  # 2.10 - 1.54


def test_check_clashes():
    # Distant carbons (no clashes)
    coords = torch.tensor([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ])
    elements = ["C", "C"]
    bonds = []
    is_ligand = torch.tensor([True, True])

    res = check_clashes(coords, elements, bonds, is_ligand, clash_intra_threshold=1.5)
    assert res["num_intra_clashes"] == 0

    # Overlapping carbons (clash)
    coords_clash = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    res2 = check_clashes(coords_clash, elements, bonds, is_ligand, clash_intra_threshold=1.5)
    assert res2["num_intra_clashes"] == 1


def test_check_chirality():
    # Simple tetrahedron
    coords = torch.tensor([
        [0.0, 0.0, 0.0],   # center
        [1.0, 0.0, 0.0],   # a
        [0.0, 1.0, 0.0],   # b
        [0.0, 0.0, 1.0],   # c
    ])

    # Chiral centers list: (center, a, b, c, expected_sign)
    # The triple product of (a-center) x (b-center) . (c-center)
    # is [1,0,0] x [0,1,0] . [0,0,1] = [0,0,1] . [0,0,1] = 1.0 > 0 ("+")
    centers = [(0, 1, 2, 3, "+")]

    res = check_chirality(coords, centers)
    assert res["num_violations"] == 0

    # Mismatched sign
    centers_mismatch = [(0, 1, 2, 3, "-")]
    res2 = check_chirality(coords, centers_mismatch)
    assert res2["num_violations"] == 1
