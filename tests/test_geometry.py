"""Unit tests for IsoDDE geometry utilities."""

from __future__ import annotations

import torch
import pytest

from isodde.utils.geometry import (
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    Rigids,
    compute_fape,
    compute_backbone_frames,
    compute_dihedral_angles,
)


def test_quaternions():
    # Identity quaternions
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q2 = torch.tensor([0.0, 1.0, 0.0, 0.0])

    # Hamilton product with identity should be itself
    prod = quaternion_multiply(q1, q2)
    assert torch.allclose(prod, q2)

    # Conjugate
    q2_conj = quaternion_conjugate(q2)
    assert torch.allclose(q2_conj, torch.tensor([0.0, -1.0, -0.0, -0.0]))


def test_quaternion_rotations():
    # 90 degrees rotation around Z axis
    angle = torch.tensor(3.14159265 / 4.0)  # half angle
    q = torch.tensor([torch.cos(angle), 0.0, 0.0, torch.sin(angle)])

    R = quaternion_to_rotation_matrix(q)
    assert R.shape == (3, 3)

    # Roundtrip conversion
    q_back = rotation_matrix_to_quaternion(R)
    # Check orientation matches (signs can be flipped for same rotation, check absolute dot product)
    assert torch.allclose(torch.abs(torch.dot(q, q_back)), torch.tensor(1.0), atol=1e-5)


def test_rigids():
    # Create identity rigids
    r1 = Rigids.identity((2, 5))
    assert r1.quats.shape == (2, 5, 4)
    assert r1.trans.shape == (2, 5, 3)

    # Apply rigid transforms
    pts = torch.randn(2, 5, 3)
    pts_applied = r1.apply(pts)
    assert torch.allclose(pts, pts_applied)

    # Compose rigids
    R = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(2, 5, -1, -1)
    trans = torch.ones(2, 5, 3)
    r2 = Rigids.from_rotation_and_translation(R, trans)

    pts_r2 = r2.apply(pts)
    assert torch.allclose(pts_r2, pts + 1.0)


def test_backbone_frames():
    # Mock orthogonal positions
    n_coords = torch.tensor([[1.0, 0.0, 0.0]])
    ca_coords = torch.tensor([[0.0, 0.0, 0.0]])
    c_coords = torch.tensor([[0.0, 1.0, 0.0]])

    frames = compute_backbone_frames(n_coords, ca_coords, c_coords)
    assert frames.quats.shape == (1, 4)
    assert torch.allclose(frames.trans, ca_coords)
