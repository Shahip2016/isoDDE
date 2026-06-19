"""Geometry primitives for biomolecular structure prediction.

Provides rigid-body transforms, quaternion operations, frame-aligned point
error (FAPE), and coordinate conversion utilities used throughout IsoDDE.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Quaternion operations
# ---------------------------------------------------------------------------

def quaternion_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product of two quaternion tensors (..., 4)."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def quaternion_conjugate(q: Tensor) -> Tensor:
    """Conjugate of quaternion tensor (..., 4)."""
    return q * torch.tensor([1, -1, -1, -1], dtype=q.dtype, device=q.device)


def quaternion_to_rotation_matrix(q: Tensor) -> Tensor:
    """Convert unit quaternion (..., 4) to rotation matrix (..., 3, 3)."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).unflatten(-1, (3, 3))


def rotation_matrix_to_quaternion(R: Tensor) -> Tensor:
    """Convert rotation matrix (..., 3, 3) to unit quaternion (..., 4).

    Uses Shepperd's method for numerical stability.
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Four cases for numerical stability
    q = torch.zeros(R.shape[0], 4, dtype=R.dtype, device=R.device)

    # Case 1: trace > 0
    mask = trace > 0
    s = torch.sqrt(trace[mask] + 1.0) * 2
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s

    # Case 2: R[0,0] is largest diagonal
    mask2 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
    q[mask2, 1] = 0.25 * s
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

    # Case 3: R[1,1] is largest diagonal
    mask3 = (~mask) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
    q[mask3, 2] = 0.25 * s
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

    # Case 4: R[2,2] is largest diagonal
    mask4 = (~mask) & (~mask2) & (~mask3)
    s = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
    q[mask4, 3] = 0.25 * s

    q = F.normalize(q, dim=-1)
    return q.reshape(*batch_shape, 4)


# ---------------------------------------------------------------------------
# Rigid body transforms
# ---------------------------------------------------------------------------

class Rigids:
    """Rigid-body transformation (rotation + translation).

    Stores rotation as a unit quaternion for memory efficiency and
    numerical stability during composition.

    Parameters
    ----------
    quats : Tensor (..., 4)
        Unit quaternions representing rotations.
    trans : Tensor (..., 3)
        Translation vectors.
    """

    def __init__(self, quats: Tensor, trans: Tensor) -> None:
        self.quats = quats
        self.trans = trans

    @classmethod
    def identity(cls, shape: tuple, dtype: torch.dtype = torch.float32,
                 device: torch.device | str = "cpu") -> "Rigids":
        """Create identity transforms with the given batch shape."""
        quats = torch.zeros(*shape, 4, dtype=dtype, device=device)
        quats[..., 0] = 1.0
        trans = torch.zeros(*shape, 3, dtype=dtype, device=device)
        return cls(quats, trans)

    @classmethod
    def from_rotation_and_translation(
        cls, rotation: Tensor, translation: Tensor
    ) -> "Rigids":
        """Create from rotation matrix (..., 3, 3) and translation (..., 3)."""
        quats = rotation_matrix_to_quaternion(rotation)
        return cls(quats, translation)

    @property
    def rotation_matrix(self) -> Tensor:
        """Return rotation as a 3×3 matrix."""
        return quaternion_to_rotation_matrix(self.quats)

    def apply(self, points: Tensor) -> Tensor:
        """Apply rigid transform to points (..., 3)."""
        R = self.rotation_matrix
        return torch.einsum("...ij,...j->...i", R, points) + self.trans

    def compose(self, other: "Rigids") -> "Rigids":
        """Compose two rigid transforms: self ∘ other."""
        new_quats = quaternion_multiply(self.quats, other.quats)
        new_trans = self.apply(other.trans)
        return Rigids(new_quats, new_trans)

    def inverse(self) -> "Rigids":
        """Return the inverse transform."""
        inv_quats = quaternion_conjugate(self.quats)
        inv_R = quaternion_to_rotation_matrix(inv_quats)
        inv_trans = -torch.einsum("...ij,...j->...i", inv_R, self.trans)
        return Rigids(inv_quats, inv_trans)

    def to(self, device: torch.device | str) -> "Rigids":
        """Move to device."""
        return Rigids(self.quats.to(device), self.trans.to(device))


# ---------------------------------------------------------------------------
# Frame Aligned Point Error (FAPE)
# ---------------------------------------------------------------------------

def compute_fape(
    pred_frames: Rigids,
    target_frames: Rigids,
    pred_positions: Tensor,
    target_positions: Tensor,
    mask: Tensor,
    pair_mask: Tensor | None = None,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    epsilon: float = 1e-8,
) -> Tensor:
    """Compute Frame Aligned Point Error.

    Measures the error between predicted and target atom positions
    expressed in local coordinate frames, enabling SE(3)-invariant
    structure comparison.

    Parameters
    ----------
    pred_frames, target_frames : Rigids (..., N_frames)
        Local coordinate frames for each residue.
    pred_positions, target_positions : Tensor (..., N_atoms, 3)
        Atom coordinates.
    mask : Tensor (..., N_atoms)
        Mask for valid atoms.
    pair_mask : Tensor | None (..., N_frames, N_atoms)
        Optional pairwise mask.
    clamp_distance : float
        Distance clamping threshold in Angstroms.
    length_scale : float
        Normalisation length scale.
    epsilon : float
        Small constant for numerical stability.

    Returns
    -------
    Tensor
        Scalar FAPE loss.
    """
    # Express positions in each frame's local coordinate system
    inv_frames = target_frames.inverse()

    # (..., N_frames, N_atoms, 3)
    local_pred = inv_frames.apply(pred_positions.unsqueeze(-3))
    local_target = target_frames.inverse().apply(target_positions.unsqueeze(-3))

    # Compute per-pair distances
    error = torch.sqrt(
        ((local_pred - local_target) ** 2).sum(dim=-1) + epsilon
    )

    # Clamp
    error = torch.clamp(error, max=clamp_distance)

    # Normalise
    error = error / length_scale

    # Apply masks
    if pair_mask is not None:
        error = error * pair_mask
        return (error.sum(dim=(-1, -2)) /
                (pair_mask.sum(dim=(-1, -2)) + epsilon)).mean()

    error = error * mask.unsqueeze(-2)
    n_atoms = mask.sum(dim=-1, keepdim=True).unsqueeze(-2).clamp(min=1)
    return (error.sum(dim=-1) / n_atoms.squeeze(-2)).mean()


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def compute_backbone_frames(
    n_coords: Tensor,
    ca_coords: Tensor,
    c_coords: Tensor,
) -> Rigids:
    """Compute backbone local frames from N, Cα, C atom positions.

    Follows the Gram-Schmidt procedure to construct an orthonormal
    frame for each residue.

    Parameters
    ----------
    n_coords, ca_coords, c_coords : Tensor (..., 3)
        Backbone atom positions.

    Returns
    -------
    Rigids
        Per-residue local frames.
    """
    # Vectors from Cα
    v1 = c_coords - ca_coords
    v2 = n_coords - ca_coords

    # Gram-Schmidt orthogonalisation
    e1 = F.normalize(v1, dim=-1)
    u2 = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)

    # Rotation matrix: columns are [e1, e2, e3]
    rotation = torch.stack([e1, e2, e3], dim=-1)

    return Rigids.from_rotation_and_translation(rotation, ca_coords)


def compute_dihedral_angles(
    p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor
) -> Tensor:
    """Compute dihedral angle between four points.

    Parameters
    ----------
    p0, p1, p2, p3 : Tensor (..., 3)
        Four consecutive atom positions.

    Returns
    -------
    Tensor (...)
        Dihedral angles in radians.
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    n1 = F.normalize(n1, dim=-1)
    n2 = F.normalize(n2, dim=-1)

    b2_norm = F.normalize(b2, dim=-1)
    m1 = torch.cross(n1, b2_norm, dim=-1)

    x = (n1 * n2).sum(dim=-1)
    y = (m1 * n2).sum(dim=-1)

    return torch.atan2(y, x)


def pairwise_distances(coords: Tensor, mask: Tensor | None = None) -> Tensor:
    """Compute pairwise Euclidean distance matrix.

    Parameters
    ----------
    coords : Tensor (..., N, 3)
    mask : Tensor | None (..., N)

    Returns
    -------
    Tensor (..., N, N)
    """
    diff = coords.unsqueeze(-2) - coords.unsqueeze(-3)
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
    if mask is not None:
        pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        dist = dist * pair_mask
    return dist
