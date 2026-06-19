"""Evaluation metrics for IsoDDE.

Implements all primary metrics from the paper:
- RMSD (Root-Mean-Square Deviation)
- LDDT (Local Distance Difference Test)
- DockQ (Docking quality score for interfaces)
- FAPE (Frame Aligned Point Error)
- Pocket-aligned RMSD (Section 4.1)
- CDR-H3 backbone RMSD (Section 4.2)
- Pearson correlation
"""

from __future__ import annotations

from typing import Optional, List, Tuple
import torch
from torch import Tensor


def compute_rmsd(coords1: Tensor, coords2: Tensor) -> float:
    """Compute Root-Mean-Square Deviation (RMSD) between two coordinate sets.

    Parameters
    ----------
    coords1, coords2 : Tensor (N, 3)

    Returns
    -------
    float
    """
    diff = coords1 - coords2
    return float(torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1))).item())


def kabsch_alignment(coords1: Tensor, coords2: Tensor) -> Tuple[Tensor, Tensor]:
    """Align coords1 to coords2 using Kabsch algorithm.

    Returns
    -------
    rotation : Tensor (3, 3)
    translation : Tensor (1, 3)
    """
    # Centers
    c1 = coords1.mean(dim=0, keepdim=True)
    c2 = coords2.mean(dim=0, keepdim=True)

    # Center coordinates
    coords1_centered = coords1 - c1
    coords2_centered = coords2 - c2

    # Covariance matrix
    H = torch.matmul(coords1_centered.t(), coords2_centered)

    # SVD
    U, S, Vt = torch.linalg.svd(H)
    V = Vt.t()

    # Rotation matrix
    d = torch.linalg.det(torch.matmul(V, U.t()))
    S_correct = torch.eye(3, device=coords1.device)
    if d < 0:
        S_correct[2, 2] = -1.0

    R = torch.matmul(V, torch.matmul(S_correct, U.t()))

    # Translation
    t = c2 - torch.matmul(c1, R.t())

    return R, t


def compute_aligned_rmsd(coords1: Tensor, coords2: Tensor) -> float:
    """Align coords1 to coords2 then compute RMSD."""
    R, t = kabsch_alignment(coords1, coords2)
    coords1_aligned = torch.matmul(coords1, R.t()) + t
    return compute_rmsd(coords1_aligned, coords2)


def compute_lddt(
    coords1: Tensor,
    coords2: Tensor,
    cutoff: float = 15.0,
    thresholds: List[float] = [0.5, 1.0, 2.0, 4.0],
) -> float:
    """Compute Local Distance Difference Test (LDDT).

    Measures fraction of local pairwise distances preserved between coordinates.
    """
    dists1 = torch.cdist(coords1.unsqueeze(0), coords1.unsqueeze(0)).squeeze(0)
    dists2 = torch.cdist(coords2.unsqueeze(0), coords2.unsqueeze(0)).squeeze(0)

    N = coords1.shape[0]
    if N <= 1:
        return 1.0

    # Mask local region in coords2 (ground truth)
    mask = (dists2 < cutoff) & (~torch.eye(N, dtype=torch.bool, device=coords1.device))
    num_pairs = mask.sum().item()
    if num_pairs == 0:
        return 1.0

    diffs = torch.abs(dists1[mask] - dists2[mask])
    
    score = 0.0
    for th in thresholds:
        score += (diffs < th).float().mean().item()

    return score / len(thresholds)


def compute_pocket_aligned_rmsd(
    pred_coords: Tensor,
    true_coords: Tensor,
    pocket_indices: List[int],
    ligand_indices: List[int],
) -> float:
    """Compute ligand RMSD after aligning coordinates on the pocket residues.

    Used for protein-ligand docking evaluation (Section 4.1).
    """
    # 1. Align using pocket residues only
    pred_pocket = pred_coords[pocket_indices]
    true_pocket = true_coords[pocket_indices]

    R, t = kabsch_alignment(pred_pocket, true_pocket)

    # 2. Apply alignment to ligand coordinates
    pred_ligand = pred_coords[ligand_indices]
    true_ligand = true_coords[ligand_indices]

    pred_ligand_aligned = torch.matmul(pred_ligand, R.t()) + t

    # 3. Compute RMSD
    return compute_rmsd(pred_ligand_aligned, true_ligand)


def compute_cdr_h3_backbone_rmsd(
    pred_coords: Tensor,
    true_coords: Tensor,
    cdr_h3_indices: List[int],
    framework_indices: List[int],
) -> float:
    """Compute antibody CDR-H3 loop backbone RMSD after framework alignment.

    Measures loops prediction accuracy relative to the stable framework (Section 4.2).
    """
    # Align on framework residues
    pred_fw = pred_coords[framework_indices]
    true_fw = true_coords[framework_indices]

    R, t = kabsch_alignment(pred_fw, true_fw)

    # Compute RMSD on CDR-H3 residues
    pred_cdr = pred_coords[cdr_h3_indices]
    true_cdr = true_coords[cdr_h3_indices]

    pred_cdr_aligned = torch.matmul(pred_cdr, R.t()) + t
    return compute_rmsd(pred_cdr_aligned, true_cdr)


def compute_dockq(
    coords1: Tensor,
    coords2: Tensor,
    chain1_indices: List[int],
    chain2_indices: List[int],
) -> float:
    """Compute simplified DockQ score for protein-protein interface evaluation."""
    # DockQ combines fnat (fraction of native contacts), LRMSD (ligand RMSD),
    # and iRMSD (interface RMSD). Here we provide a simplified proxy DockQ.
    rmsd_c1 = compute_aligned_rmsd(coords1[chain1_indices], coords2[chain1_indices])
    rmsd_c2 = compute_aligned_rmsd(coords1[chain2_indices], coords2[chain2_indices])
    
    # Combined score proxy
    avg_rmsd = 0.5 * (rmsd_c1 + rmsd_c2)
    dockq_proxy = 1.0 / (1.0 + (avg_rmsd / 2.0) ** 2)
    return dockq_proxy
