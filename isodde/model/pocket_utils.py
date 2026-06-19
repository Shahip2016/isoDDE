"""Pocket evaluation and validation utilities for IsoDDE.

Includes P2Rank-style comparisons, AUPRC calculation, and pocket similarity metrics.
"""

from __future__ import annotations

import math
from typing import List, Dict, Any, Tuple
import torch
from torch import Tensor


def compute_auprc(preds: Tensor, targets: Tensor) -> float:
    """Compute Area Under Precision-Recall Curve (AUPRC).

    Parameters
    ----------
    preds : Tensor (N,)
        Predicted probabilities.
    targets : Tensor (N,)
        Ground truth binary labels (0 or 1).

    Returns
    -------
    float
        AUPRC score.
    """
    preds = preds.detach().cpu().view(-1)
    targets = targets.detach().cpu().view(-1)

    # Sort by prediction probability descending
    sorted_indices = torch.argsort(preds, descending=True)
    sorted_targets = targets[sorted_indices]

    total_positives = int(targets.sum().item())
    if total_positives == 0:
        return 0.0

    # Running counts of true positives and false positives
    tp = torch.cumsum(sorted_targets, dim=0)
    fp = torch.cumsum(1.0 - sorted_targets, dim=0)

    precisions = tp / (tp + fp + 1e-10)
    recalls = tp / total_positives

    # Compute AUPRC using trapezoidal integration
    auprc = 0.0
    prev_recall = 0.0
    for i in range(len(recalls)):
        curr_recall = float(recalls[i].item())
        curr_precision = float(precisions[i].item())
        recall_diff = curr_recall - prev_recall
        if recall_diff > 0:
            auprc += curr_precision * recall_diff
            prev_recall = curr_recall

    return auprc


def p2rank_success_rate(
    predicted_pocket_centers: List[List[float]],
    true_ligand_centers: List[List[float]],
    distance_threshold_angstrom: float = 4.0,
) -> float:
    """Evaluate pocket prediction using P2Rank distance success rate.

    A predicted pocket is a hit if its center is within the threshold distance of
    any true ligand center.

    Parameters
    ----------
    predicted_pocket_centers : List of predicted centers [x, y, z]
    true_ligand_centers : List of true ligand centers [x, y, z]
    distance_threshold_angstrom : float

    Returns
    -------
    float
        Success rate (hits / total true ligands).
    """
    if not true_ligand_centers:
        return 0.0
    if not predicted_pocket_centers:
        return 0.0

    hits = 0
    for true_center in true_ligand_centers:
        # Check distance to closest predicted pocket center
        min_dist = float("inf")
        for pred_center in predicted_pocket_centers:
            dist = math.sqrt(sum((t - p) ** 2 for t, p in zip(true_center, pred_center)))
            if dist < min_dist:
                min_dist = dist
        
        if min_dist <= distance_threshold_angstrom:
            hits += 1

    return hits / len(true_ligand_centers)


def pocket_intersection_over_union(
    pred_residues: Set[int],
    true_residues: Set[int],
) -> float:
    """Calculate intersection over union (IoU) of pocket residue sets."""
    intersection = len(pred_residues.intersection(true_residues))
    union = len(pred_residues.union(true_residues))
    if union == 0:
        return 0.0
    return intersection / union
