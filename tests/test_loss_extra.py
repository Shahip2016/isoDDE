"""Extra unit tests for affinity loss module."""

from __future__ import annotations

import torch
from isodde.model.affinity_loss import (
    PearsonCorrelationLoss,
    PairwiseRankingLoss,
    SoftSpearmanLoss,
    IsoDDEAffinityLoss,
)


def test_pairwise_ranking_loss():
    loss_fn = PairwiseRankingLoss(margin=1.0)

    # 1. Perfectly ranked predictions
    x_good = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    y = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    loss_good = loss_fn(x_good, y)
    
    # 2. Incorrectly ranked predictions
    x_bad = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
    loss_bad = loss_fn(x_bad, y)

    assert loss_good < loss_bad
    assert loss_good >= 0.0


def test_soft_spearman_loss():
    loss_fn = SoftSpearmanLoss(temp=0.1)

    # Perfectly correlated
    x_good = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    y = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    loss_good = loss_fn(x_good, y)

    # Inversely correlated
    x_bad = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
    loss_bad = loss_fn(x_bad, y)

    assert loss_good < 1.0  # highly correlated soft Spearman loss should be close to 0
    assert loss_bad > 1.0   # anti-correlated should be close to 2.0


def test_composite_affinity_loss():
    # Test that we can instantiate and compute composite loss
    loss_fn = IsoDDEAffinityLoss(
        mse_weight=0.25,
        pearson_weight=0.25,
        ranking_weight=0.25,
        spearman_weight=0.25,
        normalize_per_assay=True,
    )

    preds = torch.tensor([1.2, 2.1, 3.4, 0.8, 1.5, 2.9], dtype=torch.float32)
    targets = torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    assay_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    loss_val = loss_fn(preds, targets, assay_ids)
    assert loss_val.ndim == 0
    assert not torch.isnan(loss_val)
