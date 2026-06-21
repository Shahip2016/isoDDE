"""Binding affinity loss function for IsoDDE.

Implements Pearson correlation-aware loss and per-assay normalization strategy
described in Section 2.2-2.3 of the paper.
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor


class PearsonCorrelationLoss(nn.Module):
    """Loss module based on negative Pearson Correlation Coefficient.

    Encourages predicted and actual affinity values to be linearly correlated.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute negative Pearson correlation coefficient loss.

        Parameters
        ----------
        x : Tensor (B,) or (B, 1)
            Predictions.
        y : Tensor (B,) or (B, 1)
            Targets.

        Returns
        -------
        Tensor (1,)
            Loss value in [0, 2] (1 - correlation).
        """
        x = x.view(-1)
        y = y.view(-1)

        mean_x = torch.mean(x)
        mean_y = torch.mean(y)

        xm = x - mean_x
        ym = y - mean_y

        r_num = torch.sum(xm * ym)
        r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))

        r = r_num / (r_den + self.eps)
        return 1.0 - r


class PairwiseRankingLoss(nn.Module):
    """Pairwise ranking loss module.

    Encourages predicted values to preserve the relative ranking order of targets.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute ranking loss.

        Parameters
        ----------
        x : Tensor (B,)
            Predictions.
        y : Tensor (B,)
            Targets.

        Returns
        -------
        Tensor (1,)
            Scalar loss.
        """
        x = x.view(-1)
        y = y.view(-1)
        n = x.size(0)
        if n < 2:
            return torch.tensor(0.0, device=x.device)

        # Compute pairwise differences
        diff_x = x.unsqueeze(1) - x.unsqueeze(0)  # (N, N)
        diff_y = y.unsqueeze(1) - y.unsqueeze(0)  # (N, N)

        # Label is the sign of actual difference
        sign_y = torch.sign(diff_y)

        # Calculate margin ranking loss for all pairs
        pair_loss = torch.clamp(self.margin - sign_y * diff_x, min=0.0)

        # Only average over valid pairs
        mask = (sign_y != 0).float()
        return (pair_loss * mask).sum() / (mask.sum() + 1e-8)


class SoftSpearmanLoss(nn.Module):
    """Loss module based on negative soft Spearman rank correlation coefficient."""

    def __init__(self, temp: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.temp = temp
        self.eps = eps
        self.pearson = PearsonCorrelationLoss(eps=eps)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute negative soft Spearman correlation loss.

        Parameters
        ----------
        x : Tensor (B,)
            Predictions.
        y : Tensor (B,)
            Targets.

        Returns
        -------
        Tensor (1,)
            Loss value in [0, 2] (1 - soft Spearman correlation).
        """
        x = x.view(-1)
        y = y.view(-1)
        n = x.size(0)
        if n < 3:
            return self.pearson(x, y)

        # Compute soft ranks for x: rank_i = 1.0 + sum_j sigmoid((x_i - x_j)/temp) - 0.5
        diff_x = (x.unsqueeze(1) - x.unsqueeze(0)) / self.temp
        soft_ranks_x = 1.0 + torch.sigmoid(diff_x).sum(dim=1) - 0.5

        # Compute soft ranks for y
        diff_y = (y.unsqueeze(1) - y.unsqueeze(0)) / self.temp
        soft_ranks_y = 1.0 + torch.sigmoid(diff_y).sum(dim=1) - 0.5

        return self.pearson(soft_ranks_x, soft_ranks_y)


class IsoDDEAffinityLoss(nn.Module):
    """Composite binding affinity training loss.

    Combines Mean Squared Error (MSE) with Pearson Correlation Loss,
    Pairwise Ranking Loss, and Soft Spearman Loss, with support for
    per-assay grouping/normalization.
    """

    def __init__(
        self,
        mse_weight: float = 0.5,
        pearson_weight: float = 0.5,
        ranking_weight: float = 0.0,
        spearman_weight: float = 0.0,
        normalize_per_assay: bool = True,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.pearson_weight = pearson_weight
        self.ranking_weight = ranking_weight
        self.spearman_weight = spearman_weight
        self.normalize_per_assay = normalize_per_assay

        self.mse = nn.MSELoss()
        self.pearson = PearsonCorrelationLoss()
        self.ranking = PairwiseRankingLoss()
        self.spearman = SoftSpearmanLoss()

    def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        assay_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Calculate loss.

        Parameters
        ----------
        preds : Tensor (B, 1) or (B,)
            Predicted affinity values.
        targets : Tensor (B, 1) or (B,)
            True affinity values.
        assay_ids : Tensor (B,), optional
            Integer identifiers of the source assay/dataset type.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        preds = preds.view(-1)
        targets = targets.view(-1)

        if not self.normalize_per_assay or assay_ids is None:
            # Global loss computation
            loss = torch.tensor(0.0, device=preds.device)
            if self.mse_weight > 0:
                loss = loss + self.mse_weight * self.mse(preds, targets)
            if self.pearson_weight > 0:
                loss = loss + self.pearson_weight * self.pearson(preds, targets)
            if self.ranking_weight > 0:
                loss = loss + self.ranking_weight * self.ranking(preds, targets)
            if self.spearman_weight > 0:
                loss = loss + self.spearman_weight * self.spearman(preds, targets)
            return loss

        # Per-assay normalization and loss aggregation
        unique_assays = torch.unique(assay_ids)
        total_loss = torch.tensor(0.0, device=preds.device)
        count = 0

        for assay in unique_assays:
            mask = (assay_ids == assay)
            sub_preds = preds[mask]
            sub_targets = targets[mask]

            # We need at least 3 samples to calculate correlation stably
            if len(sub_preds) >= 3:
                # Normalize targets and predictions within the assay
                p_mean, p_std = sub_preds.mean(), sub_preds.std().clamp(min=1e-5)
                t_mean, t_std = sub_targets.mean(), sub_targets.std().clamp(min=1e-5)

                norm_preds = (sub_preds - p_mean) / p_std
                norm_targets = (sub_targets - t_mean) / t_std

                loss_val = torch.tensor(0.0, device=preds.device)
                if self.mse_weight > 0:
                    loss_val = loss_val + self.mse_weight * self.mse(norm_preds, norm_targets)
                if self.pearson_weight > 0:
                    loss_val = loss_val + self.pearson_weight * self.pearson(sub_preds, sub_targets)
                if self.ranking_weight > 0:
                    loss_val = loss_val + self.ranking_weight * self.ranking(sub_preds, sub_targets)
                if self.spearman_weight > 0:
                    loss_val = loss_val + self.spearman_weight * self.spearman(sub_preds, sub_targets)

                total_loss = total_loss + loss_val
                count += 1
            elif len(sub_preds) > 0:
                # Fallback to pure MSE if not enough samples for correlation
                total_loss = total_loss + self.mse(sub_preds, sub_targets)
                count += 1

        if count > 0:
            return total_loss / count
        else:
            return self.mse(preds, targets)
