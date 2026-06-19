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


class IsoDDEAffinityLoss(nn.Module):
    """Composite binding affinity training loss.

    Combines Mean Squared Error (MSE) with Pearson Correlation Loss,
    with support for per-assay grouping/normalization.
    """

    def __init__(
        self,
        mse_weight: float = 0.5,
        pearson_weight: float = 0.5,
        normalize_per_assay: bool = True,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.pearson_weight = pearson_weight
        self.normalize_per_assay = normalize_per_assay

        self.mse = nn.MSELoss()
        self.pearson = PearsonCorrelationLoss()

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
            Integer identifiers of the source assay/dataset type
            (e.g., to distinguish Kd vs IC50 vs Ki).

        Returns
        -------
        Tensor
            Scalar loss.
        """
        preds = preds.view(-1)
        targets = targets.view(-1)

        if not self.normalize_per_assay or assay_ids is None:
            # Global loss computation
            loss_mse = self.mse(preds, targets)
            loss_pearson = self.pearson(preds, targets)
            return self.mse_weight * loss_mse + self.pearson_weight * loss_pearson

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

                loss_mse = self.mse(norm_preds, norm_targets)
                loss_pearson = self.pearson(sub_preds, sub_targets)

                total_loss = total_loss + (
                    self.mse_weight * loss_mse + self.pearson_weight * loss_pearson
                )
                count += 1
            elif len(sub_preds) > 0:
                # Fallback to pure MSE if not enough samples for correlation
                total_loss = total_loss + self.mse(sub_preds, sub_targets)
                count += 1

        if count > 0:
            return total_loss / count
        else:
            return self.mse(preds, targets)
