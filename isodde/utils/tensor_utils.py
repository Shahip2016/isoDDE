"""Tensor manipulation utilities for IsoDDE.

Provides batched operations, masking helpers, and chunk-processing
routines for memory-efficient computation of O(n³) operations.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor


def batched_index_select(
    values: Tensor, indices: Tensor, dim: int = 1
) -> Tensor:
    """Batched version of torch.index_select along a given dimension.

    Parameters
    ----------
    values : Tensor (B, N, ...)
        Source tensor.
    indices : Tensor (B, M)
        Indices to select.
    dim : int
        Dimension to select along.

    Returns
    -------
    Tensor (B, M, ...)
    """
    assert dim == 1, "Only dim=1 is supported"
    batch_size = values.shape[0]
    idx = indices + torch.arange(
        batch_size, device=indices.device
    ).unsqueeze(1) * values.shape[1]
    flat = values.reshape(-1, *values.shape[2:])
    return flat[idx.reshape(-1)].reshape(batch_size, -1, *values.shape[2:])


def mask_mean(
    values: Tensor, mask: Tensor, dim: int | tuple = -1, eps: float = 1e-8
) -> Tensor:
    """Compute masked mean along specified dimensions.

    Parameters
    ----------
    values : Tensor
    mask : Tensor
        Binary mask (same shape as values or broadcastable).
    dim : int or tuple of ints
    eps : float
        Prevent division by zero.

    Returns
    -------
    Tensor
    """
    masked = values * mask
    return masked.sum(dim=dim) / (mask.sum(dim=dim) + eps)


def one_hot(indices: Tensor, num_classes: int) -> Tensor:
    """Create one-hot encoding.

    Parameters
    ----------
    indices : Tensor (...)
        Integer indices in [0, num_classes).
    num_classes : int

    Returns
    -------
    Tensor (..., num_classes)
    """
    return torch.nn.functional.one_hot(
        indices.long().clamp(0, num_classes - 1), num_classes
    ).float()


def chunk_layer(
    fn: Callable,
    inputs: dict[str, Tensor],
    chunk_size: int,
    dim: int = 0,
    extra_args: Optional[dict] = None,
) -> Tensor:
    """Apply a function in chunks for memory-efficient processing.

    Used for O(n³) triangular operations to reduce peak memory from
    O(n³) to O(n² · chunk_size) as described in Section 1.1.

    Parameters
    ----------
    fn : Callable
        Function to apply. Must accept keyword arguments matching `inputs`.
    inputs : dict[str, Tensor]
        Input tensors to be chunked along `dim`.
    chunk_size : int
        Number of elements per chunk.
    dim : int
        Dimension to chunk along.
    extra_args : dict, optional
        Additional non-chunked keyword arguments.

    Returns
    -------
    Tensor
        Concatenated output.
    """
    if extra_args is None:
        extra_args = {}

    # Get total size along chunk dimension
    sample_key = next(iter(inputs))
    total = inputs[sample_key].shape[dim]

    if chunk_size >= total:
        return fn(**inputs, **extra_args)

    outputs = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk_inputs = {
            k: v.narrow(dim, start, end - start)
            for k, v in inputs.items()
        }
        outputs.append(fn(**chunk_inputs, **extra_args))

    return torch.cat(outputs, dim=dim)


def add_to_pair(pair: Tensor, left: Tensor, right: Tensor) -> Tensor:
    """Add outer sum of single representations to pair representation.

    pair[i,j] += left[i] + right[j]

    Parameters
    ----------
    pair : Tensor (..., N, N, C)
    left, right : Tensor (..., N, C)

    Returns
    -------
    Tensor (..., N, N, C)
    """
    return pair + left.unsqueeze(-2) + right.unsqueeze(-3)


def permute_final_dims(tensor: Tensor, inds: tuple) -> Tensor:
    """Permute only the final dimensions of a tensor."""
    ndim = tensor.ndim
    first_inds = list(range(ndim - len(inds)))
    return tensor.permute(*first_inds, *[ndim - len(inds) + i for i in inds])


def flatten_final_dims(tensor: Tensor, n: int) -> Tensor:
    """Flatten the last n dimensions into one."""
    shape = tensor.shape
    return tensor.reshape(*shape[:-n], -1)


def rbf_encoding(
    distances: Tensor,
    d_min: float = 0.0,
    d_max: float = 22.0,
    num_rbf: int = 64,
) -> Tensor:
    """Radial basis function encoding for pairwise distances.

    Parameters
    ----------
    distances : Tensor (...)
        Input distances.
    d_min, d_max : float
        Range of centres.
    num_rbf : int
        Number of basis functions.

    Returns
    -------
    Tensor (..., num_rbf)
    """
    centers = torch.linspace(d_min, d_max, num_rbf, device=distances.device)
    width = (d_max - d_min) / num_rbf
    return torch.exp(-0.5 * ((distances.unsqueeze(-1) - centers) / width) ** 2)
