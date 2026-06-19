"""Score-based diffusion module for 3D structure generation.

Generates atom-level 3D coordinates from trunk embeddings using a
denoising score matching approach. The diffusion model is conditioned
on single and pair representations from the Pairformer trunk.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from isodde.model.primitives import IsoLinear, Transition, Attention


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timestep.

    Maps scalar timestep to a high-dimensional embedding using
    sinusoidal functions at logarithmically spaced frequencies.

    Parameters
    ----------
    dim : int
        Output embedding dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """Embed timestep.

        Parameters
        ----------
        t : Tensor (B,) or (B, 1)
            Timestep values (noise level σ).

        Returns
        -------
        Tensor (B, dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device) / half_dim
        )
        args = t * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionConditioning(nn.Module):
    """Conditioning module for the diffusion network.

    Projects single and pair representations, along with timestep
    embedding, into conditioning signals for the denoising network.

    Parameters
    ----------
    single_dim : int
        Single representation dimension from trunk.
    pair_dim : int
        Pair representation dimension from trunk.
    atom_dim : int
        Atom-level representation dimension.
    time_dim : int
        Timestep embedding dimension.
    """

    def __init__(
        self,
        single_dim: int = 256,
        pair_dim: int = 384,
        atom_dim: int = 128,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Sequential(
            IsoLinear(time_dim, atom_dim),
            nn.SiLU(),
            IsoLinear(atom_dim, atom_dim),
        )

        self.single_proj = IsoLinear(single_dim, atom_dim)
        self.pair_proj = IsoLinear(pair_dim, atom_dim)

        self.norm = nn.LayerNorm(atom_dim)

    def forward(
        self,
        single: Tensor,
        pair: Tensor,
        sigma: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute conditioning signals.

        Parameters
        ----------
        single : Tensor (B, N, single_dim)
        pair : Tensor (B, N, N, pair_dim)
        sigma : Tensor (B,)
            Current noise level.

        Returns
        -------
        single_cond : Tensor (B, N, atom_dim)
        pair_cond : Tensor (B, N, N, atom_dim)
        """
        # Timestep conditioning added to single
        t_emb = self.time_proj(self.time_embed(sigma))  # (B, atom_dim)
        single_cond = self.single_proj(single) + t_emb.unsqueeze(1)
        single_cond = self.norm(single_cond)

        pair_cond = self.pair_proj(pair)

        return single_cond, pair_cond


class DenoisingBlock(nn.Module):
    """Single denoising block in the diffusion network.

    Updates noisy atom coordinates conditioned on trunk embeddings
    using attention over atom features with pair bias.

    Parameters
    ----------
    atom_dim : int
        Atom representation dimension.
    num_heads : int
        Attention heads.
    """

    def __init__(self, atom_dim: int = 128, num_heads: int = 8) -> None:
        super().__init__()
        # Atom self-attention with pair bias
        self.attention = Attention(
            q_dim=atom_dim,
            kv_dim=atom_dim,
            output_dim=atom_dim,
            num_heads=num_heads,
            gating=True,
            pair_bias=True,
            pair_dim=atom_dim,
        )

        # Coordinate update MLP
        self.coord_norm = nn.LayerNorm(atom_dim)
        self.coord_update = nn.Sequential(
            IsoLinear(atom_dim + 3, atom_dim),
            nn.SiLU(),
            IsoLinear(atom_dim, 3, init_zeros=True),
        )

        self.transition = Transition(atom_dim)

    def forward(
        self,
        atom_feat: Tensor,
        coords: Tensor,
        single_cond: Tensor,
        pair_cond: Tensor,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Apply denoising block.

        Parameters
        ----------
        atom_feat : Tensor (B, N, atom_dim)
        coords : Tensor (B, N, 3)
        single_cond : Tensor (B, N, atom_dim)
        pair_cond : Tensor (B, N, N, atom_dim)
        mask : Tensor (B, N), optional

        Returns
        -------
        atom_feat : Tensor (B, N, atom_dim)
        coord_update : Tensor (B, N, 3)
        """
        # Add conditioning
        h = atom_feat + single_cond

        # Self-attention
        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        h = h + self.attention(h, pair=pair_cond, mask=attn_mask)
        h = self.transition(h)

        # Predict coordinate update
        coord_input = torch.cat([self.coord_norm(h), coords], dim=-1)
        delta_coords = self.coord_update(coord_input)

        atom_feat = h
        return atom_feat, delta_coords


class DiffusionModule(nn.Module):
    """Full diffusion module for structure generation.

    Implements the score-based diffusion process that generates 3D
    atom coordinates from Gaussian noise, conditioned on trunk
    representations. Uses a variance-exploding noise schedule.

    Parameters
    ----------
    single_dim : int
        Single representation dimension from trunk.
    pair_dim : int
        Pair representation dimension from trunk.
    atom_dim : int
        Atom-level representation dimension.
    num_steps : int
        Number of diffusion timesteps.
    sigma_min : float
        Minimum noise level.
    sigma_max : float
        Maximum noise level.
    num_layers : int
        Number of denoising blocks.
    num_heads : int
        Attention heads per denoising block.
    time_dim : int
        Timestep embedding dimension.
    """

    def __init__(
        self,
        single_dim: int = 256,
        pair_dim: int = 384,
        atom_dim: int = 128,
        num_steps: int = 200,
        sigma_min: float = 0.01,
        sigma_max: float = 160.0,
        num_layers: int = 8,
        num_heads: int = 8,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.atom_dim = atom_dim

        # Conditioning
        self.conditioning = DiffusionConditioning(
            single_dim, pair_dim, atom_dim, time_dim
        )

        # Initial atom feature projection
        self.atom_proj = IsoLinear(single_dim, atom_dim)

        # Denoising blocks
        self.blocks = nn.ModuleList([
            DenoisingBlock(atom_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Final coordinate projection
        self.final_norm = nn.LayerNorm(atom_dim)
        self.final_proj = IsoLinear(atom_dim, 3, init_zeros=True)

    def noise_schedule(self, t: Tensor) -> Tensor:
        """Compute noise level σ(t) using log-linear schedule.

        Parameters
        ----------
        t : Tensor
            Timestep in [0, 1].

        Returns
        -------
        Tensor
            Noise level σ.
        """
        log_sigma = (
            math.log(self.sigma_min)
            + t * (math.log(self.sigma_max) - math.log(self.sigma_min))
        )
        return torch.exp(log_sigma)

    def forward(
        self,
        single: Tensor,
        pair: Tensor,
        coords_noisy: Tensor,
        sigma: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict denoised coordinates (score estimation).

        Parameters
        ----------
        single : Tensor (B, N, single_dim)
        pair : Tensor (B, N, N, pair_dim)
        coords_noisy : Tensor (B, N, 3)
            Noisy coordinates at noise level σ.
        sigma : Tensor (B,)
            Current noise level.
        mask : Tensor (B, N), optional

        Returns
        -------
        Tensor (B, N, 3)
            Predicted denoised coordinates.
        """
        # Compute conditioning
        single_cond, pair_cond = self.conditioning(single, pair, sigma)

        # Initial atom features
        atom_feat = self.atom_proj(single)
        coords = coords_noisy

        # Apply denoising blocks
        for block in self.blocks:
            atom_feat, delta = block(
                atom_feat, coords, single_cond, pair_cond, mask
            )
            coords = coords + delta

        # Final projection
        final_delta = self.final_proj(self.final_norm(atom_feat))
        coords = coords + final_delta

        return coords

    @torch.no_grad()
    def sample(
        self,
        single: Tensor,
        pair: Tensor,
        mask: Optional[Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """Generate structures via iterative denoising.

        Samples from the learned distribution by starting from Gaussian
        noise and iteratively denoising.

        Parameters
        ----------
        single : Tensor (B, N, single_dim)
        pair : Tensor (B, N, N, pair_dim)
        mask : Tensor (B, N), optional
        num_steps : int, optional

        Returns
        -------
        Tensor (B, N, 3)
            Generated 3D coordinates.
        """
        if num_steps is None:
            num_steps = self.num_steps

        B, N, _ = single.shape
        device = single.device

        # Start from noise
        coords = torch.randn(B, N, 3, device=device) * self.sigma_max

        # Discretise timesteps
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            sigma = self.noise_schedule(t.unsqueeze(0).expand(B))
            sigma_next = self.noise_schedule(t_next.unsqueeze(0).expand(B))

            # Predict denoised coordinates
            denoised = self.forward(single, pair, coords, sigma, mask)

            # DDPM-style update
            # x_{t-1} = denoised + sigma_next * noise (if not last step)
            if i < num_steps - 1:
                noise = torch.randn_like(coords)
                coords = denoised + sigma_next.view(B, 1, 1) * noise
            else:
                coords = denoised

        return coords
