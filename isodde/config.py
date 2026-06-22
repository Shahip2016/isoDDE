"""Model configuration dataclasses for IsoDDE.

Hyperparameters are inspired by AlphaFold 3's published architecture and the
improvements described in the IsoDDE technical report (wider pairformer
representations, reordered MSA operations, improved diffusion head).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MSAConfig:
    """Configuration for the MSA processing module."""

    num_msa_sequences: int = 512
    msa_embedding_dim: int = 64
    num_msa_blocks: int = 4
    num_heads: int = 8
    dropout: float = 0.0
    outer_product_mean_dim: int = 32


@dataclass
class PairformerConfig:
    """Configuration for the Pairformer stack.

    The wider pair representation (384 vs AF3's 128) follows improvements
    noted in Zhou et al. (2025) / SeedFold referenced in the paper.
    """

    pair_dim: int = 384
    single_dim: int = 256
    num_blocks: int = 48
    num_heads_pair: int = 16
    num_heads_single: int = 16
    num_transition_blocks: int = 2
    transition_multiplier: float = 4.0
    dropout: float = 0.0
    chunk_size: Optional[int] = None  # For memory-efficient triangular ops


@dataclass
class DiffusionConfig:
    """Configuration for the score-based diffusion structure generator."""

    atom_dim: int = 128
    num_diffusion_steps: int = 200
    sigma_min: float = 0.01
    sigma_max: float = 160.0
    num_denoising_layers: int = 8
    num_heads: int = 8
    time_embedding_dim: int = 256
    dropout: float = 0.0


@dataclass
class ConfidenceConfig:
    """Configuration for the confidence prediction head."""

    pair_dim: int = 384
    single_dim: int = 256
    num_bins_distogram: int = 64
    plddt_bins: int = 50
    num_heads: int = 4
    num_layers: int = 4
    max_dist: float = 22.0


@dataclass
class AffinityConfig:
    """Configuration for the binding affinity prediction head.

    Predicts pKd / pIC50 / pEC50 from structure-conditioned
    pair representations and ligand-pocket interaction features.
    """

    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    output_dim: int = 1  # scalar affinity


@dataclass
class PocketConfig:
    """Configuration for the pocket identification head.

    Predicts residue-level ligand-binding probability.
    Clustering uses single-linkage at 5 Å threshold (Section 4.6).
    """

    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    cluster_threshold_angstrom: float = 5.0
    min_pocket_residues: int = 10


@dataclass
class InterfaceContactConfig:
    """Configuration for the interface contact prediction head."""

    hidden_dim: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    contact_threshold_angstrom: float = 8.0


@dataclass
class DataConfig:
    """Configuration for input data processing."""

    max_protein_length: int = 1800
    max_ligand_atoms: int = 150
    max_msa_depth: int = 512
    max_template_hits: int = 4
    crop_size: int = 384
    num_residue_types: int = 21  # 20 amino acids + unknown
    num_atom_types: int = 128   # element types
    num_bond_types: int = 5     # single, double, triple, aromatic, other


@dataclass
class InferenceConfig:
    """Configuration for inference-time behaviour."""

    num_seeds: int = 25
    max_seeds: int = 1000  # Up to 1000 seeds for antibody-antigen (Figure 6)
    violation_filter: bool = True
    violation_bond_threshold: float = 0.25
    violation_angle_threshold: float = 0.25
    clash_intra_threshold: float = 1.5  # Å
    clash_inter_threshold: float = 1.7  # Å (allows covalent bonds, Section 4.3)


@dataclass
class IsoDDEConfig:
    """Top-level configuration for the IsoDDE model.

    Aggregates all sub-configurations. Default values correspond to the
    full-scale architecture described in the paper.
    """

    msa: MSAConfig = field(default_factory=MSAConfig)
    pairformer: PairformerConfig = field(default_factory=PairformerConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    affinity: AffinityConfig = field(default_factory=AffinityConfig)
    pocket: PocketConfig = field(default_factory=PocketConfig)
    interface_contact: InterfaceContactConfig = field(default_factory=InterfaceContactConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Training cutoff (Section 4.4)
    training_cutoff_date: str = "2021-09-30"  # AF3 cutoff

    @classmethod
    def small(cls) -> "IsoDDEConfig":
        """Return a small configuration for testing and development."""
        return cls(
            msa=MSAConfig(
                num_msa_sequences=32,
                msa_embedding_dim=32,
                num_msa_blocks=2,
                num_heads=4,
            ),
            pairformer=PairformerConfig(
                pair_dim=64,
                single_dim=32,
                num_blocks=4,
                num_heads_pair=4,
                num_heads_single=4,
            ),
            diffusion=DiffusionConfig(
                atom_dim=32,
                num_diffusion_steps=10,
                num_denoising_layers=2,
                num_heads=4,
                time_embedding_dim=32,
            ),
            confidence=ConfidenceConfig(
                pair_dim=64,
                single_dim=32,
                num_bins_distogram=16,
                plddt_bins=10,
                num_heads=4,
                num_layers=2,
            ),
            affinity=AffinityConfig(hidden_dim=32, num_layers=2, num_heads=4),
            pocket=PocketConfig(hidden_dim=32, num_layers=2, num_heads=4),
            interface_contact=InterfaceContactConfig(hidden_dim=32, num_layers=2, num_heads=4),
            data=DataConfig(max_protein_length=128, crop_size=64),
        )
