"""Example: Binding affinity prediction using IsoDDE."""

import torch
from isodde.config import IsoDDEConfig
from isodde.model.affinity import BindingAffinityHead


def main():
    # Setup configuration
    config = IsoDDEConfig.small()
    
    # Initialize affinity head
    affinity_head = BindingAffinityHead(
        config.affinity,
        pair_dim=config.pairformer.pair_dim,
        single_dim=config.pairformer.single_dim,
    )
    affinity_head.eval()

    # Generate mock inputs for a complex of size 128
    B, N = 1, 128
    
    # Mock Pairformer pair representations
    pair_repr = torch.randn(B, N, N, config.pairformer.pair_dim)
    
    # Mock coordinates
    coords = torch.randn(B, N, 3) * 5.0
    
    # Label last 10 residues as ligand atoms
    is_ligand = torch.zeros(B, N, dtype=torch.bool)
    is_ligand[0, -10:] = True

    # Predict affinity
    with torch.no_grad():
        pred_affinity = affinity_head(pair_repr, coords, is_ligand)

    print(f"Predicted binding affinity (pKd): {pred_affinity.item():.4f}")


if __name__ == "__main__":
    main()
