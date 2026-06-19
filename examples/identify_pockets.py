"""Example: Blind ligand-binding pocket identification using IsoDDE."""

import torch
from isodde.config import IsoDDEConfig
from isodde.model.pocket import PocketIdentificationHead


def main():
    config = IsoDDEConfig.small()
    
    # Initialize pocket identification head
    pocket_head = PocketIdentificationHead(
        config.pocket,
        single_dim=config.pairformer.single_dim,
    )
    pocket_head.eval()

    # Generate mock inputs for a protein with 150 residues
    B, N = 1, 150
    single_repr = torch.randn(B, N, config.pairformer.single_dim)
    coords = torch.randn(B, N, 3) * 10.0

    # Run pocket detection head
    with torch.no_grad():
        outputs = pocket_head(single_repr, coords)

    # Output clusters
    pockets = outputs["pockets"][0]
    print(f"Detected {len(pockets)} pocket(s) exceeding size threshold:")
    print("-" * 50)
    for idx, pocket in enumerate(pockets):
        print(f"Pocket #{idx+1}:")
        print(f"  Number of residues: {pocket['size']}")
        print(f"  Center coordinates: {pocket['center']}")
        print(f"  Residue indices:    {pocket['residue_indices']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
