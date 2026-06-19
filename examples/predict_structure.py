"""Example: Protein-Ligand cofolding prediction using IsoDDE."""

import os
from isodde.pipeline import IsoDDEPipeline
from isodde.io import write_coords_to_pdb


def main():
    # Initialize pipeline
    print("Initializing IsoDDE pipeline...")
    pipeline = IsoDDEPipeline()

    # Define inputs (mock target protein + ligand)
    protein_sequence = "MTEYKLVVVGAGGVGKSALTI"
    ligand_elements = ["C", "C", "O", "N"]

    print("Running unified cofolding prediction...")
    results = pipeline.run_cofolding(
        protein_sequence=protein_sequence,
        ligand_elements=ligand_elements,
        num_seeds=3,
    )

    # Output details
    print("\nPrediction Results:")
    print("-" * 30)
    print(f"Confidence score (pLDDT): {results['pLDDT']:.4f}")
    print(f"Predicted pTM:            {results['ptm']:.4f}")
    print(f"Predicted Affinity (pKd): {results['binding_affinity_pkd']:.4f}")
    print(f"Pockets Identified:       {len(results['pockets'])}")

    # Save to PDB
    output_pdb = "examples_predicted_complex.pdb"
    full_elements = [e for e in protein_sequence] + ligand_elements
    write_coords_to_pdb(results["predicted_coords"], full_elements, output_pdb)
    print(f"\nStructure saved to: {output_pdb}")


if __name__ == "__main__":
    main()
