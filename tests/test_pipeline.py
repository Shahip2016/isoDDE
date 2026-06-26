"""Integration test for the high-level IsoDDE cofolding pipeline."""

from __future__ import annotations

import pytest
from isodde.pipeline import IsoDDEPipeline


def test_pipeline_integration():
    # Initialize pipeline with small configuration
    pipeline = IsoDDEPipeline()

    protein_seq = "MTEYKLVVVGAGGVGKSALTI"
    ligand_elements = ["C", "C", "O", "N"]

    # Run unified pipeline
    results = pipeline.run_cofolding(
        protein_sequence=protein_seq,
        ligand_elements=ligand_elements,
        num_seeds=2,
    )

    # Validate results output structure
    assert "predicted_coords" in results
    assert "pLDDT" in results
    assert "ptm" in results
    assert "binding_affinity_pkd" in results
    assert "pockets" in results
    assert "secondary_structure" in results
    assert "solvent_accessibility" in results

    assert len(results["predicted_coords"]) == len(protein_seq) + len(ligand_elements)
    assert isinstance(results["pLDDT"], float)
    assert isinstance(results["ptm"], float)
    assert isinstance(results["binding_affinity_pkd"], float)
    assert isinstance(results["pockets"], list)
    assert isinstance(results["secondary_structure"], list)
    assert len(results["secondary_structure"]) == len(protein_seq)
    assert isinstance(results["solvent_accessibility"], list)
    assert len(results["solvent_accessibility"]) == len(protein_seq)
