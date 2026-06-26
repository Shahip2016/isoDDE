"""Command-Line Interface (CLI) for running IsoDDE predictions.

Supports structure prediction, affinity estimation, and pocket detection.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from isodde.config import IsoDDEConfig
from isodde.pipeline import IsoDDEPipeline
from isodde.io import write_coords_to_pdb, save_prediction_json, parse_pdb_sequence


def main(args_list: List[str] = None) -> int:
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(
        description="IsoDDE: Isomorphic Labs Drug Design Engine — Structure, Affinity & Pockets Predictor."
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Protein sequence to predict. If not provided, --pdb must be specified.",
    )
    parser.add_argument(
        "--pdb",
        type=str,
        help="Input protein PDB file to load sequence and structures from.",
    )
    parser.add_argument(
        "--ligand",
        type=str,
        help="SMILES string or comma-separated element list of the ligand (e.g. C,C,O,N).",
    )
    parser.add_argument(
        "--sdf-ligand",
        type=str,
        help="Path to an input SDF file to load ligand elements and coordinates from.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="predictions",
        help="Directory to save generated PDB structure and metrics JSON.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=2,
        help="Number of random seeds for structure sampling (default: 2).",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use a small model configuration for faster processing.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the interactive web UI server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the web server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to run the web server on (default: 8000).",
    )

    args = parser.parse_args(args_list)

    if args.web:
        import uvicorn
        from isodde.web import app
        print("\n" + "=" * 50)
        print(f"Starting IsoDDE Web UI server at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop.")
        print("=" * 50 + "\n")
        uvicorn.run(app, host=args.host, port=args.port)
        return 0

    if not args.sequence and not args.pdb:
        parser.print_help()
        print("\n[Error] Must specify either --sequence, --pdb, or --web.")
        return 1

    # Load sequence and initial coordinates
    protein_seq = args.sequence
    elements = []
    if args.pdb:
        if not os.path.exists(args.pdb):
            print(f"[Error] PDB file not found: {args.pdb}")
            return 1
        try:
            parsed_seq, parsed_coords, parsed_elements = parse_pdb_sequence(args.pdb)
            if not protein_seq:
                protein_seq = parsed_seq
            print(f"Loaded sequence of length {len(protein_seq)} from PDB: {args.pdb}")
        except Exception as e:
            print(f"[Error] Failed to parse PDB file: {e}")
            return 1

    # Format ligand inputs
    ligand_elements = None
    if args.sdf_ligand:
        if not os.path.exists(args.sdf_ligand):
            print(f"[Error] SDF ligand file not found: {args.sdf_ligand}")
            return 1
        try:
            from isodde.io import parse_sdf_file
            _, ligand_elements = parse_sdf_file(args.sdf_ligand)
            print(f"Loaded ligand with {len(ligand_elements)} atoms from SDF: {args.sdf_ligand}")
        except Exception as e:
            print(f"[Error] Failed to parse SDF ligand file: {e}")
            return 1
    elif args.ligand:
        if "," in args.ligand:
            ligand_elements = [e.strip() for e in args.ligand.split(",")]
        else:
            # Interpret as elements list default C if smiles
            # In a real model, this converts SMILES -> atom elements. Here we treat SMILES characters
            # or elements. Let's extract letters as elements.
            import re
            ligand_elements = re.findall(r"[A-Z][a-z]?", args.ligand)
            # Filter non-elements
            from isodde.data.tokenizer import ELEMENTS
            ligand_elements = [e for e in ligand_elements if e in ELEMENTS]
            if not ligand_elements:
                ligand_elements = ["C", "C", "O", "N"]

    # Setup config
    config = IsoDDEConfig.small() if args.small else IsoDDEConfig()
    config.inference.num_seeds = args.num_seeds

    print("Initializing IsoDDE models...")
    pipeline = IsoDDEPipeline(config)

    print(f"Running prediction pipeline (seeds: {args.num_seeds})...")
    try:
        results = pipeline.run_cofolding(
            protein_sequence=protein_seq,
            ligand_elements=ligand_elements,
            num_seeds=args.num_seeds,
        )
    except Exception as e:
        print(f"[Error] Prediction pipeline failed: {e}")
        return 1

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    pdb_out = os.path.join(args.output_dir, "predicted_complex.pdb")
    json_out = os.path.join(args.output_dir, "prediction_metrics.json")

    # Save structure
    full_elements = [e for e in protein_seq] + (ligand_elements if ligand_elements else [])
    # standard protein residues have multiple atoms; mapping here uses length matching:
    if len(results["predicted_coords"]) != len(full_elements):
        # Fallback to general indexing if count differs
        full_elements = ["C"] * len(results["predicted_coords"])

    write_coords_to_pdb(results["predicted_coords"], full_elements, pdb_out)

    # Save ligand SDF if ligand exists
    if ligand_elements:
        from isodde.io import write_coords_to_sdf
        sdf_out = os.path.join(args.output_dir, "predicted_ligand.sdf")
        # Extract ligand coordinates: the last len(ligand_elements) coordinates
        ligand_coords = results["predicted_coords"][-len(ligand_elements):]
        write_coords_to_sdf(ligand_coords, ligand_elements, sdf_out)
        print(f"Ligand structure saved to:  {sdf_out}")
    
    # Save metrics
    metrics_data = {
        "pLDDT": results["pLDDT"],
        "ptm": results["ptm"],
        "binding_affinity_pkd": results["binding_affinity_pkd"],
        "pockets_detected": len(results["pockets"]),
        "pockets_info": results["pockets"],
        "interface_contact_probs": results["interface_contact_probs"],
        "protein_ligand_contact_probs": results["protein_ligand_contact_probs"],
    }
    save_prediction_json(metrics_data, json_out)

    # Count high-confidence interface contacts
    num_contacts = 0
    probs_list = results["interface_contact_probs"]
    N = len(probs_list)
    for i in range(N):
        for j in range(i + 1, N):
            if probs_list[i][j] > 0.5:
                num_contacts += 1

    # Count high-confidence protein-ligand contacts
    num_pl_contacts = 0
    pl_probs_list = results["protein_ligand_contact_probs"]
    N_pl = len(pl_probs_list)
    for i in range(N_pl):
        for j in range(i + 1, N_pl):
            if pl_probs_list[i][j] > 0.5:
                num_pl_contacts += 1

    print("\n" + "=" * 50)
    print("IsoDDE Prediction Completed Successfully!")
    print("=" * 50)
    print(f"Structure saved to:  {pdb_out}")
    print(f"Metrics saved to:    {json_out}")
    print("-" * 50)
    print(f"Predicted pLDDT:     {results['pLDDT']:.4f}")
    print(f"Predicted pTM:       {results['ptm']:.4f}")
    print(f"Predicted Affinity:  {results['binding_affinity_pkd']:.4f} pKd")
    print(f"Pockets Identified:  {len(results['pockets'])}")
    print(f"Interface Contacts:  {num_contacts}")
    print(f"Protein-Ligand Contacts: {num_pl_contacts}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
