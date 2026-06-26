"""FastAPI web server for IsoDDE.

Serves APIs for structure, affinity, and pocket prediction and hosts the static chat UI.
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from isodde.config import IsoDDEConfig
from isodde.pipeline import IsoDDEPipeline
from isodde.io import write_coords_to_pdb
from isodde.data.tokenizer import ELEMENTS

app = FastAPI(
    title="IsoDDE Chat Engine",
    description="Web API and Chat Interface for Isomorphic Labs Drug Design Engine",
    version="0.1.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (initialized on demand or at startup)
_pipeline: Optional[IsoDDEPipeline] = None


def get_pipeline() -> IsoDDEPipeline:
    """Retrieve or initialize the IsoDDE pipeline instance."""
    global _pipeline
    if _pipeline is None:
        # Default to small config for resource friendliness
        config = IsoDDEConfig.small()
        _pipeline = IsoDDEPipeline(config)
    return _pipeline


class PredictionRequest(BaseModel):
    protein_sequence: str
    ligand: Optional[str] = None  # Can be SMILES or comma-separated elements (e.g. "C,C,O,N")
    num_seeds: Optional[int] = 2


class PocketInfo(BaseModel):
    center: List[float]
    radius: float
    score: float


class PredictionResponse(BaseModel):
    pLDDT: float
    ptm: float
    binding_affinity_pkd: float
    pockets: List[Any]
    interface_contact_probs: List[List[float]]
    protein_ligand_contact_probs: List[List[float]]
    pdb_content: str
    sdf_content: Optional[str] = None
    secondary_structure: Optional[List[int]] = None
    solvent_accessibility: Optional[List[float]] = None
    plddt_list: Optional[List[float]] = None
    quality_report: Optional[Dict[str, Any]] = None
    protein_length: int
    ligand_length: int
    elements: List[str]


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint to verify backend status."""
    return {"status": "healthy", "model": "IsoDDE (small configuration)"}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> Dict[str, Any]:
    """Run structure, affinity, and pocket prediction on inputs."""
    # Clean sequence
    protein_seq = request.protein_sequence.strip().upper()
    if not protein_seq:
        raise HTTPException(status_code=400, detail="Protein sequence cannot be empty.")

    # Validate protein sequence format (amino acids letters only)
    if not re.match(r"^[A-Z]+$", protein_seq):
        raise HTTPException(status_code=400, detail="Protein sequence must contain only single-letter amino acid codes.")

    # Parse ligand inputs
    ligand_elements = None
    if request.ligand:
        ligand_str = request.ligand.strip()
        if "," in ligand_str:
            ligand_elements = [e.strip() for e in ligand_str.split(",")]
        elif ligand_str:
            # Interpret as elements list or extract symbols
            ligand_elements = re.findall(r"[A-Z][a-z]?", ligand_str)
            # Filter valid elements
            ligand_elements = [e for e in ligand_elements if e in ELEMENTS]
            if not ligand_elements:
                # Default fallback
                ligand_elements = ["C", "C", "O", "N"]
    else:
        ligand_elements = []

    # Get pipeline and run
    try:
        pipeline = get_pipeline()
        
        # Temporarily override inference seeds if specified
        old_seeds = pipeline.config.inference.num_seeds
        if request.num_seeds is not None:
            pipeline.config.inference.num_seeds = request.num_seeds
            
        results = pipeline.run_cofolding(
            protein_sequence=protein_seq,
            ligand_elements=ligand_elements if ligand_elements else None,
            num_seeds=request.num_seeds,
        )
        
        # Restore configuration
        pipeline.config.inference.num_seeds = old_seeds
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction pipeline failed: {str(e)}")

    # Construct PDB file content in memory using a temporary file
    full_elements = [e for e in protein_seq] + (ligand_elements if ligand_elements else [])
    if len(results["predicted_coords"]) != len(full_elements):
        full_elements = ["C"] * len(results["predicted_coords"])

    pdb_content = ""
    try:
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".pdb", delete=False) as tmp:
            tmp_name = tmp.name
        
        # Write coordinates to the temp PDB
        write_coords_to_pdb(results["predicted_coords"], full_elements, tmp_name)
        
        # Read the file content
        with open(tmp_name, "r") as f:
            pdb_content = f.read()
            
        # Clean up temp file
        os.remove(tmp_name)
    except Exception as e:
        # Fallback raw PDB generation if file operations fail
        lines = []
        for idx, (coord, elem) in enumerate(zip(results["predicted_coords"], full_elements)):
            name = f"{elem}{idx+1}"[:4].rjust(4)
            lines.append(
                f"ATOM  {idx+1:5d} {name} TRP A {idx+1:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00 20.00"
                f"           {elem:2s}\n"
            )
        pdb_content = "".join(lines)

    # Generate SDF content if ligand elements are present
    sdf_content = None
    if ligand_elements:
        try:
            from isodde.io import write_coords_to_sdf
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".sdf", delete=False) as tmp_sdf:
                tmp_sdf_name = tmp_sdf.name
            
            # Extract ligand coords: last N_ligand coords
            ligand_coords = results["predicted_coords"][-len(ligand_elements):]
            write_coords_to_sdf(ligand_coords, ligand_elements, tmp_sdf_name)
            
            with open(tmp_sdf_name, "r") as f:
                sdf_content = f.read()
            os.remove(tmp_sdf_name)
        except Exception as e:
            # Fallback simple SDF generation
            lines = ["IsoDDE_Ligand\n  IsoDDE 3D Predictor\n\n"]
            num_atoms = len(ligand_elements)
            lines.append(f"{num_atoms:3d}  0  0  0  0  0  0  0  0  0999 V2000\n")
            ligand_coords = results["predicted_coords"][-len(ligand_elements):]
            for coord, elem in zip(ligand_coords, ligand_elements):
                lines.append(
                    f"{coord[0]:10.4f}{coord[1]:10.4f}{coord[2]:10.4f} "
                    f"{elem:<3s} 0  0  0  0  0  0  0  0  0  0  0  0\n"
                )
            lines.append("M  END\n$$$$\n")
            sdf_content = "".join(lines)

    return {
        "pLDDT": results["pLDDT"],
        "ptm": results["ptm"],
        "binding_affinity_pkd": results["binding_affinity_pkd"],
        "pockets": results["pockets"],
        "interface_contact_probs": results["interface_contact_probs"],
        "protein_ligand_contact_probs": results["protein_ligand_contact_probs"],
        "pdb_content": pdb_content,
        "sdf_content": sdf_content,
        "secondary_structure": results.get("secondary_structure"),
        "solvent_accessibility": results.get("solvent_accessibility"),
        "plddt_list": results.get("plddt_list"),
        "quality_report": results.get("quality_report"),
        "protein_length": len(protein_seq),
        "ligand_length": len(ligand_elements) if ligand_elements else 0,
        "elements": full_elements
    }


# Mount the static files directory to serve the frontend UI
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    # Safe fallback if static directory hasn't been created yet
    @app.get("/")
    def read_root() -> Dict[str, str]:
        return {"message": "IsoDDE API server is running. Static files directory is missing."}
