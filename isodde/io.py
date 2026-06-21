"""Input/Output utilities for IsoDDE.

Handles PDB formatting, coordinates exporting, and prediction serialization.
"""

from __future__ import annotations

import json
from typing import Dict, Any, List, Tuple


def write_coords_to_pdb(
    coords: List[List[float]],
    elements: List[str],
    output_path: str,
    chain_id: str = "A",
) -> None:
    """Write coordinates and element types to a standard PDB format file.

    Parameters
    ----------
    coords : List of [x, y, z] coordinates.
    elements : List of element symbols corresponding to each coordinate.
    output_path : Target PDB file path.
    """
    with open(output_path, "w") as f:
        for idx, (coord, elem) in enumerate(zip(coords, elements)):
            # PDB line format details:
            # ATOM/HETATM, atom index, atom name, residue name, chain ID,
            # residue sequence number, x, y, z, occupancy, temp factor, element
            name = f"{elem}{idx+1}"[:4].rjust(4)
            line = (
                f"ATOM  {idx+1:5d} {name} TRP {chain_id} {idx+1:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00 20.00"
                f"           {elem:2s}\n"
            )
            f.write(line)


def parse_pdb_sequence(pdb_path: str) -> Tuple[str, List[List[float]], List[str]]:
    """Parse protein sequence, coords, and elements from a PDB file.

    Parameters
    ----------
    pdb_path : Path to PDB file.

    Returns
    -------
    Tuple (sequence, coordinates, elements)
    """
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }

    sequence_list = []
    coords = []
    elements = []
    last_res_num = None

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                res_name = line[17:20].strip()
                res_num = int(line[22:26].strip())
                elem = line[76:78].strip()
                if not elem:
                    elem = line[12:14].strip()[0]  # Guess element from name

                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                coords.append([x, y, z])
                elements.append(elem)

                if last_res_num is None or res_num != last_res_num:
                    sequence_list.append(three_to_one.get(res_name, "X"))
                    last_res_num = res_num

    return "".join(sequence_list), coords, elements


def save_prediction_json(data: Dict[str, Any], output_path: str) -> None:
    """Save prediction metrics and metadata to JSON."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def write_coords_to_sdf(
    coords: List[List[float]],
    elements: List[str],
    output_path: str,
    molecule_name: str = "IsoDDE_Ligand",
) -> None:
    """Write coordinates and elements to a standard MDL SDF format file.

    Parameters
    ----------
    coords : List of [x, y, z] coordinates.
    elements : List of element symbols corresponding to each coordinate.
    output_path : Target SDF file path.
    molecule_name : Name of the molecule.
    """
    with open(output_path, "w") as f:
        f.write(f"{molecule_name}\n")
        f.write("  IsoDDE 3D Predictor\n\n")
        
        num_atoms = len(coords)
        f.write(f"{num_atoms:3d}  0  0  0  0  0  0  0  0  0999 V2000\n")
        
        for coord, elem in zip(coords, elements):
            f.write(
                f"{coord[0]:10.4f}{coord[1]:10.4f}{coord[2]:10.4f} "
                f"{elem:<3s} 0  0  0  0  0  0  0  0  0  0  0  0\n"
            )
            
        f.write("M  END\n")
        f.write("$$$$\n")


def parse_sdf_file(sdf_path: str) -> Tuple[List[List[float]], List[str]]:
    """Parse coordinates and elements from a standard MDL SDF format file.

    Parameters
    ----------
    sdf_path : Path to SDF file.

    Returns
    -------
    Tuple (coordinates, elements)
        List of [x, y, z] coordinates and list of element symbols.
    """
    coords = []
    elements = []
    
    with open(sdf_path, "r") as f:
        lines = f.readlines()
        
    if len(lines) < 4:
        return coords, elements
        
    counts_line = lines[3]
    try:
        num_atoms = int(counts_line[:3].strip())
    except ValueError:
        return coords, elements
        
    for i in range(4, 4 + num_atoms):
        if i >= len(lines):
            break
        line = lines[i]
        try:
            x = float(line[0:10].strip())
            y = float(line[10:20].strip())
            z = float(line[20:30].strip())
            elem = line[31:34].strip()
            coords.append([x, y, z])
            elements.append(elem)
        except (ValueError, IndexError):
            continue
            
    return coords, elements
