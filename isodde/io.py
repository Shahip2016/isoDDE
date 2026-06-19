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
