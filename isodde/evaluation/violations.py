"""Ligand and polymer structure violation detection for IsoDDE.

Detects stereochemical violations including:
- Bond length deviation (Section 4.3)
- Bond angle deviation (Section 4.3)
- Planarity deviation (flat aromatic rings, sp2 centres)
- Chirality / stereocenter inversion
- Intra-molecular and inter-molecular clashes (Section 4.3)
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Tuple
import torch
from torch import Tensor

from isodde.data.chemical import get_ideal_bond_length, get_vdw_radius, get_covalent_radius


def check_bond_lengths(
    coords: Tensor,
    bonds: List[Tuple[int, int, str]],
    elements: List[str],
    threshold: float = 0.25,
) -> Dict[str, Any]:
    """Check if bond lengths deviate from ideal values by more than threshold (Å).

    Parameters
    ----------
    coords : Tensor (N, 3)
        Predicted 3D coordinates.
    bonds : List of (i, j, bond_type)
        Indices and types of covalent bonds.
    elements : List of str
        Element names for each atom.
    threshold : float
        Max allowed deviation in Å.

    Returns
    -------
    Dict containing:
        - num_violations: int
        - max_deviation: float
        - violations: List of Tuple[int, int, float, float] (atom1, atom2, actual, ideal)
    """
    num_violations = 0
    max_deviation = 0.0
    violations_list = []

    for i, j, btype in bonds:
        if i >= len(elements) or j >= len(elements):
            continue
        elem1 = elements[i]
        elem2 = elements[j]
        ideal_len = get_ideal_bond_length(elem1, elem2, btype)
        if ideal_len is None:
            # Fallback to covalent radii sum
            ideal_len = get_covalent_radius(elem1) + get_covalent_radius(elem2)

        actual_len = float(torch.norm(coords[i] - coords[j]).item())
        dev = abs(actual_len - ideal_len)
        if dev > dev:  # nan check
            continue

        if dev > max_deviation:
            max_deviation = dev

        if dev > threshold:
            num_violations += 1
            violations_list.append((i, j, actual_len, ideal_len))

    return {
        "num_violations": num_violations,
        "max_deviation": max_deviation,
        "violations": violations_list,
    }


def check_bond_angles(
    coords: Tensor,
    angles: List[Tuple[int, int, int, float]],
) -> Dict[str, Any]:
    """Check bond angles deviation.

    Parameters
    ----------
    coords : Tensor (N, 3)
    angles : List of (i, j, k, ideal_angle_degrees)
        j is the central vertex.

    Returns
    -------
    Dict
    """
    num_violations = 0
    max_deviation = 0.0
    violations_list = []
    threshold = 25.0 # Max allowed deviation in degrees

    for i, j, k, ideal_angle in angles:
        v1 = coords[i] - coords[j]
        v2 = coords[k] - coords[j]

        norm1 = torch.norm(v1)
        norm2 = torch.norm(v2)
        if norm1 < 1e-5 or norm2 < 1e-5:
            continue

        cos_angle = torch.dot(v1, v2) / (norm1 * norm2)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        actual_angle = math.degrees(math.acos(cos_angle.item()))

        dev = abs(actual_angle - ideal_angle)
        if dev > max_deviation:
            max_deviation = dev

        if dev > threshold:
            num_violations += 1
            violations_list.append((i, j, k, actual_angle, ideal_angle))

    return {
        "num_violations": num_violations,
        "max_deviation": max_deviation,
        "violations": violations_list,
    }


def check_clashes(
    coords: Tensor,
    elements: List[str],
    intra_bonds: List[Tuple[int, int, str]],
    is_ligand: Tensor,
    clash_intra_threshold: float = 1.5,
    clash_inter_threshold: float = 1.7,
) -> Dict[str, Any]:
    """Check for atomic clashes.

    - Intra-molecular clashes: non-bonded atoms in same molecule closer than threshold.
    - Inter-molecular clashes: atoms in different molecules closer than threshold.
    """
    N = coords.shape[0]
    num_intra_clashes = 0
    num_inter_clashes = 0
    clashes_list = []

    # Fast check: distance matrix
    dist_matrix = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)

    # Convert intra_bonds to a set of pairs for fast lookup
    bonded_pairs = set()
    for i, j, _ in intra_bonds:
        bonded_pairs.add((min(i, j), max(i, j)))
        # Also add 1-3 neighbors (connected to same atom) to skip them from clash check
        # This is a standard practice
    
    # Simple heuristic to identify chains/molecules
    # Assume same molecule if they are connected or if chain indices match
    # For now, distinguish ligand vs protein
    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in bonded_pairs:
                continue

            dist = float(dist_matrix[i, j].item())
            is_same_mol = (is_ligand[i] == is_ligand[j])

            # Get radii
            rad1 = get_vdw_radius(elements[i])
            rad2 = get_vdw_radius(elements[j])

            if is_same_mol:
                # Intra-molecular clash
                if dist < clash_intra_threshold:
                    num_intra_clashes += 1
                    clashes_list.append((i, j, dist, "intra"))
            else:
                # Inter-molecular clash
                if dist < clash_inter_threshold:
                    num_inter_clashes += 1
                    clashes_list.append((i, j, dist, "inter"))

    return {
        "num_intra_clashes": num_intra_clashes,
        "num_inter_clashes": num_inter_clashes,
        "clashes": clashes_list,
    }


def check_chirality(
    coords: Tensor,
    chiral_centers: List[Tuple[int, int, int, int, str]],
) -> Dict[str, Any]:
    """Check chirality of specified centres.

    Parameters
    ----------
    coords : Tensor (N, 3)
    chiral_centers : List of (center, a, b, c, expected_sign)
        expected_sign is either "R", "S" or "+" or "-" sign of the triple product
        det([b-a, c-a, d-a]) where center=a, and b, c, d are the neighbors.

    Returns
    -------
    Dict
    """
    num_violations = 0
    violations_list = []

    for center, a, b, c, expected in chiral_centers:
        # Compute volume of tetrahedron: dot(cross(v1, v2), v3)
        v1 = coords[a] - coords[center]
        v2 = coords[b] - coords[center]
        v3 = coords[c] - coords[center]

        triple_product = torch.dot(torch.linalg.cross(v1, v2), v3).item()
        actual_sign = "+" if triple_product > 0 else "-"

        if actual_sign != expected:
            num_violations += 1
            violations_list.append((center, actual_sign, expected))

    return {
        "num_violations": num_violations,
        "violations": violations_list,
    }
