"""Sampling and structure generation utilities for IsoDDE.

Implements multi-seed generation, ranking based on composite confidence scores,
and stereochemistry violation filtering as described in Sections 1.1 and 4.3.
"""

from __future__ import annotations

import random
from typing import Dict, Any, List, Tuple, Optional
import torch
from torch import Tensor

from isodde.config import IsoDDEConfig
from isodde.evaluation.violations import check_bond_lengths, check_bond_angles, check_clashes, check_chirality


def sample_multi_seed(
    model: torch.nn.Module,
    token_ids: Tensor,
    token_type: Tensor,
    residue_index: Tensor,
    chain_index: Tensor,
    msa_tokens: Optional[Tensor] = None,
    has_deletion: Optional[Tensor] = None,
    deletion_value: Optional[Tensor] = None,
    msa_mask: Optional[Tensor] = None,
    template_pair_feat: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    interface_mask: Optional[Tensor] = None,
    num_seeds: int = 5,
    violation_filter: bool = True,
    bonds: Optional[List[Tuple[int, int, str]]] = None,
    angles: Optional[List[Tuple[int, int, int, float]]] = None,
    chiral_centers: Optional[List[Tuple[int, int, int, int, str]]] = None,
    elements: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate structures across multiple seeds and rank/filter them.

    Parameters
    ----------
    model : nn.Module
        The IsoDDEStructurePrediction model.
    token_ids, token_type, residue_index, chain_index : Tensor
        Input tokens and annotations.
    num_seeds : int
        Number of random seed structure predictions to run.
    violation_filter : bool
        If True, filters out predictions that violate physical constraints.
    bonds, angles, chiral_centers : List, optional
        Chemical annotations for stereochemical validation.
    elements : List of str, optional
        Element symbol for each atom.

    Returns
    -------
    Dict containing:
        - best_coords: Tensor (N, 3)
        - best_score: float
        - best_seed: int
        - all_results: List of Dict (metrics and coords per seed)
    """
    model.eval()
    device = token_ids.device

    all_results = []
    
    # Simple default annotations if none provided
    if elements is None:
        # Construct elements based on token_ids (assume carbon as default fallback)
        from isodde.data.tokenizer import ELEMENTS
        elements = []
        for tid in token_ids[0].tolist():
            idx = min(int(tid), len(ELEMENTS) - 1)
            elements.append(ELEMENTS[idx])

    if bonds is None:
        # Connect consecutive residue tokens for protein / ligand tokens
        bonds = []
        for i in range(len(elements) - 1):
            if chain_index[0, i] == chain_index[0, i + 1]:
                bonds.append((i, i + 1, "SINGLE"))

    if angles is None:
        angles = []
    if chiral_centers is None:
        chiral_centers = []

    is_ligand = (token_type == 3).squeeze(0)  # TokenType.LIGAND = 3

    for seed_idx in range(num_seeds):
        # Set seed for reproducible sampling of diffusion
        seed = 42 + seed_idx
        torch.manual_seed(seed)
        random.seed(seed)

        with torch.no_grad():
            outputs = model(
                token_ids=token_ids,
                token_type=token_type,
                residue_index=residue_index,
                chain_index=chain_index,
                msa_tokens=msa_tokens,
                has_deletion=has_deletion,
                deletion_value=deletion_value,
                msa_mask=msa_mask,
                template_pair_feat=template_pair_feat,
                mask=mask,
                interface_mask=interface_mask,
            )

        coords = outputs["pred_coords"].squeeze(0)  # (N, 3)
        ranking_score = float(outputs["ranking_score"].mean().item())
        plddt = outputs["plddt"].squeeze(0)
        ptm = float(outputs["ptm"].mean().item())
        iptm = float(outputs["iptm"].mean().item()) if "iptm" in outputs else None

        # Stereochemical validation
        violations = {}
        valid = True
        
        if violation_filter:
            bond_check = check_bond_lengths(coords, bonds, elements)
            angle_check = check_bond_angles(coords, angles)
            clash_check = check_clashes(coords, elements, bonds, is_ligand)
            chiral_check = check_chirality(coords, chiral_centers)

            violations = {
                "bond_length_violations": bond_check["num_violations"],
                "bond_angle_violations": angle_check["num_violations"],
                "intra_clashes": clash_check["num_intra_clashes"],
                "inter_clashes": clash_check["num_inter_clashes"],
                "chirality_violations": chiral_check["num_violations"],
            }

            # Filter criterion: too many core violations
            # (Allows mild violations, but rejects highly unnatural structures)
            if (
                bond_check["num_violations"] > len(bonds) * 0.15 + 3
                or clash_check["num_intra_clashes"] > len(elements) * 0.2 + 5
            ):
                valid = False

        result = {
            "seed": seed,
            "coords": coords,
            "ranking_score": ranking_score,
            "plddt": plddt,
            "ptm": ptm,
            "iptm": iptm,
            "valid": valid,
            "violations": violations,
        }
        all_results.append(result)

    # Filter out invalid structures if possible, otherwise keep all
    valid_results = [r for r in all_results if r["valid"]]
    if not valid_results:
        # Fallback to all if everything violated threshold
        valid_results = all_results

    # Select the candidate with the highest composite ranking score
    best_result = max(valid_results, key=lambda r: r["ranking_score"])

    return {
        "best_coords": best_result["coords"],
        "best_score": best_result["ranking_score"],
        "best_seed": best_result["seed"],
        "best_ptm": best_result["ptm"],
        "best_iptm": best_result["iptm"],
        "best_plddt": best_result["plddt"],
        "all_results": all_results,
    }
