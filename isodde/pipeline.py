"""High-level inference pipeline for IsoDDE.

Provides unified API endpoints for structural prediction, pocket identification,
and binding affinity prediction.
"""

from __future__ import annotations

import os
from typing import Dict, Any, List, Optional, Tuple
import torch

from isodde.config import IsoDDEConfig
from isodde.data.tokenizer import (
    tokenize_protein_sequence,
    tokenize_ligand_atoms,
    merge_tokenized_inputs,
)
from isodde.model.isodde_structure import IsoDDEStructurePrediction
from isodde.model.affinity import BindingAffinityHead
from isodde.model.pocket import PocketIdentificationHead
from isodde.model.sample import sample_multi_seed


class IsoDDEPipeline:
    """Unified computational pipeline for drug design using IsoDDE."""

    def __init__(self, config: Optional[IsoDDEConfig] = None) -> None:
        if config is None:
            config = IsoDDEConfig.small()  # default to small config for resource friendliness
        self.config = config

        # Instantiate models
        self.structure_model = IsoDDEStructurePrediction(config)
        self.affinity_head = BindingAffinityHead(
            config.affinity,
            pair_dim=config.pairformer.pair_dim,
            single_dim=config.pairformer.single_dim,
        )
        self.pocket_head = PocketIdentificationHead(
            config.pocket,
            single_dim=config.pairformer.single_dim,
        )

        # Set evaluation mode
        self.structure_model.eval()
        self.affinity_head.eval()
        self.pocket_head.eval()

    def run_cofolding(
        self,
        protein_sequence: str,
        ligand_smiles: Optional[str] = None,
        ligand_elements: Optional[List[str]] = None,
        num_seeds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Perform end-to-end structure, pocket, and affinity prediction.

        Parameters
        ----------
        protein_sequence : str
            One-letter protein amino acid sequence.
        ligand_smiles : str, optional
            SMILES representation of the ligand.
        ligand_elements : list of str, optional
            List of chemical element symbols for ligand atoms.
        num_seeds : int, optional
            Number of random seeds for structure sampling.

        Returns
        -------
        Dict
            Consolidated outputs (coordinates, pockets, binding affinity).
        """
        if num_seeds is None:
            num_seeds = self.config.inference.num_seeds

        device = next(self.structure_model.parameters()).device

        # 1. Tokenize inputs
        tokenized_protein = tokenize_protein_sequence(protein_sequence, chain_id=0)
        inputs_list = [tokenized_protein]

        if ligand_elements:
            tokenized_ligand = tokenize_ligand_atoms(
                len(ligand_elements), ligand_elements, chain_id=1
            )
            inputs_list.append(tokenized_ligand)

        merged = merge_tokenized_inputs(*inputs_list)

        # Expand batch dimension
        token_ids = merged.token_ids.unsqueeze(0).to(device)
        token_type = merged.token_type.unsqueeze(0).to(device)
        residue_index = merged.residue_index.unsqueeze(0).to(device)
        chain_index = merged.chain_index.unsqueeze(0).to(device)
        mask = torch.ones_like(token_ids, dtype=torch.bool, device=device)

        full_elements = ["C"] * len(protein_sequence) + (ligand_elements if ligand_elements else [])

        # 2. Multi-seed structure generation
        sample_out = sample_multi_seed(
            model=self.structure_model,
            token_ids=token_ids,
            token_type=token_type,
            residue_index=residue_index,
            chain_index=chain_index,
            mask=mask,
            num_seeds=num_seeds,
            violation_filter=self.config.inference.violation_filter,
            elements=full_elements,
        )

        best_coords = sample_out["best_coords"].unsqueeze(0)  # (1, N, 3)

        # Run final forward pass to get Pairformer representations for coordinates
        with torch.no_grad():
            # Get trunk representations
            single, pair = self.structure_model.embedding(
                token_ids, token_type, residue_index, chain_index
            )
            msa_tokens = token_ids.unsqueeze(1)
            has_deletion = torch.zeros_like(msa_tokens, dtype=torch.float)
            deletion_value = torch.zeros_like(msa_tokens, dtype=torch.float)
            msa_feat = self.structure_model.msa_features(msa_tokens, has_deletion, deletion_value)
            _, pair = self.structure_model.msa_module(msa_feat, pair)
            pair_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            pair, single = self.structure_model.pairformer(pair, single, pair_mask=pair_mask, single_mask=mask)

            # 3. Predict binding pockets
            pocket_out = self.pocket_head(single, best_coords, mask)

            # 4. Predict binding affinity
            is_lig = merged.is_ligand.unsqueeze(0).to(device)
            affinity = float(self.affinity_head(pair, best_coords, is_lig, mask).item())

            # 5. Predict interface contacts
            contact_logits = self.structure_model.interface_contact_head(pair, mask)
            contact_probs = torch.sigmoid(contact_logits).squeeze(0)  # (N, N)

            # 6. Predict protein-ligand contacts
            pl_contact_logits = self.structure_model.protein_ligand_contact_head(pair, mask)
            pl_contact_probs = torch.sigmoid(pl_contact_logits).squeeze(0)  # (N, N)

            # 7. Predict secondary structure
            ss_logits = self.structure_model.secondary_structure_head(single)
            ss_pred = torch.argmax(ss_logits, dim=-1).squeeze(0)  # (N,)
            ss_list = ss_pred[:len(protein_sequence)].tolist()

            # 8. Predict solvent accessibility
            rsa_tensor = self.structure_model.solvent_accessibility_head(single).squeeze(0)  # (N,)
            rsa_list = rsa_tensor[:len(protein_sequence)].tolist()

            # 9. Extract residue plddt list
            plddt_list = sample_out["best_plddt"].squeeze(0)[:len(protein_sequence)].tolist()

        return {
            "predicted_coords": best_coords.squeeze(0).tolist(),
            "pLDDT": sample_out["best_score"],
            "ptm": sample_out["best_ptm"],
            "binding_affinity_pkd": affinity,
            "pockets": pocket_out["pockets"][0],
            "interface_contact_probs": contact_probs.tolist(),
            "protein_ligand_contact_probs": pl_contact_probs.tolist(),
            "secondary_structure": ss_list,
            "solvent_accessibility": rsa_list,
            "plddt_list": plddt_list,
        }
