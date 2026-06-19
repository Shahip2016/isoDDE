"""Residue, atom, and ligand tokenization for IsoDDE.

Handles conversion of protein sequences, ligand SMILES, and nucleic acid
sequences into integer token indices and associated metadata for the
model's input embedding layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Amino acid vocabulary
# ---------------------------------------------------------------------------

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
UNK_AA_INDEX = len(AMINO_ACIDS)  # 20
NUM_AA_TYPES = len(AMINO_ACIDS) + 1  # +1 for unknown

# Standard amino acid atom counts (heavy atoms)
AA_HEAVY_ATOM_COUNTS = {
    "G": 4, "A": 5, "V": 7, "L": 8, "I": 8, "P": 7, "F": 11,
    "W": 14, "M": 8, "S": 6, "T": 7, "C": 6, "Y": 12, "H": 10,
    "D": 8, "E": 9, "N": 8, "Q": 9, "K": 9, "R": 11,
}

# ---------------------------------------------------------------------------
# Nucleotide vocabulary
# ---------------------------------------------------------------------------

DNA_BASES = "ACGT"
RNA_BASES = "ACGU"
DNA_TO_INDEX = {b: i for i, b in enumerate(DNA_BASES)}
RNA_TO_INDEX = {b: i for i, b in enumerate(RNA_BASES)}

# ---------------------------------------------------------------------------
# Element vocabulary (for ligand atoms)
# ---------------------------------------------------------------------------

ELEMENTS = [
    "H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl",
    "Se", "Br", "I",  # Allowed elements per Section 4.1
]
ELEMENT_TO_INDEX = {e: i for i, e in enumerate(ELEMENTS)}
UNK_ELEMENT_INDEX = len(ELEMENTS)
NUM_ELEMENT_TYPES = len(ELEMENTS) + 1

# Bond types
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "OTHER"]
BOND_TO_INDEX = {b: i for i, b in enumerate(BOND_TYPES)}
NUM_BOND_TYPES = len(BOND_TYPES)


# ---------------------------------------------------------------------------
# Token types (polymer vs ligand vs ion etc.)
# ---------------------------------------------------------------------------

class TokenType:
    """Enumeration of token types for the unified input representation."""
    PROTEIN = 0
    DNA = 1
    RNA = 2
    LIGAND = 3
    ION = 4
    WATER = 5
    NUM_TYPES = 6


# ---------------------------------------------------------------------------
# Tokenization outputs
# ---------------------------------------------------------------------------

@dataclass
class TokenizedInput:
    """Container for tokenized model inputs.

    Attributes
    ----------
    token_ids : Tensor (N,)
        Integer token indices.
    token_type : Tensor (N,)
        TokenType for each token.
    atom_mask : Tensor (N, max_atoms)
        Which atoms are valid for each token.
    atom_element : Tensor (N, max_atoms)
        Element type for each atom.
    residue_index : Tensor (N,)
        Residue index within the chain.
    chain_index : Tensor (N,)
        Chain identifier.
    is_protein : Tensor (N,)
        Boolean mask for protein tokens.
    is_ligand : Tensor (N,)
        Boolean mask for ligand tokens.
    """
    token_ids: Tensor
    token_type: Tensor
    atom_mask: Tensor
    atom_element: Tensor
    residue_index: Tensor
    chain_index: Tensor
    is_protein: Tensor
    is_ligand: Tensor


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

def tokenize_protein_sequence(
    sequence: str,
    chain_id: int = 0,
    max_atoms_per_residue: int = 14,
) -> TokenizedInput:
    """Tokenize a protein amino acid sequence.

    Parameters
    ----------
    sequence : str
        One-letter amino acid sequence.
    chain_id : int
        Chain identifier.
    max_atoms_per_residue : int
        Maximum heavy atoms per residue (14 covers all standard AAs).

    Returns
    -------
    TokenizedInput
    """
    n = len(sequence)
    token_ids = torch.tensor(
        [AA_TO_INDEX.get(aa, UNK_AA_INDEX) for aa in sequence],
        dtype=torch.long,
    )
    token_type = torch.full((n,), TokenType.PROTEIN, dtype=torch.long)
    residue_index = torch.arange(n, dtype=torch.long)
    chain_index = torch.full((n,), chain_id, dtype=torch.long)
    is_protein = torch.ones(n, dtype=torch.bool)
    is_ligand = torch.zeros(n, dtype=torch.bool)

    # Atom-level (simplified: mark backbone atoms as present)
    atom_mask = torch.zeros(n, max_atoms_per_residue, dtype=torch.bool)
    atom_element = torch.zeros(n, max_atoms_per_residue, dtype=torch.long)
    for i, aa in enumerate(sequence):
        n_atoms = AA_HEAVY_ATOM_COUNTS.get(aa, 5)
        n_atoms = min(n_atoms, max_atoms_per_residue)
        atom_mask[i, :n_atoms] = True
        # Simplified: N, CA, C, O backbone atoms
        atom_element[i, 0] = ELEMENT_TO_INDEX["N"]
        atom_element[i, 1] = ELEMENT_TO_INDEX["C"]
        atom_element[i, 2] = ELEMENT_TO_INDEX["C"]
        atom_element[i, 3] = ELEMENT_TO_INDEX["O"]

    return TokenizedInput(
        token_ids=token_ids,
        token_type=token_type,
        atom_mask=atom_mask,
        atom_element=atom_element,
        residue_index=residue_index,
        chain_index=chain_index,
        is_protein=is_protein,
        is_ligand=is_ligand,
    )


def tokenize_ligand_atoms(
    num_atoms: int,
    elements: list[str],
    chain_id: int = 1,
) -> TokenizedInput:
    """Tokenize a small-molecule ligand at the atom level.

    Each atom becomes one token. In the full system, this would use
    CCD-aware featurization; here we provide a simplified version.

    Parameters
    ----------
    num_atoms : int
        Number of heavy atoms (must be in [6, 40] per Section 4.1).
    elements : list[str]
        Element symbol for each atom.
    chain_id : int
        Chain identifier.

    Returns
    -------
    TokenizedInput
    """
    n = num_atoms
    token_ids = torch.tensor(
        [ELEMENT_TO_INDEX.get(e, UNK_ELEMENT_INDEX) for e in elements],
        dtype=torch.long,
    )
    token_type = torch.full((n,), TokenType.LIGAND, dtype=torch.long)
    residue_index = torch.arange(n, dtype=torch.long)
    chain_index = torch.full((n,), chain_id, dtype=torch.long)
    is_protein = torch.zeros(n, dtype=torch.bool)
    is_ligand = torch.ones(n, dtype=torch.bool)

    # Each ligand atom token has exactly 1 atom
    atom_mask = torch.zeros(n, 1, dtype=torch.bool)
    atom_mask[:, 0] = True
    atom_element = token_ids.unsqueeze(-1)

    return TokenizedInput(
        token_ids=token_ids,
        token_type=token_type,
        atom_mask=atom_mask,
        atom_element=atom_element,
        residue_index=residue_index,
        chain_index=chain_index,
        is_protein=is_protein,
        is_ligand=is_ligand,
    )


def merge_tokenized_inputs(*inputs: TokenizedInput) -> TokenizedInput:
    """Merge multiple tokenized inputs into a single representation.

    Used to combine protein chains and ligands into a unified input.
    """
    # Determine max atom dimension
    max_atoms = max(inp.atom_mask.shape[-1] for inp in inputs)

    def pad_atoms(t: Tensor, target: int) -> Tensor:
        if t.shape[-1] >= target:
            return t
        pad_size = target - t.shape[-1]
        return torch.nn.functional.pad(t, (0, pad_size))

    return TokenizedInput(
        token_ids=torch.cat([inp.token_ids for inp in inputs]),
        token_type=torch.cat([inp.token_type for inp in inputs]),
        atom_mask=torch.cat([pad_atoms(inp.atom_mask, max_atoms) for inp in inputs]),
        atom_element=torch.cat(
            [pad_atoms(inp.atom_element, max_atoms) for inp in inputs]
        ),
        residue_index=torch.cat([inp.residue_index for inp in inputs]),
        chain_index=torch.cat([inp.chain_index for inp in inputs]),
        is_protein=torch.cat([inp.is_protein for inp in inputs]),
        is_ligand=torch.cat([inp.is_ligand for inp in inputs]),
    )
