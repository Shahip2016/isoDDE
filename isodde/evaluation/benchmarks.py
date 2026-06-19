"""Benchmark dataset loaders and interfaces for IsoDDE evaluation.

Supports:
- Runs N' Poses protein-ligand co-folding benchmark (Section 4.1)
- Antibody-Antigen test set construction (Section 4.2)
- ChEMBL time-split dataset for affinity prediction (Section 4.4)
- FoldBench interface for structural accuracy (Section 4.5)
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import torch


class RunsNPosesBenchmark:
    """Runs N' Poses protein-ligand co-folding evaluation (Section 4.1)."""

    def __init__(self, data_path: Optional[str] = None) -> None:
        self.data_path = data_path

    def load_test_cases(self) -> List[Dict[str, Any]]:
        """Return list of protein-ligand test cases.

        Each test case includes:
            - target_id: str
            - protein_sequence: str
            - ligand_smiles: str
            - ideal_bonds: List of (i, j, type)
            - reference_coords: Tensor (N, 3)
        """
        # Return structured mock dataset conforming to standard test cases
        return [
            {
                "target_id": "RunsNPoses_mock_1",
                "protein_sequence": "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQH",
                "ligand_smiles": "CC(=O)NC1C(O)OC(CO)C(O)C1O",  # N-Acetylglucosamine
                "elements": ["C", "C", "O", "N", "C", "C", "O", "O", "C", "C", "O", "C", "O", "C", "O"],
                "bonds": [
                    (0, 1, "SINGLE"), (1, 2, "DOUBLE"), (1, 3, "SINGLE"), (3, 4, "SINGLE"),
                    (4, 5, "SINGLE"), (5, 6, "SINGLE"), (5, 7, "SINGLE"), (7, 8, "SINGLE"),
                    (8, 9, "SINGLE"), (9, 10, "SINGLE"), (8, 11, "SINGLE"), (11, 12, "SINGLE"),
                    (11, 13, "SINGLE"), (13, 14, "SINGLE"), (13, 4, "SINGLE")
                ],
                "reference_coords": torch.randn(181, 3) * 5.0,  # 166 AA + 15 ligand atoms
            }
        ]


class AntibodyAntigenBenchmark:
    """Antibody-Antigen docking and structure evaluation (Section 4.2)."""

    def __init__(self, data_path: Optional[str] = None) -> None:
        self.data_path = data_path

    def load_test_cases(self) -> List[Dict[str, Any]]:
        """Return antibody-antigen complex structures and metadata."""
        return [
            {
                "target_id": "AbAg_mock_1",
                "heavy_chain": "QVQLQESGPGLVKPSQTLSLTCTVSGGSISSGGYYWSWIRQHPGKGLEWIGYIYYSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARGRDYDFWSGYFDYWGQGTLVTVSS",
                "light_chain": "DIQMTQSPSSLSASVGDRVTITCRASQGISNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQLNSYPLTFGGGTKVEIK",
                "antigen": "MTEYKLVVVGAGGVGKSALTIQLIQN",
                "cdr_h3_indices": list(range(95, 107)),
                "framework_indices": list(range(0, 95)) + list(range(107, 120)),
                "reference_coords": torch.randn(254, 3) * 8.0,  # heavy (120) + light (108) + antigen (26)
            }
        ]


class ChEMBLTimeSplitBenchmark:
    """ChEMBL temporal-split affinity prediction validation (Section 4.4)."""

    def __init__(self, training_cutoff: str = "2021-09-30") -> None:
        self.training_cutoff = training_cutoff

    def load_test_cases(self) -> List[Dict[str, Any]]:
        """Return assay activity points published after the training cutoff."""
        return [
            {
                "assay_id": "CHEMBL_mock_assay_1",
                "protein_sequence": "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQ",
                "ligand_smiles": "CC(=O)NC1C(O)OC",
                "elements": ["C", "C", "O", "N", "C", "C", "O", "O", "C"],
                "target_affinity": 7.8,  # pKd / pKi
                "publication_date": "2023-05-15",
            }
        ]


class FoldBenchBenchmark:
    """FoldBench structure prediction accuracy assessment (Section 4.5)."""

    def __init__(self, data_path: Optional[str] = None) -> None:
        self.data_path = data_path

    def load_test_cases(self) -> List[Dict[str, Any]]:
        """Return structures with low sequence identity to PDB training set."""
        return [
            {
                "target_id": "FoldBench_mock_1",
                "sequence": "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQH",
                "reference_coords": torch.randn(166, 3) * 10.0,
            }
        ]
