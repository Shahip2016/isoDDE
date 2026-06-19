# IsoDDE — Isomorphic Labs Drug Design Engine

An open-source implementation of the predictive core described in the
[IsoDDE technical report](https://storage.googleapis.com/isomorphic-site-assets/isodde_technical_report.pdf)
(Isomorphic Labs, February 2026).

## Overview

IsoDDE is a unified computational system for predicting biomolecular
interactions, encompassing:

- **Structure Prediction** — Protein-ligand, protein-protein, and
  antibody-antigen cofolding with >2× accuracy over AlphaFold 3 on
  hard generalisation benchmarks (Runs N' Poses, FoldBench).
- **Binding Affinity** — Quantitative ΔG / pKd estimation that
  surpasses gold-standard FEP+ on public benchmarks.
- **Pocket Identification** — Blind detection of ligandable sites,
  including cryptic pockets, from sequence alone.

> **Note:** This repository contains the *architectural framework* faithfully
> implementing the components described in the paper. Pre-trained weights are
> not included.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from isodde.pipeline import IsoDDEPipeline

pipeline = IsoDDEPipeline()

# Structure prediction
result = pipeline.predict_structure(
    protein_sequence="MKTLLILAVL...",
    ligand_smiles="CC(=O)Oc1ccccc1C(=O)O",
)

# Binding affinity
affinity = pipeline.predict_affinity(
    protein_sequence="MKTLLILAVL...",
    ligand_smiles="CC(=O)Oc1ccccc1C(=O)O",
)

# Pocket identification
pockets = pipeline.identify_pockets(
    protein_sequence="MKTLLILAVL...",
)
```

## Architecture

The IsoDDE model follows the trunk-and-heads paradigm:

1. **Input Embedding** — Tokenizes proteins, ligands, nucleic acids
2. **MSA Module** — Processes multiple sequence alignments with improved
   single↔pair information flow
3. **Pairformer** — Triangular attention and multiplicative updates on
   pair representations (wider representations, O(n²) memory)
4. **Diffusion Head** — Score-based generative model for 3D coordinate
   prediction
5. **Confidence Head** — pLDDT, pTM, ipTM for ranking multi-seed
   predictions
6. **Affinity Head** — Structure-conditioned binding free energy
   estimation
7. **Pocket Head** — Residue-level ligandability scoring

## Citation

```bibtex
@techreport{isodde2026,
  title   = {Accurate Predictions of Novel Biomolecular Interactions with IsoDDE},
  author  = {Isomorphic Labs Team},
  year    = {2026},
  month   = {February},
  url     = {https://storage.googleapis.com/isomorphic-site-assets/isodde_technical_report.pdf}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
