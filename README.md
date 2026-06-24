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

### Python API

```python
from isodde.pipeline import IsoDDEPipeline

# Initialize the pipeline (using small config for resource friendliness)
pipeline = IsoDDEPipeline()

# Run end-to-end structure co-folding, pocket identification, and binding affinity estimation
results = pipeline.run_cofolding(
    protein_sequence="MTEYKLVVVGAGGVGKSALTI",
    ligand_elements=["C", "C", "O", "N"],
    num_seeds=3,
)

print(f"Confidence score (pLDDT): {results['pLDDT']:.4f}")
print(f"Predicted pTM:            {results['ptm']:.4f}")
print(f"Predicted Affinity (pKd): {results['binding_affinity_pkd']:.4f}")
print(f"Pockets Identified:       {len(results['pockets'])}")
```

### CLI Interface

```bash
# Run prediction using CLI
py -m isodde.cli --sequence MTEYKLVVVGAGGVGKSALTI --ligand C,C,O,N --output-dir predictions --num-seeds 3
```

### Web Chat UI Interface

IsoDDE includes an interactive web-based Chat UI to visualize protein-ligand complexes, pockets, and contact probability maps:

```bash
# Start the web UI server
py -m isodde.cli --web
```

Once started, open [http://localhost:8000](http://localhost:8000) in your web browser. Features include:
- **Interactive Chat Interface** to submit prediction requests.
- **3D Molecular Visualization** of the co-folded complex via `3Dmol.js`.
- **Blind Pocket Detection Viewer** to interactively highlight pocket centers, zones, and scores.
- **Contact Map Heatmaps** to view residue-level interface contacts and protein-ligand contact matrices.

## Tests

You can run the full test suite using the Python Launcher (`py`):

```bash
py -m pytest tests/ -v
```

## Architecture

The IsoDDE model follows the trunk-and-heads paradigm:

1. **Input Embedding** — Tokenizes proteins, ligands, nucleic acids.
2. **MSA Module** — Processes multiple sequence alignments with improved
   single↔pair information flow (OPM computed before row attention).
3. **Pairformer** — Triangular attention and multiplicative updates on
   pair representations (wider representations, O(n²) memory).
4. **Diffusion Head** — Score-based generative model for 3D coordinate
   prediction.
5. **Confidence Head** — pLDDT, pTM, ipTM for ranking multi-seed
   predictions.
6. **Affinity Head** — Structure-conditioned binding free energy
   estimation.
7. **Pocket Head** — Residue-level ligandability scoring and single-linkage spatial clustering.

For more details, see [docs/architecture.md](docs/architecture.md) and [docs/benchmarks.md](docs/benchmarks.md).

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
