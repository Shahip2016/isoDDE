# Benchmark Reproduction Guide

This guide describes how to reproduce the benchmark results described in the IsoDDE technical report using the built-in validation framework.

## Available Benchmarks

1. **Runs N' Poses (Section 4.1)**
   - Assessment of protein-ligand structural accuracy (cofolding).
   - Computes RMSD and pocket-aligned RMSD for ligands.

2. **Antibody-Antigen Complexes (Section 4.2)**
   - Assessment of CDR-H3 loop accuracy.
   - Computes CDR-H3 backbone RMSD after framework alignment.

3. **ChEMBL temporal-split (Section 4.4)**
   - Evaluation of binding affinity prediction.
   - Splits targets temporally (cutoff 2021-09-30) and evaluates Pearson correlation.

4. **FoldBench (Section 4.5)**
   - Evaluation of structural accuracy on novel folds.
   - Computes backbone RMSD and local distance difference test (LDDT).

## Running Evaluations

You can load and evaluate predictions using the dataset loaders under `isodde.evaluation.benchmarks`:

```python
from isodde.evaluation.benchmarks import RunsNPosesBenchmark
from isodde.evaluation.metrics import compute_pocket_aligned_rmsd

# Load benchmark test cases
benchmark = RunsNPosesBenchmark()
test_cases = benchmark.load_test_cases()

for case in test_cases:
    # Run pipeline prediction
    # ...
    # Compute metrics
    # ...
```
