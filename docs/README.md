## Overview

This project predicts solvent-dependent solubility (`logS`) from RDKit
molecular descriptors, solvent information, and temperature. It compares
Random Forest, Extra Trees, and CatBoost regression models.

The project follows this pipeline:

1. **Download data:** `src/download_dataset.py` fetches and verifies the
   BigSolDB v2.0 source file in `data/raw/`.
2. **Prepare datasets:** `src/dataparser.py` filters and processes the source
   data, calculates molecular descriptors, and writes three datasets to
   `data/processed/` for the basic, solvent-descriptor, and all-entry variants.
3. **Train and evaluate models:** the Random Forest, Extra Trees, and CatBoost
   scripts each run four variants (basic, group shuffle, solvent descriptors,
   and all entries), using five-fold molecule-solvent-pair-grouped
   cross-validation and a holdout test split.
4. **Summarize results:** `src/summarize_test_metrics.py` collects metrics from
   all 12 model logs and generates the consolidated results table and charts in
   `results/`.
