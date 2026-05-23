# General_Sepsis_V11 Verification Report

Generated: 2026-03-02

## Scope

This report verifies (1) data provenance + split integrity, (2) metric correctness (including the “AUROC=1.0” concern), and (3) hybrid architecture performance vs baselines across cohort categories available in this experiment.

This verification is limited to the **General_Sepsis_V11** pipeline and its emitted artifacts under `General_Sepsis_V11/results/`.

## Data provenance (no synthetic validation data)

- Raw data files are cached GEO SOFT archives under `General_Sepsis_V11/data/raw/`:
  - `GSE54514_family.soft.gz`
  - `GSE57065_family.soft.gz`
  - `GSE95233_family.soft.gz` *(may be excluded by QC/fallback policy)*
  - `GSE134347_family.soft.gz` *(fallback dataset used when GSE95233 QC fails)*
  - `GSE26378_family.soft.gz` *(external holdout)*
- Sample selection and labeling are implemented in `General_Sepsis_V11/scripts/01_download_and_preprocess.py` using GEO metadata parsing and deterministic inclusion rules.
- No code path generates synthetic expression profiles or synthetic labels for validation. Training-time *graph augmentation* (e.g., edge dropout / noise) may occur, but it does **not** create synthetic validation samples; it perturbs graphs during training only.

## Split integrity / leakage checks

- `General_Sepsis_V11/results/validation_audit_report.md` shows:
  - Train/val disjointness for each fold: **PASS**
  - Patient/group disjointness (StratifiedGroupKFold): **PASS**
  - Co-expression hyperedge construction restricted to fold training IDs: **PASS**
  - Holdout dataset excluded from CV: **PASS**

Dataset counts (from `General_Sepsis_V11/results/metadata.csv`):

- Train: 345 samples (GSE134347=239, GSE54514=53, GSE57065=53)
- Holdout: 103 samples (GSE26378=103)

## Metric correctness and the “AUROC=1.0” issue

Two different CV AUROCs are reported:

- **CV fold-mean AUROC** (mean of per-fold AUROCs): **1.0000**
- **CV pooled OOF AUROC** (single AUROC over all out-of-fold predictions): **0.9954**

Both are now reported in `General_Sepsis_V11/results/validation_audit_report.md`. The pooled OOF AUROC is the more informative single-number summary when fold sizes differ or when per-fold AUROCs saturate.

## Hybrid model vs baselines (exact metrics)

Authoritative tables + plots are in:

- `General_Sepsis_V11/results/metrics_report.md`
- `General_Sepsis_V11/results/metrics_overall.csv`
- `General_Sepsis_V11/results/metrics_by_dataset.csv`
- `General_Sepsis_V11/results/metrics_by_platform.csv`
- `General_Sepsis_V11/results/metrics_external.csv`
- `General_Sepsis_V11/results/plots/`

Key results (from those artifacts):

### Overall (CV OOF; n=345)

- Hybrid (`hybrid_v11`): AUROC **0.9954**, Accuracy **0.9855**, F1 **0.9885**, Precision **1.0000**, Recall **0.9772**
- Best baseline (Logistic Regression): AUROC **1.0000**, Accuracy **0.9942**, F1 **0.9954**

### By dataset (CV OOF)

- Hybrid AUROC:
  - GSE134347: **0.9937** (n=239)
  - GSE54514: **1.0000** (n=53)
  - GSE57065: **1.0000** (n=53)

### External holdout (GSE26378; n=103)

- Hybrid (`hybrid_v11`): AUROC **0.9942**, Accuracy **0.7961**, F1 **0.8865**, Precision **0.7961**, Recall **1.0000**
- Baseline references (refit on full train, then evaluated on holdout):
  - Logistic Regression: AUROC **0.9936**
  - MLP: AUROC **0.9907**

Note: External AUROC is very high, but threshold-dependent metrics (accuracy/precision/recall) indicate the hybrid model’s probabilities are heavily shifted upward on the holdout distribution. This is consistent with a calibration/thresholding issue under domain shift rather than a ranking failure.

