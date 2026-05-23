# Model Card: GCN Merged

## Artifact Details
- **Filename:** `gcn_best_merged.pt`
- **Date:** 2026-01-28
- **Task:** Neonatal Sepsis Classification (Binary: Control vs Sepsis)

## Training Conditions
- **Datasets:** GSE25504 (Affymetrix + Illumina separated), GSE69686.
- **Sample Count:** 319 (186 Control, 133 Sepsis).
- **Preprocessing:**
  - **Batch Correction:** ComBat (3 batches: `GSE25504_Affy`, `GSE25504_Illu`, `GSE69686`).
  - **Normalization:** Z-score standardization per sample.
  - **Feature Selection:** Variance-based (Top 2,000 genes -> Intersection with STRING -> **1,050 genes**).

## Graph Topology
- **Source:** STRING v12 (Physical + Functional).
- **Threshold:** Confidence Score > 0.9 (900).
- **Stats:** 1,050 nodes, Avg Degree ~28.3.
- **Node Features (3D):**
  1. Gene Expression (Z-score)
  2. Degree Centrality (Global)
  3. Gene Variance (Global)

## Hyperparameters (Optimized)
- **Architecture:** GCN (Graph Convolutional Network)
- **Layers:** 2
- **Hidden Channels:** 64
- **Dropout:** 0.5
- **Learning Rate:** 0.0005
- **Batch Size:** 32
- **Epochs:** 100
- **Pooling:** Global Mean Pool

## Performance (Internal 5-Fold CV)
- **Mean AUROC:** 0.6812 (+/- 0.0478)
- **Best Fold AUROC:** 0.7422
- **Comparison:** Outperforms Logistic Regression Baseline (0.576) by +10.5%.

## Constraints
- **Scope:** Strictly neonatal samples (< 28 days).
- **Exclusion:** Pediatric samples (GSE26440) were excluded from training to serve as external validation.
