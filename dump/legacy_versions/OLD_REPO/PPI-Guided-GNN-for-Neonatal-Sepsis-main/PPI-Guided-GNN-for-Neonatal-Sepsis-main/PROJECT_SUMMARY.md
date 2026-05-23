# PPI-Guided GNN for Neonatal Sepsis: Project Summary

**Date:** 2026-02-13
**Status:** In Progress (Phase 3: Reboot)
**Objective:** Use Graph Neural Networks (GNNs) and Transfer Learning from EEG Seizure Forecasting to improve Neonatal Sepsis prediction from gene expression data.

---

## 1. Project Overview & Hypothesis

### The Core Problem
Neonatal sepsis is a life-threatening condition with non-specific symptoms. Early diagnosis using gene expression is promising but hampered by:
- **Small Sample Sizes:** Clinical datasets are tiny (N < 200).
- **High Dimensionality:** 20,000+ genes vs. ~100 samples (The "Curse of Dimensionality").
- **Platform Effects:** Datasets from different microarray platforms (Illumina vs. Affymetrix) have incompatible feature distributions, causing models to learn technical noise instead of biology.

### The Innovation: Structural Transfer Learning
We hypothesize that **biological network collapse** (sepsis) shares topological features with **neurological network collapse** (epileptic seizures).
- **Source Domain:** EEG data (Seizure detection) — High N, temporal, well-structured.
- **Target Domain:** Gene Expression (Sepsis) — Low N, static, noisy.
- **Method:** Pre-train a GNN (`SeizureGAT`) on EEG data to learn "failure modes" of complex networks, then transfer these weights to Initialize a Sepsis predictor (`SepsisGAT`).

---

## 2. Methodology Evolution (The "Three Phases")

### Phase 1: The "Frankenstein" Era (Initial Attempt)
**Approach:** 
- Merged multiple datasets (GSE25504 Illumina + Affymetrix) into a single "Mega-Dataset".
- Used minimal preprocessing.
- Trained a complex GAT with Domain Adversarial Neural Networks (DANN) to "unlearn" platform differences.

**Results:**
- **Internal Validation (Illumina):** AUC 0.97 (Suspiciously high).
- **External Validation (Affymetrix):** AUC 0.50 (Random guessing).
- **Diagnosis:** The model memorized "Illumina noise" versus "Affymetrix noise". It didn't learn sepsis; it learned which machine processed the sample. This is a classic "Shortcut Learning" failure.

### Phase 2: Optimization & Diagnostics (The Bridge)
**Approach:**
- **Diagnostic Audit:** We recognized the platform confounding.
- **Optimization:** 
    - Switched to **GCN (Graph Convolutional Network)** for stability.
    - Implemented **Edge Dropout (5-10%)** and **Feature Noise** to combat overfitting.
    - Split validation strictly by platform to measure true generalization.
    - Tuned hyperparameters: 2-3 layers, 64 hidden channels, high dropout (0.5).

**Key Results (from Optimization Reports):**
- **GCN Optimized:** Mean AUROC ~0.685.
    - *Best Fold:* 0.81 (Approaching baseline).
    - *Worst Fold:* 0.58 (High instability).
- **Baseline (Logistic Regression):** Mean AUROC ~0.82.
- **Conclusion:** The GNN is *learning* (better than random), but it is not yet beating simple linear baselines. The high variance suggests it is still struggling with the small sample size (N=319).

### Phase 3: The "Reboot" (Current Strategy)
**Document:** `Master_Project_Plan.md` (Version 2.0)
**Key Shift:** Move from **Model-Centric** (DANN) to **Data-Centric** (ComBat) correction.

**New Methodology:**
1.  **ComBat Harmonization:** Explicitly correct batch effects (Illumina vs. Affy) *before* the model sees data. This removes the need for complex DANN architectures.
2.  **Variance Filtering:** Select Top-1000 genes by **Median Absolute Deviation (MAD)** *after* ComBat. This focuses the graph on biologically variable genes, not technical artifacts.
3.  **Simplified Architecture:** 
    - **SepsisGAT v2:** Standard GAT structure (no DANN).
    - **Transfer Learning:** Initialize with `SeizureGAT` weights (Conv2/Conv3 layers).
4.  **Strict External Verification:** 
    - Train on Neonates (GSE25504 + GSE69686).
    - **Test ONLY ONCE** on Pediatrics (GSE26440) to prove cross-age, cross-platform generalization.

---

## 3. Results Summary Matrix

| Metric | Phase 1 (Initial) | Phase 2 (Optimization) | Phase 3 (Target) |
| :--- | :--- | :--- | :--- |
| **Architecture** | GAT + DANN | GCN (2-3 layers) | SepsisGAT v2 (Transfer) |
| **Data Strategy** | Raw Merge | Split Platforms | ComBat Harmonization |
| **Internal AUC** | ~0.97 (Overfit) | ~0.68 (Realistic) | Target: > 0.78 |
| **External AUC** | ~0.50 (Random) | Not Tested | Target: > 0.65 |
| **Stability (Std)** | N/A | High Variance (±0.09) | Low Variance (±0.05) |
| **Key Insight** | Learned Platform ID | Found Signal, High Var | Fix Data First |

---

## 4. Current Status & Next Steps

### Status
- **Planning:** Complete (`Master_Project_Plan.md`).
- **Data Engineering:** `02_merge_combat.py` and `04_create_graphs_enhanced.py` are critical.
- **Model:** `06_train_gnn_optimized.py` represents the Phase 2 best effort. We need to create `06_train_gat_v2.py` for Phase 3.

### Immediate Next Steps (from Task List)
1.  **Implement `06_train_gat_v2.py`:** The clean, transfer-learning enabled training script.
2.  **Verify Data Engineering:** Ensure ComBat uses the `mod=Condition` covariate to preserve biological signal.
3.  **Execute Phase 3 Training:** Run the 5-fold CV with the new ComBat data and SepsisGAT v2.
4.  **External Validation:** Run the one-shot test on GSE26440.

---

## 5. Artifacts Key
- **Plan:** `Master_Project_Plan.md`
- **Results:** `logs/gnn_optimization_final_report.md`
- **Code (Optimized):** `06_train_gnn_optimized.py`
- **Summary:** `ISEF_GNNs/SUMMARY.md`
