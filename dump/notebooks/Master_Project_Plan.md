# Master Project Plan: PPI-Guided GNN for Neonatal Sepsis — V2 Reboot

**Version:** 2.0 — Post-Mortem Reboot  
**Date:** 2026-02-11  
**Role:** Lead ML Engineer & Bioinformatician  

---

## 1. Post-Mortem Diagnosis

### What Happened
| Phase | Internal AUC | External AUC | Verdict |
|-------|-------------|-------------|---------|
| Transfer (frozen) | 0.52 | 0.50 | Near-random: EEG structure ≠ PPI structure |
| Transfer (unfrozen) | 0.90 | 0.33 | Memorized Illumina noise |
| Mixed-Domain | 0.78 | — | Improved but leaked platform info |
| + Rank Norm | 0.86 | — | Best so far, but tested on mixed holdout |
| + DANN | 0.86 | — | No incremental gain over Rank Norm |

### Root Cause: The "Frankenstein Dataset"
GSE25504 is **not** a single-platform dataset. It contains:
- **Illumina GPL6947** samples (GSM1404xxx prefix)
- **Affymetrix GPL570** samples (GSM627xxx prefix)

When the 0.97 internal AUC was achieved by training only on Illumina samples, the model learned the Illumina intensity distribution, not sepsis biology. External Affymetrix data presented a fundamentally different feature space → AUC 0.50.

### Why DANN Failed
DANN requires the feature extractor to see *both* domains during training with a balanced domain signal. When training data itself mixes two platforms *within a single batch label*, the domain discriminator receives corrupted supervision. DANN cannot learn what "platform" means when its own labels are wrong.

---

## 2. Critique of the Proposed Solution

### ✅ Verdict: The ComBat-over-DANN strategy is **scientifically sound** with caveats.

### 2.1 Is Replacing DANN with Upstream ComBat Mathematically Sound?

**Yes, and here is why it is the superior choice for this pipeline:**

| Factor | DANN (Model-Centric) | ComBat (Data-Centric) |
|--------|----------------------|----------------------|
| **Sample Requirement** | Needs hundreds per domain for stable adversarial training | Works with N ≥ 3 per batch (Empirical Bayes shrinkage) |
| **Failure Mode** | Mode collapse, oscillation, sensitivity to λ schedule | Over-correction if batch ≈ biology (addressable with covariates) |
| **Transparency** | Black-box; cannot verify what was "removed" | Produces an explicit corrected matrix; PCA visual verification |
| **GNN Compatibility** | Requires multi-task loss + GRL in forward pass | Preprocessing only; GNN sees clean data, no architectural changes |
| **Your N** | N=320 total across 4 batches → too small for stable DANN | Well within ComBat's sweet spot |

**Mathematical justification:** ComBat models each gene's batch effect as:

$$Y_{ijg} = \alpha_g + X\beta_g + \gamma_{ig} + \delta_{ig}\epsilon_{ijg}$$

Where $\gamma_{ig}$ (additive batch) and $\delta_{ig}$ (multiplicative batch) are shrunk toward their cross-gene empirical Bayes estimates. This is exactly the right model for microarray platform effects, which are predominantly probe-level additive shifts and multiplicative scaling differences. DANN, by contrast, tries to learn an invariant *representation* but has no guarantee it removes *all* platform variance without also destroying biological signal.

> [!IMPORTANT]
> **Critical caveat:** You MUST pass the biological covariate (Sepsis/Control) to ComBat's `mod` parameter. Without it, ComBat may remove biological variance that is confounded with batch membership. Since your Sepsis/Control ratios differ across datasets, this is a real risk.

### 2.2 Will ComBat-Normalized Data Break EEG Transfer Learning?

**No, but the benefit of transfer learning will likely be modest. Here's the nuanced analysis:**

The `SeizureGAT` weights in `conv2`/`conv3` were trained on EEG correlation graphs (23 nodes, ~10 spectral features/node). The sepsis PPI graph has ~500-2000 nodes with 1 expression feature/node. The two key considerations:

1. **What transfers:** The attention patterns learned by `TransformerConv` — how to weight neighbors in a message-passing step. These are *structural priors* about how to aggregate neighborhood information. This is architecture-independent of the input feature distribution.

2. **What won't transfer:** The magnitude/scale expectations of the hidden representations. ComBat normalization will shift the input distribution closer to zero-mean unit-variance per gene, which is actually *more compatible* with the pre-trained weights than raw platform-specific values (which varied wildly).

3. **The real question:** Your diagnostic report shows the GCN (no transfer, no attention) at 0.68 AUROC, while the LR baseline hits 0.82. This suggests the graph topology may not be adding signal to the *current* feature space. ComBat harmonization may improve this by making the features more consistent, allowing the GNN to actually learn from topological patterns rather than memorizing per-sample noise.

> [!TIP]
> **Recommendation:** Run an ablation: Train the SepsisGAT **with** and **without** SeizureGAT weight initialization on the ComBat-corrected data. If the gap is < 2% AUROC, transfer learning is not contributing and you should present the from-scratch model as primary (simpler = more believable to judges).

### 2.3 Additional Concerns

> [!WARNING]
> **The "Unknown" Label Problem:** 107/170 GSE25504 samples are currently labeled "Unknown" and excluded. This represents a 34% data loss. Before running the pipeline, you should manually verify these labels from the GEO Series Matrix supplementary file. Many of these may be recoverable ("Suspected", "NEC", "Viral" samples that have clear clinical classifications). Recovering even 50 of these would significantly improve statistical power (N=370 vs N=320).

> [!WARNING]
> **GSE26440 as External Validation — Age Confound:** GSE26440 contains *pediatric* patients (older children), not neonates. This is a different biological population with different immune maturity. A performance drop on this set could be due to real biological differences, not just technical failure. You should frame this carefully: success here would be remarkable and prove cross-age generalization; partial failure is still scientifically valid if framed as an expected domain gap.

---

## 3. Data Engineering Specification

### 3.1 Dataset Inventory

| Dataset | GEO ID | Platform | GPL | Samples (est.) | Role |
|---------|--------|----------|-----|-----------------|------|
| Training Batch A | GSE25504 | Illumina | GPL6947 | ~63 | Train/CV |
| Training Batch B | GSE25504 | Affymetrix | GPL570 | ~107 | Train/CV |
| Training Batch C | GSE69686 | Affymetrix | GPL570 | 149 | Train/CV |
| External Holdout | GSE26440 | Affymetrix | GPL570 | ~130 (pediatric) | **Strict holdout — NEVER in training** |

**Total Training:** ~319 neonatal samples  
**Total External:** ~130 pediatric samples

### 3.2 Platform Splitting Logic

The existing `02_merge_combat.py` already implements batch assignment via sample ID prefix:
```python
def assign_platform_batch(sample_id):
    if sample_id.startswith('GSM627'):     # → GSE25504_Affy (Batch B)
        return 'GSE25504_Affy'
    elif sample_id.startswith('GSM1404'):  # → GSE25504_Illu (Batch A)
        return 'GSE25504_Illu'
    elif sample_id in gse26440_filtered.columns:
        return 'GSE26440_Neo'              # → External (Batch D)
    else:
        return 'GSE69686'                  # → Batch C
```

> [!CAUTION]
> **Verify the prefix-based splitting against actual GPL annotations.** Download the GPL platform annotation for each sample from GEO to confirm that `GSM627xxx = Affymetrix` and `GSM1404xxx = Illumina`. A misassignment here would propagate through the entire pipeline.

### 3.3 ComBat Implementation

**Library:** `pycombat` from the `combat` Python package (already in use in `02_merge_combat.py`)

**Parameters:**
```python
from combat.pycombat import pycombat

# batch: list of platform labels per sample
# mod: biological covariate matrix (CRITICAL — protects biology)
corrected = pycombat(
    data=combined_expression,      # genes × samples DataFrame
    batch=batch_labels,            # ['GSE25504_Affy', 'GSE25504_Illu', 'GSE69686', ...]
    mod=covariate_matrix           # pd.DataFrame with 'Condition' column (Sepsis=1, Control=0)
)
```

**Pre-ComBat Requirements:**
1. Log2-transform all expression values (if not already log-scale)
2. Verify distributions are approximately normal per gene (QQ-plots on 10 random genes)
3. Remove genes with zero variance within any batch (these cause ComBat to fail)
4. Impute remaining NaNs with row-wise (gene-wise) mean

**Post-ComBat Verification:**
1. PCA colored by batch → clusters should overlap
2. PCA colored by condition → Sepsis/Control should separate
3. PVCA (Principal Variance Component Analysis) → % variance explained by "batch" should drop below 5%

### 3.4 Variance Filtering (Curse of Dimensionality Mitigation)

**Problem:** ~20,000 common genes, ~320 samples → p >> n → guaranteed overfitting.

**Strategy:** Select Top-K most variable genes *after* ComBat correction. This ensures we filter on *biological* variance, not platform variance.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Initial genes | ~10,000-20,000 (post-intersection) | Common across all 3 training datasets |
| Variance metric | MAD (Median Absolute Deviation) | More robust to outliers than standard deviation |
| Top-K candidates | 500, 1000, 1500, 2000 | Sweep as hyperparameter |
| **Recommended K** | **1000** | Balances expressivity vs. overfitting; provides ~500 connected PPI nodes after STRING intersection |
| Additional filter | Must have ≥ 1 STRING edge | Isolated nodes cannot participate in message passing |

**Implementation:**
```python
import numpy as np
from scipy.stats import median_abs_deviation

# After ComBat correction
mad_scores = combined_corrected.apply(median_abs_deviation, axis=1)
top_genes = mad_scores.nlargest(K).index.tolist()

# Further filter to genes present in STRING network
top_genes_in_ppi = [g for g in top_genes if g in string_gene_set]
```

### 3.5 External Validation Set Handling

> [!IMPORTANT]
> **GSE26440 must be ComBat-harmonized INTO the training batch space, but NEVER used for training, hyperparameter tuning, or early stopping.**

**Procedure:**
1. Include GSE26440 as a 4th batch during ComBat harmonization (this is the standard approach — ComBat needs to see the target distribution to align it)
2. After ComBat: split the corrected matrix back into Training (Batches A+B+C) and External (Batch D)
3. Store External data separately: `data/processed/external_harmonized.csv`
4. Only evaluate on this set *once* — after all internal CV is finalized and the best model is selected

---

## 4. Graph Construction Specification

### 4.1 STRING Network

| Parameter | Value |
|-----------|-------|
| Database | STRING v12 (9606.protein.links.v12.0.txt) |
| Organism | Homo sapiens (Taxon 9606) |
| Score Threshold | **700** (high confidence) |
| Edge Type | Undirected, unweighted |
| ID Mapping | ENSP → Gene Symbol (via `mygene` or STRING aliases) |

> [!NOTE]
> Previous experiments used threshold 900 (~10,700 edges, avg degree 28.3) and threshold 700 (~18,482 edges, avg degree 24.8). The 700 threshold yielded better GCN performance (0.81 best fold vs 0.74). **Stick with 700.**

### 4.2 Graph Construction per Patient

For each patient sample $k$:

$$G_k = (V, E, X_k)$$

| Component | Definition |
|-----------|-----------|
| $V$ | Top-K variance-filtered genes that exist in STRING (~500-1500 nodes) |
| $E$ | STRING interactions among $V$ at confidence ≥ 700 |
| $X_k$ | ComBat-corrected, per-sample Z-scored expression values |
| $y_k$ | Binary label: 0 = Control, 1 = Sepsis |

**Node Features ($X_k$):**
- **Primary:** ComBat-corrected log2 expression value (1 feature per node)
- **Optional enrichment (Phase 2):** Add gene-level statistics as additional features:
  - Per-gene MAD rank (encodes variance importance)
  - Node degree in STRING (encodes network centrality)
  - This would give 3 features/node → may improve GNN performance

**Edge Construction:**
- Static across all patients (PPI topology does not change per sample)
- Store as a single `edge_index` tensor of shape [2, num_edges]
- No edge features (unweighted); could add STRING confidence score as `edge_attr` in future

### 4.3 Graph Statistics Target

| Metric | Target Range |
|--------|-------------|
| Nodes | 500–1500 |
| Edges | 5,000–20,000 |
| Average degree | 15–30 |
| Connected components | 1 (largest component only) |
| Density | < 0.05 (sparse) |

---

## 5. Model Architecture Specification

### 5.1 Primary Model: SepsisGAT v2 (Simplified)

Remove the DANN components (GradientReversalLayer, domain classifier). The architecture post-reboot:

```
Input: x ∈ ℝ^{N×F}   (N = num_nodes, F = num_features_per_node)
       edge_index ∈ ℤ^{2×E}

Layer 1: TransformerConv(F → 64, heads=4) + LayerNorm + LeakyReLU + Residual
         Output: ℝ^{N×256}

Layer 2: TransformerConv(256 → 64, heads=4) + LayerNorm + LeakyReLU + Residual
         Output: ℝ^{N×256}
         [FREEZE if using transfer learning]

Layer 3: TransformerConv(256 → 64, heads=1, concat=False) + LayerNorm + LeakyReLU
         Output: ℝ^{N×64}
         [FREEZE if using transfer learning]

Pooling: GlobalMeanPool ⊕ GlobalMaxPool → ℝ^{128}

Classifier:
  Linear(128 → 64) + LeakyReLU + Dropout(0.5)
  Linear(64 → 2)
```

**Key changes from v1:**
1. **Removed:** `GradientReversalLayer`, `domain_lin1`, `domain_lin2`
2. **Removed:** `meta_proj` (metadata stream) — clinical metadata (age, sex) is often unavailable for external datasets; removing it prevents information leakage
3. **Simplified:** Classifier input is `128` (pool only) instead of `128 + 64` (pool + meta)
4. **Increased dropout:** 0.3 → 0.5 for stronger regularization

### 5.2 Transfer Learning Protocol

**Pre-trained weights:** `ISEF_GNNs/best_gat_eeg.pt` (SeizureGAT, trained on CHB-MIT EEG)

**Transfer strategy:**
```python
transfer_layers = ['conv2', 'ln2', 'conv3', 'ln3']

# Load only matching layers (shape-compatible)
checkpoint = torch.load('best_gat_eeg.pt', map_location=device)
model_dict = model.state_dict()
final_dict = {
    k: v for k, v in checkpoint.items()
    if k in model_dict
    and v.shape == model_dict[k].shape
    and any(l in k for l in transfer_layers)
}
model_dict.update(final_dict)
model.load_state_dict(model_dict)

# Freeze transferred layers
for name, param in model.named_parameters():
    if any(l in name for l in transfer_layers):
        param.requires_grad = False
```

**Shape compatibility check:**
- `conv2` in SeizureGAT: `TransformerConv(256 → 64, heads=4)` ✅ matches SepsisGAT
- `conv3` in SeizureGAT: `TransformerConv(256 → 64, heads=1)` ✅ matches SepsisGAT
- `conv1` in SeizureGAT: `TransformerConv(10 → 64, heads=4)` ❌ won't match (EEG has 10 features, Sepsis has 1) → re-initialized, as designed

### 5.3 Ablation Model: SepsisGAT v2 (From Scratch)

Same architecture as 5.1, but **no transfer weights loaded**. Xavier uniform initialization for all layers. This is the control model to quantify the value of transfer learning.

### 5.4 Baseline Model: GCN (No Attention)

Retain the existing `SepsisGCN` architecture (3 GCNConv layers) without DANN components. This controls for the value of attention mechanisms.

---

## 6. Training Specification

### 6.1 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Weight decay decoupled from gradient; better generalization |
| Learning Rate | 1e-3 | Matches best result from optimization phase |
| Weight Decay | 1e-4 | L2 regularization for small-sample regime |
| LR Scheduler | CosineAnnealingWarmRestarts(T_0=30) | Periodic restarts help escape local minima |
| Epochs | 150 | With early stopping patience=20 |
| Batch Size | 16 | Balances gradient noise and memory |
| Dropout (classifier) | 0.5 | Aggressive for N=320 |
| Edge Dropout (augmentation) | 0.1 | 10% random edge drop per forward pass |
| Feature Noise (augmentation) | 0.05 | Gaussian noise σ=0.05 added to node features during training |
| Loss Function | CrossEntropyLoss with class weights | Compensate for class imbalance |
| Class Weight | Inverse frequency: `w_c = N / (2 * n_c)` | Auto-balanced |

### 6.2 Data Augmentation (On-the-Fly)

```python
def augment_graph(data, edge_drop_rate=0.1, noise_std=0.05):
    # Edge dropout
    mask = torch.rand(data.edge_index.size(1)) > edge_drop_rate
    data.edge_index = data.edge_index[:, mask]
    
    # Feature noise
    if data.x is not None:
        data.x = data.x + torch.randn_like(data.x) * noise_std
    
    return data
```

### 6.3 Internal Validation: Stratified 5-Fold CV

**Scope:** Training data only (GSE25504 Illumina + GSE25504 Affy + GSE69686, N≈319)

**Procedure:**
1. Stratify by `Condition` (Sepsis/Control) AND `Batch` (platform) to ensure each fold has proportional representation from all batches
2. For each fold:
   - Train on 4 folds (~255 samples)
   - Validate on 1 fold (~64 samples)
   - Record: AUROC, F1, Accuracy, Sensitivity, Specificity
3. Report: Mean ± Std across 5 folds
4. Select best hyperparameters based on mean validation AUROC

**Implementation:**
```python
from sklearn.model_selection import StratifiedKFold

# Create composite stratification key: "Condition_Batch"
strat_key = [f"{cond}_{batch}" for cond, batch in zip(conditions, batches)]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, strat_key)):
    # train on train_idx, validate on val_idx
    ...
```

### 6.4 External Validation: One-Shot Holdout

**Scope:** GSE26440 Pediatric cohort (N≈130)

**Procedure:**
1. After selecting the best model from internal CV (based on mean validation AUROC), retrain on ALL training data (no validation split)
2. Evaluate ONCE on GSE26440
3. Report: AUROC, F1, Accuracy, Sensitivity, Specificity
4. **No hyperparameter adjustments based on external results** — this is a true generalization test

### 6.5 Success Criteria

| Metric | Internal CV Target | External Target | Stretch Goal |
|--------|-------------------|----------------|-------------|
| AUROC | ≥ 0.78 | ≥ 0.65 | ≥ 0.75 |
| F1 | ≥ 0.70 | ≥ 0.55 | ≥ 0.65 |
| Accuracy | ≥ 0.70 | ≥ 0.60 | ≥ 0.70 |

> [!NOTE]
> External AUROC ≥ 0.65 on a *different age group* (pediatric vs neonatal) would be a strong result for an ISEF project. The LR baseline at 0.82 internal is the benchmark to beat for the GNN on internal CV.

---

## 7. Execution Roadmap

### Phase 1: Data Engineering (Scripts 01–02)

| Step | Script | Input | Output | Verification |
|------|--------|-------|--------|-------------|
| 1.1 | `01_id_mapping.py` | Raw GEO Series Matrices | `*_mapped.csv`, `*_phenotype.csv` | Gene counts match expected; no column name collisions |
| 1.2 | `02_merge_combat_v2.py` | Mapped CSVs | `combined_expression_combat.csv`, `combined_metadata.csv` | PCA plots: batch clusters overlap, condition clusters separate |
| 1.3 | Manual | GEO metadata | Recovered "Unknown" labels | Label count = 0 unknowns (or explicitly justified exclusions) |

### Phase 2: Graph Construction (Scripts 03–04)

| Step | Script | Input | Output | Verification |
|------|--------|-------|--------|-------------|
| 2.1 | `03_variance_filter.py` | ComBat-corrected expression | `top_K_genes.txt` | K genes selected; all have ≥ 1 STRING edge |
| 2.2 | `04_create_graphs_v2.py` | Filtered expression + STRING | `patient_graphs_v2.pkl` | N graphs = N samples; node count = K'; edge count within target range |

### Phase 3: Model Training (Scripts 05–06)

| Step | Script | Input | Output | Verification |
|------|--------|-------|--------|-------------|
| 3.1 | `05_train_baselines_v2.py` | Tabular expression | LR/RF metrics | AUROC ≥ 0.78 (establishes ceiling) |
| 3.2 | `06_train_gat_v2.py` | Patient graphs + EEG weights | 5-fold CV metrics + best model | Mean AUROC ≥ 0.78 |
| 3.3 | `06_train_gat_scratch.py` | Patient graphs (no transfer) | 5-fold CV metrics | Ablation comparison |
| 3.4 | `06_train_gcn_v2.py` | Patient graphs | 5-fold CV metrics | Architecture comparison |

### Phase 4: External Validation & Explainability (Scripts 07–08)

| Step | Script | Input | Output | Verification |
|------|--------|-------|--------|-------------|
| 4.1 | `07_external_validation.py` | Best model + GSE26440 graphs | External AUROC, F1, ROC curve | AUROC ≥ 0.65 |
| 4.2 | `08_explainability.py` | Best model + GNNExplainer | Top-30 genes, attention heatmap | GO enrichment p < 0.05 for immune pathways |

### Phase 5: Documentation & ISEF Submission

| Step | Deliverable |
|------|------------|
| 5.1 | Final Results Report (figures, tables, statistical tests) |
| 5.2 | PCA before/after ComBat figure |
| 5.3 | ROC curves (internal CV + external) |
| 5.4 | GNNExplainer subgraph visualization |
| 5.5 | Gene Ontology enrichment table |

---

## 8. File Structure (Rebooted)

```
ppi_gnn_combined_dataset/
├── data/
│   ├── raw/                          # GEO downloads (Series Matrices)
│   └── processed/
│       ├── GSE25504_mapped.csv       # Probe → Gene Symbol mapped
│       ├── GSE69686_mapped.csv
│       ├── GSE26440_Neo_mapped.csv
│       ├── combined_expression_combat.csv   # ComBat-corrected (genes × samples)
│       ├── combined_metadata.csv            # Sample → Condition + Batch
│       ├── external_harmonized.csv          # GSE26440 ComBat-corrected (separate)
│       ├── top_genes.txt                    # Variance-filtered gene list
│       └── patient_graphs_v2.pkl            # PyG Data objects
├── models/
│   └── best_gat_v2.pt               # Best model weights
├── figures/
│   ├── pca_before_combat.png
│   ├── pca_after_combat.png
│   ├── roc_internal_cv.png
│   ├── roc_external.png
│   └── gnnexplainer_subgraph.png
├── logs/
│   └── training_log_v2.md
├── ISEF_GNNs/
│   └── best_gat_eeg.pt               # Pre-trained EEG weights (source)
├── 01_id_mapping.py
├── 02_merge_combat_v2.py              # [MODIFY] Add mod= covariate, improve label parsing
├── 03_variance_filter.py              # [NEW]
├── 04_create_graphs_v2.py             # [MODIFY] Use variance-filtered genes
├── 05_train_baselines_v2.py           # [MODIFY] On ComBat-corrected data
├── 06_train_gat_v2.py                 # [NEW] Clean GAT without DANN
├── 07_external_validation.py          # [NEW]
├── 08_explainability.py               # [NEW]
└── Master_Project_Plan.md             # This document
```

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| ComBat over-corrects, removing biological signal | Medium | High | Pass `mod=Condition` covariate; verify post-ComBat differential expression is preserved |
| Transfer learning provides zero benefit | Medium | Low | Ablation study (5.3) will quantify; project is valid either way |
| GNN still underperforms LR baseline | Medium | Medium | Present as negative result with analysis; LR + PPI features as fallback |
| GSE26440 age confound dominates | High | Medium | Frame as "cross-age generalization test" — partial success is publishable |
| Insufficient sample size for stable CV | Medium | Medium | Stratified by batch+condition; report confidence intervals |
| "Unknown" labels unrecoverable | Low | Medium | Proceed with N=212; document exclusion criteria |

---

## 10. Key Decision Log

| Decision | Rationale |
|----------|-----------|
| **Drop DANN** | Added no incremental benefit over Rank Normalization (both 0.86); too complex for N=320 |
| **Keep Transfer Learning (as ablation)** | Hypothesis-driven; even a null result is informative for the paper |
| **ComBat with biological covariate** | Mathematically superior for small N; transparent and verifiable |
| **MAD over StdDev for variance filtering** | Robust to outliers from batch correction residuals |
| **STRING threshold 700 over 900** | Empirically validated: more edges → better GCN/GAT performance in previous experiments |
| **Pediatric external validation** | Highest standard of generalization; exceeds typical ISEF validation rigor |
