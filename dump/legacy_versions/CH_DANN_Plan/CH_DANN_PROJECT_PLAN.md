# Causal-Hypergraph Domain-Adversarial Integration for Neonatal Sepsis Detection

## Revised Project Plan (V2 — Critically Reviewed)

**Date:** 2026-02-17
**Based On:** "Causal-Hypergraph Domain-Adversarial Integration" Proposal (V1)
**Revision Purpose:** Critical audit of V1 claims against empirical project data, correction of factual errors, feasibility re-scoping, and addition of Chain of Verification (CoVe) checkpoints.

---

## 0. Critical Review of the Original Proposal

Before presenting the revised plan, we document every claim in V1 that was **kept**, **corrected**, or **rejected**, and **why**. This section is the "audit trail" proving intellectual honesty.

### 0.1 What We KEPT (Strong Ideas)

| # | Claim / Idea | Verdict | Rationale |
|---|---|---|---|
| K1 | **Hypergraph formulation** using KEGG/Reactome pathways as hyperedges | ✅ **Keep** | Biologically principled. Pathway-level aggregation genuinely captures polyadic interactions (e.g., TLR4→MYD88→NFKB1 as one hyperedge). Superior to pairwise STRING edges for modeling cascades. |
| K2 | **Domain Adversarial Training (DANN)** concept | ✅ **Keep (with caveats)** | The *theory* is sound: force the feature extractor to produce scanner-invariant embeddings. However, V1 ignores that **we already tried DANN in Phase 1 and it failed** (see K2-caveat below). |
| K3 | **ForwardGNN** (ICLR 2024, Park et al.) | ✅ **Keep** | Verified: real paper, published at ICLR 2024. Uses top-down signaling for layer-local training without backpropagation. Legitimate efficiency gain for streaming/edge deployment. |
| K4 | **External "Pediatric Challenge"** validation on GSE26440 | ✅ **Keep** | This is the gold standard of our validation strategy. Zero-shot cross-age generalization is the strongest proof of biological validity. |
| K5 | **Motivation & Clinical Framing** (blood culture delays, AMR) | ✅ **Keep** | Excellent, accurate, and compelling. |
| K6 | **Causal Discovery** concept (pruning spurious edges) | ✅ **Keep (re-scoped)** | The *idea* of de-confounding is good, but the PC algorithm with N=320 is statistically underpowered. Re-scoped to use literature-curated causal priors instead. |

**K2-Caveat (DANN Failure History):**

> [!WARNING]
> **This project has already tried DANN.** Results from `ISEF_GNNs/SUMMARY.md`:
> - Mixed-Domain + DANN: Internal AUC **0.86** / No external test
> - DANN provided **zero incremental gain** over simple Rank Normalization (both 0.86)
> - `Master_Project_Plan.md` explicitly states: *"DANN requires the feature extractor to see both domains during training with a balanced domain signal. When training data itself mixes two platforms within a single batch label, the domain discriminator receives corrupted supervision."*
>
> **Implication:** DANN is not a silver bullet. V2 retains DANN but as an **ablation study**, not as the primary defense. The primary defense is **upstream data harmonization** (ComBat/Rank-in).

---

### 0.2 What We CORRECTED (Factual Errors / Misleading Claims)

| # | V1 Claim | Correction | Evidence |
|---|---|---|---|
| C1 | *"GSE69686 Platform: Affymetrix Human Transcriptome Array 2.0 (GPL20292)"* | **Partially correct, but originally listed as GPL570 in Master_Project_Plan.md.** The actual platform is GPL20292 (Affy HTA 2.0). V1 is correct here; our earlier docs were wrong. | `SOLUTION.md` line 4: *"AUC 0.52 on external testing (Agilent GPL20292)"*. The Master Plan incorrectly listed GPL570. |
| C2 | *"GSE26440: 98 children with septic shock and 32 controls"* | Sample count is approximately correct (~130 total). Our Master Plan says ~130 samples. | `Master_Project_Plan.md` line 91. |
| C3 | *"Rank-in outperforms ComBat in maintaining cluster purity"* | **Misleading.** Rank-in is a *supervised* method (it uses outcome labels during integration), which can inflate Type-I error rates and introduce bias. ComBat with `mod=Condition` is the established standard for batch correction in low-N transcriptomics. | Rank-in paper caveat: *"supervised methods might introduce artificial bias by incorporating outcome information"* (BioRxiv benchmarking). ComBat is mathematically validated for N≥3 per batch with Empirical Bayes shrinkage. |
| C4 | *"ComBat can be aggressive, removing biological signal if experimental designs are unbalanced"* | **True only without the `mod` covariate.** Passing `mod=Condition` to ComBat explicitly protects biological signal. V1 fails to mention this critical parameter. | `Master_Project_Plan.md` Section 2.1: *"CRITICAL caveat: You MUST pass the biological covariate (Sepsis/Control) to ComBat's mod parameter."* |
| C5 | V1 claims *"GSE25504 contains approximately 170 samples"* with Illumina subset of *"63 samples (28 sepsis, 35 controls)"* | The total is ~170, but after removing "Unknown" labels, the usable count is lower (~63 from Illumina + ~107 from Affymetrix, minus unknowns). V1 does not mention the "Unknown" label problem (107/170 samples labeled Unknown). | `Master_Project_Plan.md` Section 2.3: *"107/170 GSE25504 samples are currently labeled 'Unknown' and excluded."* |
| C6 | V1 states the Bi-GNN validation *"lacks rigorous external testing"* | Cannot verify without the specific Bi-GNN paper. This claim may be accurate for many GNN papers but should be stated carefully. | General knowledge; paper not provided for audit. |

---

### 0.3 What We REJECTED (Infeasible / Unsupported)

| # | V1 Claim | Rejection Reason |
|---|---|---|
| R1 | **MIMIC-III physiological waveform integration** with GEO transcriptomic data | **Fatal flaw: Zero patient overlap.** MIMIC-III patients are ICU adults/some neonates at Beth Israel Deaconess Medical Center. GEO datasets (GSE25504, GSE69686, GSE26440) are from entirely different institutions and patient cohorts. There is **no link** between a MIMIC-III patient ID and a GEO sample ID. You cannot construct a multi-modal graph with nodes from both modalities for the *same patient*. This would require fabricating an alignment that does not exist. |
| R2 | **CHB-MIT → MIMIC-III EEG Transfer Learning** | **Already empirically disproven.** Our Phase 3 results: Frozen transfer AUC = **0.52** (random). Unfrozen: internal AUC **0.90** / external AUC **0.33** (catastrophic overfitting). `RESEARCH_PLAN.md`: *"The 'structure' learned from EEG (correlation) does not map to Sepsis (PPI)."* The hypothesis that "brain network collapse ≈ immune network collapse" has been **falsified by our own experiments**. |
| R3 | **Causal Discovery with PC Algorithm / NOTEARS at N=320** | **Statistically underpowered.** The PC algorithm relies on conditional independence tests (e.g., Fisher's Z). With ~320 samples and ~2000 variables, the tests are severely underpowered, leading to either (a) almost no edges detected or (b) massive false discovery rates. NOTEARS requires the number of samples to exceed the number of variables for reliable structure learning. |
| R4 | **"Causal Digital Twin" / Counterfactual reasoning** | **Overpromising.** A classification model cannot simulate interventions. Counterfactual reasoning requires a structural causal model (SCM), which in turn requires either randomized experiments or strong assumptions (e.g., faithfulness) that are untestable with N=320. This framing sets expectations the project cannot meet. |
| R5 | **"Physiological Hyperedges" connecting HR, RR, SpO2** | **Dependent on R1.** Without MIMIC-III patient overlap, there are no physiological variables to create hyperedges from. |
| R6 | **NVIDIA A100 GPUs** | **Unrealistic for an ISEF project.** The model should be designed to train on consumer GPUs (e.g., RTX 3060/4060) or Google Colab. |
| R7 | **Rank-in as PRIMARY normalization (replacing ComBat)** | **Rank-in is supervised — it leaks labels into preprocessing.** In a rigorous pipeline, any step that uses the outcome variable before train/test splitting is at risk of data leakage. ComBat with `mod=Condition` is the safer default. Rank-in can be tested as an *ablation*, not a primary strategy. |

---

## 1. Introduction: The Clinical & Computational Imperative

*(Retained from V1 — the framing is accurate and compelling.)*

Neonatal sepsis is a leading cause of NICU mortality, characterized by a dysregulated host immune response to infection. Clinical symptoms (temperature instability, lethargy, feeding intolerance) are nonspecific and overlap with non-infectious conditions like NEC or prematurity-related metabolic disturbances.

### 1.1 The Blood Culture Problem

- **Gold Standard:** Blood culture.
- **Turnaround:** 24–48 hours — retrospective, not actionable in the acute phase.
- **Failure Modes:** High false-negative rates due to low-level bacteremia, prior antibiotic exposure, and insufficient blood volumes from preterm infants.
- **Consequence:** Empiric broad-spectrum antibiotics → antimicrobial resistance (AMR) and neonatal microbiome disruption.

### 1.2 The AI Generalization Crisis

Prior models — including our own earlier phases — fail because they learn **technical artifacts** instead of **biology**:

| Phase | Internal AUC | External AUC | What Went Wrong |
|-------|-------------|-------------|-----------------|
| Phase 1 (Naive GAT + DANN) | 0.97 | 0.50 | Learned Illumina scanner noise |
| Phase 2 (Optimized GCN) | 0.68 ± 0.09 | Not tested | High variance; GCN < Logistic Regression baseline |
| Phase 3 (EEG Transfer, Frozen) | 0.52 | 0.50 | EEG structure ≠ PPI structure |
| Phase 3 (EEG Transfer, Unfrozen) | 0.90 | 0.33 | Catastrophic forgetting |
| Phase 3 (Mixed + Rank Norm) | 0.86 | — | Best so far, but no external test |
| Phase 3 (Mixed + DANN) | 0.86 | — | Zero incremental gain over Rank Norm |

**Lesson:** The bottleneck is **data quality and batch effects**, not model complexity. Adding more architectural sophistication (DANN, Transfer Learning, TransformerConv) without fixing the data does not help.

### 1.3 What This Plan Proposes

A **Pathway-Hypergraph GNN** with:

1. **Upstream data harmonization** (ComBat with `mod=Condition` — primary; Rank-in — ablation)
2. **Hypergraph convolution** over KEGG/Reactome pathway hyperedges (not just pairwise STRING edges)
3. **DANN** as an ablation study (not primary defense)
4. **ForwardGNN** for efficiency (optional, advanced)
5. **Literature-curated causal priors** for edge pruning (instead of underpowered statistical causal discovery)

---

## 2. The Data Landscape

### 2.1 Transcriptomic Datasets

| Dataset | GEO ID | Platform | GPL | Usable Samples | Role |
|---------|--------|----------|-----|-----------------|------|
| **Batch A** | GSE25504 | Illumina HumanHT-12 V3 | GPL6947 | ~63 | Training (Source Domain 1) |
| **Batch B** | GSE25504 | Affymetrix HG-U133 Plus 2.0 | GPL570 | ~107 (after "Unknown" recovery) | Training (Source Domain 2) |
| **Batch C** | GSE69686 | Affymetrix Human Transcriptome Array 2.0 | GPL20292 | 149 (64 Sepsis / 85 Control) | Training (Target Domain for DANN) |
| **External** | GSE26440 | Affymetrix HG-U133 Plus 2.0 | GPL570 | ~130 (98 Sepsis / 32 Control) | **Strict holdout — NEVER in training** |

**Total Training N:** ~319 (after harmonization)
**External Validation N:** ~130

> [!IMPORTANT]
> **Platform Heterogeneity:** GSE69686 uses GPL20292 (Affy HTA 2.0), which is a *different* Affymetrix platform from GPL570. This means we have **3 distinct platforms** across training data, not 2. The DANN must discriminate among 3 domains.

> [!WARNING]
> **The "Unknown" Label Problem:** 107/170 GSE25504 samples were originally labeled "Unknown". Our `02_merge_combat.py` script recovers many of these using title-prefix parsing (Con→Control, Inf→Sepsis, NEC/Vir→Sepsis, Sus→Control). Verify recovery rate before proceeding.

### 2.2 Why MIMIC-III Integration Is NOT Included

V1 proposes integrating MIMIC-III physiological waveforms. This is **rejected** because:

1. **No patient overlap:** GEO samples and MIMIC-III records come from completely different institutions and patient populations. There is no identifier linking a GSE25504 sample to a MIMIC waveform.
2. **No meaningful multi-modal fusion possible:** Without paired (transcriptomic + physiological) data from the same patient, creating multi-modal hyperedges is scientifically invalid — it would amount to randomly pairing unrelated patients.
3. **Scope:** This project is a transcriptomic classifier. Physiological integration is a separate project requiring prospective data collection with paired modalities.

### 2.3 Why EEG Transfer Learning Is NOT Included

V1 proposes pre-training on CHB-MIT EEG and fine-tuning for sepsis. This is **rejected** based on empirical evidence from our own experiments:

- **Frozen Transfer:** AUC 0.52 (random guessing).
- **Unfrozen Transfer:** Internal AUC 0.90 / External AUC 0.33 (catastrophic overfitting).
- **Root Cause:** EEG correlation graphs (temporal, 23 nodes, 10 spectral features) and PPI graphs (static, ~1500 nodes, 1 expression feature) share no structural homology. The hypothesis that "brain network collapse ≈ immune network collapse" has been **falsified**.

---

## 3. Data Harmonization Strategy

### 3.1 Primary: ComBat with Biological Covariate

ComBat models each gene's batch effect as:

$$Y_{ijg} = \alpha_g + X\beta_g + \gamma_{ig} + \delta_{ig}\epsilon_{ijg}$$

Where $\gamma_{ig}$ (additive batch) and $\delta_{ig}$ (multiplicative batch) are shrunk toward cross-gene Empirical Bayes estimates.

**Critical implementation detail:**
```python
from combat.pycombat import pycombat

corrected = pycombat(
    data=combined_expression,       # genes × samples DataFrame
    batch=batch_labels,             # ['GSE25504_Affy', 'GSE25504_Illu', 'GSE69686', ...]
    mod=covariate_matrix            # pd.DataFrame with 'Condition' column (1=Sepsis, 0=Control)
)
```

The `mod=Condition` parameter **protects biological signal** from being removed during batch correction. Without it, ComBat may treat the sepsis signal as batch noise if class proportions differ across batches.

### 3.2 Ablation: Rank-in Normalization

Rank-in transforms expression values to within-sample ranks, then applies SVD to remove platform effects. It is tested as an **alternative** to ComBat, not a replacement.

> [!CAUTION]
> **Rank-in is a supervised method** — it uses outcome labels during integration, which risks inflating performance estimates if not handled within cross-validation folds. If used, it must be applied **inside** each CV fold, never on the full dataset before splitting.

### 3.3 CoVe: Post-Harmonization Verification

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| **Batch mixing** | PCA colored by Batch | Batch clusters must overlap (no visible separation) |
| **Biology preserved** | PCA colored by Condition | Sepsis/Control should show partial separation |
| **Variance attribution** | PVCA | % variance explained by "Batch" < 5% |
| **DE preserved** | Wilcoxon rank-sum test on top 50 DEGs | At least 40/50 DEGs remain significant (p < 0.05) after correction |

---

## 4. Graph Construction: From Pairwise to Pathway-Level

### 4.1 Standard Graph (Baseline — STRING PPI)

This is our established baseline from Phase 2:

- **Nodes:** Top 2,000 MAD-filtered genes present in STRING v12.
- **Edges:** STRING interactions with combined_score > 700.
- **Features:** ComBat-corrected log2 expression (1 feature/node).
- **Result:** ~1,491 nodes, ~18,482 edges.

### 4.2 Hypergraph (Novel — KEGG/Reactome Pathways)

**The Innovation:**

Instead of connecting genes in pairs (Gene A ↔ Gene B), we connect *entire pathways* as single hyperedges:

```
Hyperedge "Toll-Like Receptor Signaling" = {TLR4, MYD88, NFKB1, IRAK1, TRAF6, ...}
Hyperedge "Neutrophil Degranulation" = {MPO, MMP9, S100A8, S100A9, CEACAM8, ...}
Hyperedge "Complement Cascade" = {C3, C5, CFB, CFD, MASP1, ...}
```

**Construction Protocol:**

1. Take the top 2,000 MAD-filtered genes (same as baseline).
2. Query MSigDB (C2:KEGG, C2:Reactome) for all pathways containing ≥ 3 of these genes.
3. Each qualifying pathway becomes a hyperedge connecting all its member genes in our gene set.
4. Genes with zero pathway membership are connected via STRING pairwise edges as fallback.

**Incidence Matrix $H$:**

$H_{ij} = 1$ if gene $i$ belongs to pathway hyperedge $j$, else 0.

**Hypergraph Convolution:**

$$X^{(l+1)} = \sigma \left( D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X^{(l)} \Theta^{(l)} \right)$$

Using `torch_geometric.nn.HypergraphConv`.

### 4.3 Causal Priors (Literature-Curated, Not Statistical)

> Instead of running the PC algorithm (underpowered at N=320), we curate causal priors from the literature:

**Tier 1 (Known Sepsis Causal Genes — Must Be Present):**

| Gene | Role | Evidence |
|------|------|----------|
| CD64 (FCGR1A) | Neutrophil activation marker | PMID: 23283738 |
| MMP9 | Extracellular matrix degradation | PMID: 25783525 |
| S100A8 / S100A9 | Calprotectin (innate immunity) | PMID: 26098424 |
| TLR4 | Pathogen recognition | Core innate immunity |
| MYD88 | TLR signaling adaptor | Core innate immunity |
| IL6 | Pro-inflammatory cytokine | Standard sepsis biomarker |
| CXCL8 (IL8) | Neutrophil chemotaxis | Standard sepsis biomarker |

**Causal Pruning Rule:**
- Hyperedges that contain ≥ 2 Tier 1 genes receive a weight boost ($w = 1.5$).
- Hyperedges containing only housekeeping genes (e.g., ribosomal proteins) receive a weight penalty ($w = 0.5$).
- This is a soft prior, not a hard constraint.

### 4.4 CoVe: Graph Construction Verification

| Check | Method | Pass Criteria |
|-------|--------|---------------|
| **Node count** | `len(V)` | 500–2,000 nodes |
| **Hyperedge count** | `len(E_hyper)` | 50–500 pathway hyperedges |
| **Connectivity** | Largest Connected Component | > 80% of nodes |
| **Biomarker coverage** | Tier 1 gene check | ≥ 7/10 Tier 1 genes present in graph |
| **Average hyperedge size** | `mean([len(e) for e in E_hyper])` | 5–50 genes per hyperedge |

---

## 5. Model Architecture

### 5.1 Primary Model: Pathway-HGCN (Hypergraph Convolutional Network)

```
Input: x ∈ ℝ^{N×1}        (N = num_genes, 1 = expression value)
       H ∈ {0,1}^{N×M}    (Incidence matrix, M = num_hyperedges)

Layer 1: HypergraphConv(1 → 64) + BatchNorm + LeakyReLU + Dropout(0.5)
         Output: ℝ^{N×64}

Layer 2: HypergraphConv(64 → 64) + BatchNorm + LeakyReLU + Dropout(0.5)
         Output: ℝ^{N×64}

Pooling: GlobalMeanPool ⊕ GlobalMaxPool → ℝ^{128}

Classifier:
  Linear(128 → 64) + LeakyReLU + Dropout(0.5)
  Linear(64 → 2)
```

**Design Rationale:**
- **2 layers only:** Prevents over-smoothing. Our Phase 2 experiments showed 2 layers outperform 3.
- **64 hidden channels:** Matches Phase 2 optimal.
- **Dropout 0.5:** Aggressively regularize for N=320.
- **No metadata stream:** Clinical metadata (age, sex) is often unavailable across all datasets. Including it risks information leakage.

### 5.2 Ablation Model: Standard GCN on STRING (Baseline)

Same architecture but using `GCNConv` on pairwise STRING edges instead of `HypergraphConv` on pathway hyperedges. This isolates the contribution of hypergraph topology.

### 5.3 Ablation: DANN Component

For the DANN ablation study *only*, we add:

```
Domain Discriminator:
  GradientReversalLayer(λ)
  Linear(128 → 64) + LeakyReLU
  Linear(64 → 3)  # 3 domains: GSE25504_Illu, GSE25504_Affy, GSE69686
```

**Loss:**
$$L = L_{sepsis}(y, \hat{y}) - \lambda \cdot L_{domain}(d, \hat{d})$$

**Lambda Schedule:** $\lambda$ starts at 0 and linearly increases to 1 over the first 50 epochs. This lets the model learn discriminative features before forcing invariance.

> [!NOTE]
> DANN is an **ablation**, not the primary strategy. Prior experiments showed DANN added zero gain over Rank Normalization at N=320. We include it to test whether *Hypergraph + DANN* behaves differently than *Simple Graph + DANN*.

### 5.4 Optional Advanced: ForwardGNN

If time permits, we implement ForwardGNN (Park et al., ICLR 2024) as a training alternative:

- **Advantage:** No backpropagation; each layer trains independently using local "goodness" signals with top-down feedback.
- **Use Case:** Proof-of-concept for edge deployment (tablet/bedside monitor in LMICs).
- **Priority:** LOW. This is a "stretch goal" — get the HGCN working first.

---

## 6. Training Protocol

### 6.1 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Decoupled weight decay; better generalization |
| Learning Rate | 1e-3 | Matched Phase 2 best |
| Weight Decay | 1e-4 | L2 regularization for small N |
| LR Scheduler | CosineAnnealingWarmRestarts(T_0=30) | Periodic restarts escape local minima |
| Epochs | 150 | With early stopping patience=20 |
| Batch Size | 16 | Balanced gradient noise and memory |
| Dropout | 0.5 | Aggressive for N=320 |
| Hyperedge Dropout | 0.1 | Randomly drop 10% of hyperedges per forward pass |
| Feature Noise | 0.05 | Gaussian noise σ=0.05 on node features during training |
| Loss | CrossEntropyLoss | With inverse-frequency class weights |

### 6.2 Data Augmentation (On-the-Fly)

```python
def augment_hypergraph(data, hedge_drop_rate=0.1, noise_std=0.05):
    # Hyperedge dropout: randomly remove hyperedges
    hedge_mask = torch.rand(data.num_hyperedges) > hedge_drop_rate
    data.hyperedge_index = data.hyperedge_index[:, hedge_mask]

    # Feature noise
    data.x = data.x + torch.randn_like(data.x) * noise_std

    return data
```

### 6.3 Internal Validation: Stratified 5-Fold CV

**Scope:** Training data only (GSE25504 Illumina + GSE25504 Affy + GSE69686, N≈319).

**Stratification:** By `Condition` AND `Batch` to ensure proportional representation in each fold.

```python
from sklearn.model_selection import StratifiedKFold

strat_key = [f"{cond}_{batch}" for cond, batch in zip(conditions, batches)]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Metrics per fold:** AUROC, F1, Accuracy, Sensitivity, Specificity.

### 6.4 External Validation: One-Shot Holdout

**Scope:** GSE26440 (Pediatric, N≈130). Evaluated **once** after all internal CV is finalized. **No hyperparameter adjustments based on external results.**

### 6.5 Success Criteria

| Metric | Internal CV Target | External Target | Stretch Goal |
|--------|-------------------|----------------|-------------|
| AUROC | ≥ 0.78 | ≥ 0.65 | ≥ 0.75 |
| F1 | ≥ 0.70 | ≥ 0.55 | ≥ 0.65 |
| Accuracy | ≥ 0.70 | ≥ 0.60 | ≥ 0.70 |

> [!NOTE]
> External AUROC ≥ 0.65 on a *different age group* (pediatric vs neonatal) would be a strong result. The LR baseline at 0.82 internal is the benchmark to beat for the GNN.

---

## 7. Phase-by-Phase Implementation Roadmap

### Phase 1: Data Engineering & Harmonization

| Step | Script | Input | Output | CoVe Checkpoint |
|------|--------|-------|--------|-----------------|
| 1.1 | `01_id_mapping.py` | Raw GEO Series Matrices | `*_mapped.csv`, `*_phenotype.csv` | ✅ Gene counts match expected per platform |
| 1.2 | `02_merge_combat_v2.py` | Mapped CSVs | `combined_expression_combat.csv` | ✅ PCA: batch clusters overlap, condition clusters separate |
| 1.3 | `02b_rank_in_ablation.py` *(new)* | Mapped CSVs | `combined_expression_rankin.csv` | ✅ Same PCA verification as 1.2 |
| 1.4 | Manual | GEO metadata | Recovered "Unknown" labels | ✅ Unknown count ≤ 20 (or justified exclusions) |

**CoVe Gate 1:** Pipeline does NOT proceed unless PCA shows batch mixing AND condition separation.

---

### Phase 2: Graph & Hypergraph Construction

| Step | Script | Input | Output | CoVe Checkpoint |
|------|--------|-------|--------|-----------------|
| 2.1 | `03_variance_filter.py` | ComBat-corrected expression | `top_2000_genes.txt` | ✅ 2000 genes selected; ≥ 7/10 Tier 1 biomarkers present |
| 2.2 | `04_build_string_graph.py` | Filtered genes + STRING v12 | `string_edge_index.pt` | ✅ 5,000–20,000 edges; avg degree 15–30; LCC > 80% |
| 2.3 | `04b_build_hypergraph.py` *(new)* | Filtered genes + MSigDB KEGG/Reactome | `hyperedge_incidence.pt` | ✅ 50–500 hyperedges; avg size 5–50; ≥ 5 immune pathways |
| 2.4 | `04c_build_patient_graphs.py` *(new)* | Expression + edge_index / incidence | `patient_graphs_hyper.pkl` | ✅ N graphs = N samples; node count = K' |

**CoVe Gate 2:** Inspect 3 random patient graphs. Ensure node features vary between patients (not all identical). Verify edge structure is correct.

---

### Phase 3: Baseline & Model Training

| Step | Script | Input | Output | CoVe Checkpoint |
|------|--------|-------|--------|-----------------|
| 3.1 | `05_train_baselines_v2.py` | Tabular expression | LR/RF metrics | ✅ AUROC ≥ 0.78 (establishes ceiling) |
| 3.2 | `06_train_hgcn.py` *(new)* | Patient hypergraphs | 5-fold CV metrics + best model | ✅ Mean AUROC ≥ 0.78 |
| 3.3 | `06b_train_gcn_baseline.py` | Patient STRING graphs | 5-fold CV metrics | ✅ Architecture comparison |
| 3.4 | `06c_train_hgcn_dann.py` *(new)* | Patient hypergraphs + domain labels | 5-fold CV metrics | ✅ DANN ablation: does AUROC improve over 3.2? |

**CoVe Gate 3:** All training runs must save:
- Per-fold AUROC, F1, Acc
- Training + validation loss curves (check for overfitting)
- Best model weights (`.pt` file)

---

### Phase 4: External Validation & Explainability

| Step | Script | Input | Output | CoVe Checkpoint |
|------|--------|-------|--------|-----------------|
| 4.1 | `07_external_validation.py` | Best model + GSE26440 graphs | External AUROC, F1, ROC curve | ✅ AUROC ≥ 0.65 |
| 4.2 | `08_explainability.py` | Best model + GNNExplainer | Top-30 genes, hyperedge weights | ✅ GO enrichment p < 0.05 for immune pathways |

**CoVe Gate 4:** 
- Top 30 attention genes must include ≥ 3 Tier 1 biomarkers.
- If external AUROC < 0.55, document as expected biological domain gap (neonatal → pediatric) rather than model failure.

---

### Phase 5: Documentation & Presentation

| Step | Deliverable | CoVe Checkpoint |
|------|-------------|-----------------|
| 5.1 | Final Results Report | ✅ All numbers match saved CSV/logs |
| 5.2 | PCA Before/After ComBat figure | ✅ Visual inspection |
| 5.3 | ROC curves (internal + external) | ✅ Generated from saved predictions |
| 5.4 | Hyperedge attention heatmap | ✅ Top pathways are biologically interpretable |
| 5.5 | GO enrichment table | ✅ p-values are from hypergeometric test |

---

## 8. Ablation Study Matrix

| Experiment | Architecture | Normalization | DANN | Expected Internal AUROC |
|------------|-------------|---------------|------|------------------------|
| **A1** (Primary) | HGCN (Hypergraph) | ComBat | No | Target: ≥ 0.78 |
| **A2** | HGCN (Hypergraph) | Rank-in | No | Compare vs A1 |
| **A3** | HGCN (Hypergraph) | ComBat | Yes | Compare vs A1 (DANN adds value?) |
| **A4** (Baseline) | GCN (STRING) | ComBat | No | ~0.68 (baseline from Phase 2) |
| **A5** (Baseline) | Logistic Regression | ComBat | N/A | ~0.82 (established ceiling) |
| **A6** (Optional) | ForwardGNN | ComBat | No | Compare inference latency vs A1 |

---

## 9. Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| ComBat over-corrects biological signal | Medium | High | Pass `mod=Condition`; verify with DE preservation test |
| Hypergraph adds no benefit over STRING | Medium | Medium | Ablation A4 vs A1 will quantify; project valid either way |
| GNN still underperforms LR baseline | Medium | Medium | Present as negative result with analysis; LR + pathway features as fallback |
| GSE26440 age confound dominates | High | Medium | Frame as "cross-age generalization test"; partial success is publishable |
| Rank-in leaks labels | Medium | High | Apply strictly inside CV folds; never on full dataset |
| DANN diverges (minimax instability) | Medium | Medium | Lambda schedule (0→1 over 50 epochs); spectral normalization on discriminator |
| Insufficient sample size for stable CV | Medium | Medium | Stratify by batch+condition; report confidence intervals |

---

## 10. Software & Hardware Stack

| Component | Tool | Version / Notes |
|-----------|------|-----------------|
| Graph Operations | PyTorch Geometric (PyG) | `HypergraphConv` layer |
| Batch Correction | `pycombat` | ComBat with `mod=` parameter |
| Causal Priors | Manual curation | KEGG + literature-curated biomarker list |
| Pathway Mapping | MSigDB / `gseapy` | C2:KEGG, C2:Reactome |
| Baselines | scikit-learn | LogisticRegression, RandomForest |
| Visualization | matplotlib, seaborn | PCA, ROC, heatmaps |
| GPU | Google Colab (T4) or consumer GPU (RTX 3060+) | Realistic for ISEF |
| Training Framework | PyTorch 2.x | CUDA 11.8+ |

---

## 11. Key Decision Log

| Decision | Rationale |
|----------|-----------|
| **Drop MIMIC-III integration** | No patient overlap with GEO data; multi-modal fusion is scientifically invalid without paired data |
| **Drop EEG Transfer Learning** | Empirically falsified: frozen AUC 0.52, unfrozen AUC 0.33 external |
| **Drop statistical Causal Discovery (PC/NOTEARS)** | N=320 is statistically underpowered for structure learning on ~2000 variables |
| **Keep Hypergraph (KEGG pathways)** | Biologically principled grouping; forces pathway-level aggregation |
| **ComBat as primary, Rank-in as ablation** | ComBat + `mod` is gold standard for low-N; Rank-in is supervised and risks label leakage |
| **DANN as ablation only** | Prior experiments showed zero gain at N=320; worth retesting with Hypergraph but not defaulting to it |
| **ForwardGNN as stretch goal** | Real paper (ICLR 2024), but lower priority than getting the base model working |
| **2 layers, 64 hidden channels** | Empirically validated in Phase 2 optimization |
| **STRING threshold 700** | Empirically validated: more edges → better performance (0.81 best fold vs 0.74) |
