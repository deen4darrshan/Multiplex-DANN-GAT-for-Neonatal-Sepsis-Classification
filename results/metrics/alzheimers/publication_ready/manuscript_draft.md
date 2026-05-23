# Generalizing a Multiplex Hypergraph Neural Network From Neonatal Sepsis to Alzheimer's Disease Transcriptomics

## Authors
Terry et al.  
Affiliation: [Insert institution]

## Abstract
**Background:** Graph and hypergraph neural networks trained on biomedical transcriptomics often overfit platform- or cohort-specific artifacts. We tested whether a high-performing neonatal sepsis architecture can generalize to Alzheimer's disease (AD) blood and brain transcriptomic cohorts when transferred without redesigning core model structure.

**Objective:** Evaluate whether a multiplex HypergraphConv + MLP architecture (V11 lineage) and its domain-adversarial variant can maintain strong performance on AD classification (AD vs Control) and robustly generalize across cohorts.

**Methods:** We cloned the neonatal sepsis repository architecture and rebuilt AD datasets from raw GEO SOFT files with true cohort domain labels. Brain cohorts included GSE5281, GSE1297, and GSE28146 (n=222). Blood cohorts included GSE63060 and GSE63061 (n=522). Features were top-2000 genes by MAD after cohort-intersection and per-sample z-scoring. Static relation edges (15,000 pairwise, undirected) and KEGG hyperedges were included. We ran (1) 5-fold CV, (2) leave-one-cohort-out (LOCO), and (3) 5-seed stability (seeds: 7, 21, 42, 77, 123), each with and without DANN.

**Results:** On brain cohorts, 5-fold CV reached mean accuracy 0.9053 and mean AUROC 0.9437 (DANN), with best fold accuracy 0.9773. Five-seed stability was strong: mean accuracy 0.9160 (95% CI: 0.9069-0.9250) and mean AUROC 0.9430 (DANN), and 0.9178 accuracy (95% CI: 0.9103-0.9252) and 0.9522 AUROC (no DANN). Under strict LOCO, performance dropped to 0.7344 accuracy / 0.8174 AUROC (DANN) and 0.7277 / 0.7945 (no DANN). Blood transfer remained lower (0.7740-0.7798 accuracy range).

**Conclusion:** The architecture generalizes to AD brain classification at >90% mean CV accuracy and remains stable across seeds, but LOCO results show meaningful residual domain shift. The method generalizes in-distribution and partially out-of-cohort, with clear room for stronger domain-robust training.

**Keywords:** Alzheimer's disease, hypergraph neural network, domain adaptation, transcriptomics, GEO, transfer learning

## 1. Introduction
Transcriptomic disease classifiers often fail to transfer across studies because learned signals mix biology with technical variation. This is a major issue for small-to-moderate biomedical cohorts where platform effects and cohort-specific protocols can dominate model behavior.  

We previously developed a strong neonatal sepsis model lineage in a multiplex hypergraph framework (V11). The key open question is whether that same architecture can transfer to a different disease domain without architecture replacement. Here we evaluate transfer to Alzheimer's disease (AD) classification using public GEO cohorts and true cohort-domain labels.

This study specifically asks:
1. Can the neonatal V11 architecture reach high performance on AD with only data/domain adaptation?
2. Is performance stable across random seeds?
3. How much does strict out-of-cohort generalization (LOCO) degrade?
4. Does DANN materially improve AD transfer?

## 2. Methods

### 2.1 Study Design
We performed an architecture transfer study using the same core model family developed for neonatal sepsis. The model and preprocessing pipeline were applied to AD data with minimal structural changes:
1. Build true-domain AD datasets from raw GEO SOFT files.
2. Train/evaluate the transferred architecture on AD cohorts.
3. Run matched ablations (DANN vs no DANN).
4. Quantify stability via repeated-seed CV.

### 2.2 Datasets
All datasets are public GEO transcriptomic cohorts.

**Brain cohorts (primary transfer target):**
- GSE5281
- GSE1297
- GSE28146
- Final brain sample count: n=222 (AD=131, Control=91)

**Blood cohorts (secondary transfer target):**
- GSE63060
- GSE63061
- Final blood sample count: n=522 (AD=284, Control=238)

Only AD and Control labels were retained for this manuscript.

### 2.3 Preprocessing and Graph Construction
We used a true-domain construction script that:
1. Parses GEO SOFT files and platform tables.
2. Maps probes to gene symbols and aggregates probe-level values by gene mean.
3. Intersects genes across included cohorts.
4. Applies per-sample z-score normalization.
5. Selects top 2000 genes by MAD.
6. Builds graph objects with:
   - `x`: node expression (1 feature per gene),
   - `global_feat`: sample-level expression vector,
   - `domain_y` and `batch_label`: true cohort domain labels,
   - `edge_index`: static top absolute-correlation edges.

For this study:
- Static pairwise relation edges: 15,000 undirected pairs (stored as 30,000 directed edges).
- KEGG hyperedges: 268 (after overlap filtering).
- Co-expression relation during training: threshold 0.70, capped at 20,000 edges per split.

### 2.4 Model
The transferred model is V11 multiplex HGCN + MLP with optional DANN:
1. **Multiplex Hypergraph Branch:** three relation channels (KEGG, static pairwise, dynamic co-expression), each with two HypergraphConv blocks and residual connections.
2. **Relation Attention:** learned soft weighting across relation channels.
3. **Feature-Guided MLP Branch:** gene importance weighting followed by dense classification pathway.
4. **Domain-Adversarial Head (optional):** gradient reversal + domain classifier.

Matched no-DANN experiments set domain loss weight to 0.0.

### 2.5 Evaluation Protocols
We ran three protocols:

1. **5-fold stratified CV (brain and blood)**  
   Metrics: accuracy, macro-F1, macro-AUROC OVR.

2. **LOCO on brain cohorts**  
   Train on two cohorts, test on held-out third cohort.

3. **5-seed stability on brain cohorts**  
   Seeds: 7, 21, 42, 77, 123.  
   For each seed: full 5-fold stratified CV, then aggregate across seeds.

### 2.6 Reproducibility Assets
Core scripts:
- `C:\Users\terry\Downloads\Projects\ISEF\CH_DANN_Plan\scripts\23_prepare_alz_true_domains.py`
- `C:\Users\terry\Downloads\Projects\ISEF\CH_DANN_Plan\scripts\22_train_v11_alzheimers_transfer.py`
- `C:\Users\terry\Downloads\Projects\ISEF\CH_DANN_Plan\scripts\24_evaluate_alz_brain_loco.py`
- `C:\Users\terry\Downloads\Projects\ISEF\CH_DANN_Plan\scripts\25_evaluate_alz_brain_seed_stability.py`
- `C:\Users\terry\Downloads\Projects\ISEF\CH_DANN_Plan\scripts\26_build_publication_summary.py`

## 3. Results

### 3.1 Main Performance Summary
Table 1 summarizes the main AD transfer outcomes.

| Experiment | n | Mean Accuracy | Mean AUROC | Notes |
|---|---:|---:|---:|---|
| Brain 5-fold CV (DANN) | 222 | 0.9053 | 0.9437 | Best fold acc 0.9773 |
| Brain 5-fold CV (No DANN) | 222 | 0.9053 | 0.9424 | Best fold acc 0.9556 |
| Blood 5-fold CV (DANN) | 522 | 0.7740 | 0.8239 | Secondary target |
| Blood 5-fold CV (No DANN) | 522 | 0.7798 | 0.8379 | Secondary target |
| Brain 5-seed CV (DANN) | 222 | 0.9160 | 0.9430 | 95% CI acc: 0.9069-0.9250 |
| Brain 5-seed CV (No DANN) | 222 | 0.9178 | 0.9522 | 95% CI acc: 0.9103-0.9252 |

Interpretation:
1. Brain AD transfer achieves >90% mean CV accuracy.
2. Performance is stable across seeds.
3. DANN does not clearly improve in-distribution CV relative to no-DANN.

### 3.2 Strict Out-of-Cohort Generalization (LOCO)
LOCO performance was notably lower than standard CV:

- **DANN LOCO:** mean accuracy 0.7344, mean AUROC 0.8174
- **No-DANN LOCO:** mean accuracy 0.7277, mean AUROC 0.7945

Per-held-out cohort results:

| Config | Held-out cohort | Accuracy | AUROC |
|---|---|---:|---:|
| DANN | GSE1297 | 0.7742 | 0.7727 |
| DANN | GSE28146 | 0.7333 | 0.8409 |
| DANN | GSE5281 | 0.6957 | 0.8385 |
| No DANN | GSE1297 | 0.6774 | 0.7828 |
| No DANN | GSE28146 | 0.7667 | 0.7670 |
| No DANN | GSE5281 | 0.7391 | 0.8335 |

Interpretation:
1. LOCO confirms substantial cohort shift remains.
2. DANN gives a modest mean LOCO gain over no-DANN in this setup.
3. Transfer is meaningful but not domain-invariant under strict cohort holdout.

### 3.3 Five-Seed Stability
Across five full 5-fold runs on brain cohorts:

- **DANN:** seed-mean accuracy 0.9160 (SD 0.0092), seed-mean AUROC 0.9430 (SD 0.0130)
- **No-DANN:** seed-mean accuracy 0.9178 (SD 0.0076), seed-mean AUROC 0.9522 (SD 0.0076)

Per-seed metrics remained tight, indicating low variance and robust optimization behavior.

### 3.4 Relation Attention Behavior
LOCO relation-attention means suggest relation usage shifts by held-out cohort:
1. In some splits, static pairwise or co-expression channels dominated.
2. In no-DANN GSE5281 holdout, co-expression attention was especially high (~0.95), indicating strong dependence on data-driven edges under cohort shift.
3. DANN produced more mixed attention weights in several LOCO settings.

## 4. Discussion
This study demonstrates that the neonatal V11 multiplex hypergraph architecture generalizes to AD brain transcriptomics with high in-distribution performance (>90% mean CV accuracy), without replacing the core architecture.  

However, LOCO results show that cross-cohort generalization remains much harder than within-cohort stratified CV. This gap indicates the model still captures cohort-specific structure in addition to disease signal. DANN provided modest LOCO improvements in this implementation, but not a clear advantage in standard CV or seed-stability averages.

The blood transfer results (~0.78 accuracy) further emphasize disease-context and tissue-context sensitivity. Brain cohorts (with strong AD signal and consistent biology for case/control) yielded much better separability than blood cohorts under the same architecture.

### 4.1 Practical Implications
1. The architecture is reusable across diseases.
2. Strong CV alone is insufficient; cohort-held-out testing is essential.
3. Future gains likely depend more on domain harmonization and training strategy than additional architectural complexity.

### 4.2 Limitations
1. External validation was limited to the included AD cohorts; no fully independent post-hoc dataset was used here.
2. Labels were binary AD vs Control; richer diagnostic strata were excluded.
3. LOCO instability across held-out cohorts indicates remaining domain dependence.
4. No prospective or clinical utility testing was performed.

## 5. Conclusion
The transferred multiplex hypergraph + MLP architecture generalizes to Alzheimer's brain transcriptomics with high and stable in-distribution performance (about 0.91-0.92 mean accuracy across seed-aggregated CV). Under strict out-of-cohort evaluation, performance declines to about 0.73 accuracy, indicating incomplete domain invariance. Overall, the methodology is generalizable but requires stronger domain-robust training and harmonization for reliable cohort-shift performance.

## Data and Code Availability
All generated artifacts are available locally:
- Main publication bundle: `C:\Users\terry\Downloads\Projects\ISEF\CH_DANN_Plan\results\publication_ready`
- Manifest: `C:\Users\terry\Downloads\Projects\ISEF\CH_DANN_Plan\results\publication_ready\publication_manifest.json`
- LOCO results:
  - `C:\Users\terry\Downloads\Projects\ISEF\CH_DANN_Plan\results\v11_alz_brain_loco_dann_results.json`
  - `C:\Users\terry\Downloads\Projects\ISEF\CH_DANN_Plan\results\v11_alz_brain_loco_nodann_results.json`
- Seed stability results:
  - `C:\Users\terry\Downloads\Projects\ISEF\CH_DANN_Plan\results\v11_alz_brain_seed_stability_dann_results.json`
  - `C:\Users\terry\Downloads\Projects\ISEF\CH_DANN_Plan\results\v11_alz_brain_seed_stability_nodann_results.json`

## Figure Captions
**Figure 1. LOCO cohort performance (accuracy and AUROC).**  
Bars compare DANN and no-DANN performance when each brain cohort is held out from training.

**Figure 2. Five-seed stability on brain transfer.**  
Box/strip distributions of mean per-seed CV accuracy and AUROC for DANN vs no-DANN settings.

**Figure 3. Model lineage performance on source neonatal task.**  
Progression from V7 to V11 highlights architectural improvements prior to AD transfer.

## Suggested Next Experiments
1. Add fold-internal harmonization and leakage-safe normalization variants for LOCO.
2. Evaluate class-balanced focal loss and cohort-balanced batching under LOCO.
3. Add a truly external AD cohort for post-development lockbox testing.

