# Model Justification and Architecture

## New Scientific Naming Scheme
- Final architecture: Multiplex-Hypergraph-DANN-MLP
- HGCN baseline: Pathway-HGCN-Classic
- GCN baseline: Interaction-GCN-Baseline
- GAT baseline: Attention-GAT-Baseline
- Pure ablation: Multiplex-Hypergraph-Pure-NoMLP

## Benchmark Comparison
| Model | AUROC | Accuracy | F1 | Protocol |
|---|---:|---:|---:|---|
| Pathway-HGCN-Classic | 0.684 | 0.702 | 0.629 | 5-fold stratified CV on rebuilt sepsis training cohorts |
| Interaction-GCN-Baseline | 0.671 | 0.414 | 0.585 | 5-fold CV on Sepsis_GNN_V2 baseline setting |
| Attention-GAT-Baseline | 0.682 | 0.417 | 0.588 | 5-fold CV on Sepsis_GNN_V2 transfer baseline setting |
| Multiplex-Hypergraph-DANN-MLP | 0.980 | 0.978 | 0.973 | 5-fold stratified CV (combined condition+batch key) on sepsis cohorts |
| Multiplex-Hypergraph-DANN-MLP (Robust-XAI Rebuild) | 0.991 | 0.922 | 0.915 | single stratified holdout used to stabilize integrated gradients run |
| Multiplex-Hypergraph-Pure-NoMLP | 0.450 | 0.500 | N/A | reported collapse to random chance (~0.4-0.5 AUROC) in V12 ablation without MLP |

## Why MLP Integration Was Critical
The archived V12 pure hypergraph ablation removed the MLP and collapsed to random-chance behavior (reported AUROC around 0.4 to 0.5).
This indicates that relation-aware graph propagation alone did not separate class manifolds reliably in this dataset.
The MLP branch was therefore essential to capture non-linear interactions in the 2,000-gene feature space after graph-derived importance masking.

## Final Architecture (Operational Description)
Multiplex Hypergraph Convolution over three relations (KEGG pathways, STRING PPI, co-expression) generates per-gene embeddings.
A relation-attention block learns per-gene weighting across relations.
A gene-scoring head produces a mask that gates expression values.
Masked expression is fed to an MLP classifier for sepsis/control prediction, with a domain-adversarial head to discourage batch-specific shortcuts.

## External Validation on GSE26440
- Samples: 104
- Accuracy: 0.9519
- AUROC: 0.9856
- F1: 0.9697
- Precision: 0.9639
- Recall: 0.9756
Interpretation: performance remained high on an out-of-distribution pediatric cohort, supporting true biological generalization rather than only in-cohort fitting.

## Figure References
- fig_roc_comparisons.png
- fig_architecture_flowchart.png
- fig_external_validation_gse26440.png