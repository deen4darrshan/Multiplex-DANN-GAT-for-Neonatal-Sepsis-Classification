# 06 Baseline Comparison and Architectural Reasoning

Date: 2026-02-24

## Benchmarked Models
- Pathway-HGCN-Classic
- Interaction-GCN-Baseline
- Attention-GAT-Baseline
- Multiplex-Hypergraph-DANN-MLP (final)
- Multiplex-Hypergraph-Pure-NoMLP (ablation, archived report)

## Quantitative Summary
- HGCN: AUROC 0.6842, Accuracy 0.7020, F1 0.6291
- GCN: AUROC 0.6706, Accuracy 0.4138, F1 0.5853
- GAT: AUROC 0.6819, Accuracy 0.4169, F1 0.5885
- Final: AUROC 0.9796, Accuracy 0.9779, F1 0.9733
- Pure-NoMLP ablation: reported AUROC approximately 0.4 to 0.5

## Reasoning Behind Final Design
1. Single-relation or simple graph baselines underfit complex cross-cohort structure.
2. Multiplex relations improved biological context coverage.
3. Domain-adversarial head reduced cohort-specific shortcuts.
4. MLP classifier was necessary to model non-linear interactions after graph-guided masking.

## External Validity Anchor
External GSE26440 performance (AUROC 0.9856) confirms that gains were not limited to internal fold behavior.
