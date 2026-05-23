# GAT Internal Swap vs Current Hybrid

- Rule: GAT-internal must exceed hybrid AUROC and at least match hybrid accuracy.
- Current hybrid pooled OOF:
  - Accuracy: 0.6812
  - AUROC: 0.8477
  - F1: 0.6978
- GAT-internal swap pooled OOF:
  - Accuracy: 0.9884
  - AUROC: 1.0000
  - F1: 0.9908
- GAT-internal beats current hybrid: True
- Final decision: `replace_with_gat_internal`
