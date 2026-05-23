# GAT vs Hybrid Test

- Rule: GAT must exceed hybrid AUROC and at least match hybrid accuracy.
- Robust (General_Sepsis_V11) comparison:
  - Hybrid AUROC: 0.8477
  - GAT AUROC: 0.6819
  - Hybrid Accuracy: 0.8783
  - GAT Accuracy: 0.4169
  - GAT beats hybrid: False
- Compiled benchmark comparison:
  - Hybrid AUROC: 0.9796
  - GAT AUROC: 0.6819
  - Hybrid Accuracy: 0.9779
  - GAT Accuracy: 0.4169
  - GAT beats hybrid: False

- Final decision: `keep_hybrid_visuals`
