# Metrics Summary

| Evaluation | Model | Accuracy | AUC | F1 | Notes |
|---|---|---:|---:|---:|---|
| Baseline CV (2 cohorts) | LogisticRegression | 1.000 | 1.000 | 1.000 | Combined CV on early cohorts, optimistic |
| Baseline CV (2 cohorts) | RandomForest | 1.000 | 1.000 | 1.000 | Combined CV on early cohorts, optimistic |
| Baseline CV (2 cohorts) | GAT_v2 | 1.000 | 1.000 | 1.000 | Combined CV on early cohorts, optimistic |
| Baseline CV (2 cohorts) | GCN | 1.000 | 1.000 | 1.000 | Combined CV on early cohorts, optimistic |
| External holdout mean (human multicohort) | GAT_v2 | 0.542 | 0.774 | 0.565 | Leave one dataset out external test |
| External holdout mean (human multicohort) | LogisticRegression | 0.375 | 0.503 | 0.000 | Leave one dataset out external test |
| Human grouped 5-fold | LR (top_k=2000) | 0.735 | 0.780 | 0.769 | Human only, StratifiedGroupKFold |
| Human grouped 5-fold | GAT_v2 | 0.618 | 0.436 | 0.764 | Human only, StratifiedGroupKFold |
| Human grouped 5-fold | LR (tuned threshold) | 0.824 | 0.802 | 0.875 | Threshold tuned from OOF, top_k=12000, C=10.0, thr=0.19 |
| Human grouped 5-fold | L2 LR fixed (feat=diff, top_k=5000, C=50) | 0.794 | 0.835 | 0.829 | Fixed threshold 0.5 |
| Human grouped 5-fold | L2 LR tuned (feat=diff, top_k=8000, C=100) | 0.824 | 0.846 | 0.857 | Tuned threshold 0.18 |