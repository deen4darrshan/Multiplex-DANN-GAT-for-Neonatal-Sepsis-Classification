# Results and Interpretation

This file summarizes results at a high level. Full tables and figures are in `docs\publication`.

Key summary
- External holdout mean, GAT_v2: Acc=0.542, AUC=0.774, F1=0.565
- External holdout mean, LogisticRegression: Acc=0.375, AUC=0.503, F1=0.000
- Best human grouped 5-fold: L2 LR tuned (feat=diff, top_k=8000, C=100) with Acc=0.824, AUC=0.846, F1=0.857

Metrics table
| Evaluation | Model | Accuracy | AUC | F1 | Notes |
|---|---|---:|---:|---:|---|
| External holdout mean (human multicohort) | GAT_v2 | 0.542 | 0.774 | 0.565 | Leave one dataset out external test |
| External holdout mean (human multicohort) | LogisticRegression | 0.375 | 0.503 | 0.000 | Leave one dataset out external test |
| Human grouped 5-fold | LR (top_k=2000) | 0.735 | 0.780 | 0.769 | Human only, StratifiedGroupKFold |
| Human grouped 5-fold | GAT_v2 | 0.618 | 0.436 | 0.764 | Human only, StratifiedGroupKFold |
| Human grouped 5-fold | LR (tuned threshold) | 0.824 | 0.802 | 0.875 | Threshold tuned from OOF, top_k=12000, C=10.0, thr=0.19 |
| Human grouped 5-fold | L2 LR fixed (feat=diff, top_k=5000, C=50) | 0.794 | 0.835 | 0.829 | Fixed threshold 0.5 |
| Human grouped 5-fold | L2 LR tuned (feat=diff, top_k=8000, C=100) | 0.824 | 0.846 | 0.857 | Tuned threshold 0.18 |

Publication ready tables
- `C:\Users\terry\Downloads\Projects\ISEF\docs\publication\tables\metrics_summary.csv`
- `C:\Users\terry\Downloads\Projects\ISEF\docs\publication\tables\metrics_summary.md`