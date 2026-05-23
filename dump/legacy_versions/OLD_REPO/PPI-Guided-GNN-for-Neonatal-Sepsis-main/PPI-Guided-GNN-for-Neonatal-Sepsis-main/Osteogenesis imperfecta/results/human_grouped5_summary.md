# Human-only Grouped 5-Fold Results

- Validation data: human only
- CV: StratifiedGroupKFold, 5 folds, grouping by GroupID
- Total samples: 34

## Best
- Best tabular: LR (top_k=2000) -> Acc=0.735, AUC=0.780, F1=0.769
- GAT: Acc=0.618, AUC=0.436, F1=0.764