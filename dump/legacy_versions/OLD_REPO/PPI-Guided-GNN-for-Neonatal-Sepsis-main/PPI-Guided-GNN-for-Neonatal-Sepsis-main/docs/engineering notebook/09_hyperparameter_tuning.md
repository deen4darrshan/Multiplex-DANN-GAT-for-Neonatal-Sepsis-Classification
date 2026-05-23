# Hyperparameter Tuning

Tabular models
- Feature selection Top K was searched in [200, 500, 1000, 1500, 2000] for grouped 5 fold.
- Models evaluated: LogisticRegression, RandomForest, and RBF SVM.

L2 Logistic Regression tuning
- Feature ranking strategies: MAD and absolute mean difference between OI and control.
- Top K search: [2000, 5000, 8000, 12000, 15000].
- Regularization strength C search: [0.1, 0.3, 1, 3, 5, 8, 10, 15, 20, 30, 50, 100].
- Both fixed threshold 0.5 and tuned threshold were reported.

GAT tuning
- Hidden size, heads, dropout, and early stopping were varied in coarse searches.
- The grouped 5 fold GAT used the best Top K from the tabular sweep.

Key scripts
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\10_tune_5fold_combined.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\11_human_grouped5_eval.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\14_human_grouped5_l2_lr_tuning.py`
