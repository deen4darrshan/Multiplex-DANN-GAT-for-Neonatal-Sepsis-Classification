# Training and Validation

Evaluation regimes
- Cross validation on combined cohorts for initial baselines.
- Leave one dataset out external validation for real world generalization.
- Human only StratifiedGroupKFold with GroupID to prevent family or subject leakage.

Grouped 5 fold design
- StratifiedGroupKFold is used with GroupID.
- For GSE160207, GroupID is the family identifier parsed from metadata.
- For other cohorts, GroupID is the sample or subject identifier.

Loss and class weighting
- Cross entropy loss with class weights balances positive and negative classes.
- Class weights are computed per training fold.

Early stopping
- Validation AUC is monitored every few epochs.
- Training stops when no improvement is observed for a patience window.

Thresholding
- Default threshold is 0.5 for most experiments.
- A tuned threshold is optionally chosen to maximize accuracy on out of fold predictions in the L2 LR tuning script.

Metrics
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- AUC = area under ROC curve
- F1 = 2 * (precision * recall) / (precision + recall)

Key scripts
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\04_baselines.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\05_train_gnn.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\08_run_real_external_eval.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\11_human_grouped5_eval.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\14_human_grouped5_l2_lr_tuning.py`
