# 04 Training Protocol and Validation Logic

Date: 2026-02-24

## Sepsis Training/Evaluation
- Primary final model evaluation: 5-fold stratified CV in archived V11 run.
- Stratification key: combined condition and batch constraints to preserve class balance and reduce leakage.
- Additional robust-XAI run: stratified holdout used to stabilize long Integrated Gradients execution.

## Osteogenesis Validation
- Explicit grouped 5-fold validation by human cohort grouping to enforce realistic generalization boundaries.
- Separate holdout analyses for external cohort behavior.

## Metric Definitions
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 * (Precision * Recall) / (Precision + Recall)
- AUROC = area under ROC curve over decision thresholds.

## Recorded Outcomes
- Sepsis final (archived V11): AUROC 0.9796, Accuracy 0.9779, F1 0.9733.
- External GSE26440: AUROC 0.9856, Accuracy 0.9519, F1 0.9697.
- OI grouped 5-fold best: Accuracy 0.8235, AUROC 0.8022.

## Interpretation Constraint
Cross-model comparisons include protocol caveats because older baselines were run in historical settings with partially different data splits and graph variants.
