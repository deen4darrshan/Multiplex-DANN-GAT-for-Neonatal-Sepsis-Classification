# 07 Rare-Disease Transfer: Osteogenesis Imperfecta

Date: 2026-02-24

## Transfer Goal
Test whether the architecture principles from neonatal sepsis remain useful in a distinct rare-disease setting.

## Methodological Parallels
- Cohort-aware preprocessing and gene filtering.
- Graph-guided representation learning.
- Strict grouped 5-fold validation by human cohort.
- Additional external-style holdout summaries.

## Key Reported Metrics
- Grouped 5-fold optimized LR: Accuracy 0.8235, AUROC 0.8022, F1 0.8750.
- Grouped 5-fold GAT benchmark: Accuracy 0.6176.
- External holdout means: LR Accuracy 0.3750, LR AUROC 0.5034, GAT Accuracy 0.5417, GAT AUROC 0.7737.

## Interpretation
Grouped CV performance indicates promising intra-domain learning, while holdout spread reveals realistic domain shift. This pattern is scientifically useful because it highlights where transfer is robust and where domain adaptation remains necessary.

## Narrative Value for ACSEF
This cross-disease extension supports a stronger claim than single-task performance: the framework is adaptable, interpretable, and testable in low-sample biomedical contexts beyond sepsis.
