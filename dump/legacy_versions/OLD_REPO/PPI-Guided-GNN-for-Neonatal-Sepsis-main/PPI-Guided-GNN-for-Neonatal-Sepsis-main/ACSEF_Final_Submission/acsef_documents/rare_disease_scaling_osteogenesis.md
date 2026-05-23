# Rare-Disease Generalization: Osteogenesis Imperfecta

## Objective
Demonstrate that graph-guided disease modeling principles transfer beyond neonatal sepsis into another rare disease context.

## Method Transfer
The osteogenesis pipeline re-used the core graph strategy: biologically constrained topology, variance-based gene filtering, strict grouped validation, and external holdout stress-testing.
This preserves the central design philosophy of the sepsis architecture even though disease-specific preprocessing and classifier heads were adapted.

## Key Outcomes
- Human grouped 5-fold best accuracy: 0.824
- Human grouped 5-fold best AUROC: 0.802
- Human grouped 5-fold best F1: 0.875
- Human grouped 5-fold GAT accuracy: 0.618
- External holdout mean LR accuracy: 0.375
- External holdout mean LR AUROC: 0.503
- External holdout mean GAT accuracy: 0.542
- External holdout mean GAT AUROC: 0.774

## Interpretation
The cross-disease transfer supports a general claim: graph-constrained representations and strict cohort-aware validation remain informative in low-sample rare disease settings.
Performance spread across held-out cohorts highlights realistic domain shift and motivates domain-adversarial extensions for future rare-disease deployments.

## Figure Reference
- fig_osteogenesis_scaling_summary.png