# General_Sepsis_V11 Robust Metrics and Visualization Report

- CV mode: lodo
- Hybrid decision threshold (OOF): 0.0109

## Overall Metrics (CV OOF)
| split   | model               |   n |   accuracy |       f1 |   precision |   recall |    auroc |
|:--------|:--------------------|----:|-----------:|---------:|------------:|---------:|---------:|
| cv_oof  | hybrid_v11          | 345 |   0.878261 | 0.903226 |    0.911628 | 0.894977 | 0.847739 |
| cv_oof  | logistic_regression | 345 |   0.773913 | 0.795812 |    0.932515 | 0.694064 | 0.797528 |
| cv_oof  | mlp_only            | 345 |   0.75942  | 0.779841 |    0.93038  | 0.671233 | 0.801479 |
| cv_oof  | v12_no_mlp_ablation | 345 |   0.768116 | 0.786096 |    0.948387 | 0.671233 | 0.791984 |

## External Holdout Metrics
| split            | model                     |   n |   accuracy |       f1 |   precision |   recall |    auroc |
|:-----------------|:--------------------------|----:|-----------:|---------:|------------:|---------:|---------:|
| external_holdout | hybrid_v11                | 103 |   0.796117 | 0.886486 |    0.796117 |        1 | 0.991289 |
| external_holdout | logistic_regression_refit | 103 |   0.796117 | 0.886486 |    0.796117 |        1 | 0.880952 |
| external_holdout | mlp_only_refit            | 103 |   0.796117 | 0.886486 |    0.796117 |        1 | 0.5      |
| external_holdout | v12_no_mlp_ablation_refit | 103 |   0.796117 | 0.886486 |    0.796117 |        1 | 0.781069 |

## SHAP Reference Model
- External accuracy: 0.7961
- External AUROC: 0.8810
- Threshold: 0.5853

## Generated Plots
- `results/plots/roc_cv_model_comparison.png`
- `results/plots/pr_cv_model_comparison.png`
- `results/plots/roc_external_model_comparison.png`
- `results/plots/metrics_heatmap_cv.png`
- `results/plots/metrics_heatmap_external.png`
- `results/plots/auroc_heatmap_by_dataset_cv.png`
- `results/plots/relation_attention_heatmap.png`
- `results/plots/gnn_topology_3d.png`
- `results/plots/shap_summary_top20.png`
- `results/plots/shap_heatmap_top20.png`