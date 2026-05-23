# General_Sepsis_V11 Robust Metrics and Visualization Report

- CV mode: lodo
- Hybrid decision threshold (OOF): 0.0109

## Overall Metrics (CV OOF)
| split   | model               |   n |   accuracy |       f1 |   precision |   recall |    auroc |
|:--------|:--------------------|----:|-----------:|---------:|------------:|---------:|---------:|
| cv_oof  | hybrid_v11          | 345 |   0.878261 | 0.903226 |    0.911628 | 0.894977 | 0.847739 |
| cv_oof  | logistic_regression | 345 |   0.944928 | 0.958242 |    0.923729 | 0.995434 | 0.868595 |
| cv_oof  | mlp_only            | 345 |   0.933333 | 0.949227 |    0.918803 | 0.981735 | 0.873505 |
| cv_oof  | v12_no_mlp_ablation | 345 |   0.852174 | 0.879433 |    0.911765 | 0.849315 | 0.901174 |

## External Holdout Metrics
| split            | model                     |   n |   accuracy |       f1 |   precision |   recall |    auroc |
|:-----------------|:--------------------------|----:|-----------:|---------:|------------:|---------:|---------:|
| external_holdout | hybrid_v11                | 103 |   0.796117 | 0.886486 |    0.796117 | 1        | 0.990708 |
| external_holdout | logistic_regression_refit | 103 |   0.815534 | 0.868966 |    1        | 0.768293 | 0.994193 |
| external_holdout | mlp_only_refit            | 103 |   0.796117 | 0.886486 |    0.796117 | 1        | 0.5      |
| external_holdout | v12_no_mlp_ablation_refit | 103 |   0.796117 | 0.886486 |    0.796117 | 1        | 0.5      |

## SHAP Reference Model
- External accuracy: 0.8155
- External AUROC: 0.9942
- Threshold: 0.9441

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