# Overhaul Execution Log (2026-03-03)

## Scope
- Objective: remove optimistic validation artifacts, run robust cohort-aware evaluation, regenerate visuals (model-by-model comparisons, heatmaps, 3D topology), add SHAP explainability, and rebuild publication assets.
- Project root: `C:\Users\terry\Downloads\Projects\ISEF`

## Detailed Steps
1. Reviewed orchestration instructions and initialized swarm state at `.swarm/overhaul_state.json`.
2. Audited current scripts and outputs:
   - `General_Sepsis_V11/scripts/01_download_and_preprocess.py`
   - `General_Sepsis_V11/scripts/03_train_v11_general_sepsis.py`
   - `General_Sepsis_V11/scripts/04_evaluate.py`
   - `General_Sepsis_V11/scripts/06_metrics_and_plots.py`
   - `ACSEF_Final_Submission/scripts/05_generate_general_sepsis_v11_publication.py`
3. Identified validation weaknesses:
   - feature selection performed globally before fold-level validation,
   - sample-level validation overestimated cohort transfer,
   - weak comparative plotting and no SHAP/3D topology visualization.
4. Installed SHAP runtime dependency (`python -m pip install shap`).
5. Patched preprocessing script:
   - added export of raw selected matrix (`expression_raw_selected.csv`) for fold-safe training/evaluation.
6. Patched model training script:
   - added `--cv-mode {lodo,sgkf}` (default `lodo`),
   - added fold-internal MAD feature selection (`--feature-select-top-k`),
   - added fold-specific relation filtering (KEGG/STRING) and fold-specific co-expression construction,
   - added fold-only normalization (`fold_norm_mean/std`) and checkpoint persistence,
   - added fold metadata persistence (`selected_gene_indices`, `selected_genes`, `val_dataset`).
7. Patched evaluation script:
   - added threshold-aware metric calculation,
   - added OOF-derived threshold calibration,
   - added fold-selected feature support in baselines,
   - added fold-aware normalization for external inference,
   - extended leakage checks with dataset-disjoint checks under LODO.
8. Rebuilt plotting script (`06_metrics_and_plots.py`) to generate:
   - CV ROC/PR model comparison overlays,
   - external ROC model comparison,
   - CV and external metric heatmaps,
   - AUROC-by-dataset heatmap,
   - relation attention heatmap,
   - 3D GNN topology visualization,
   - SHAP summary + SHAP heatmap + SHAP top-feature CSV.
9. Rebuilt publication script (`ACSEF_Final_Submission/scripts/05_generate_general_sepsis_v11_publication.py`):
   - syncs robust plots into ACSEF figure/image folders,
   - generates poster PNG/SVG/PDF,
   - regenerates figure manifest and claim traceability CSV.
10. Updated notebook builder (`05_build_master_notebook.py`) to include this execution trace.
11. Updated dependencies (`requirements.txt`) with `shap>=0.45.0`.

## Commands Executed (Major)
- `python General_Sepsis_V11/scripts/01_download_and_preprocess.py`
- `python General_Sepsis_V11/scripts/02_build_graphs.py`
- `python General_Sepsis_V11/scripts/03_train_v11_general_sepsis.py --expression-path General_Sepsis_V11/results/expression_raw_selected.csv --epochs 60 --patience 12 --feature-select-top-k 1000 --lambda-dann 0.0 --dropout 0.3`
- `python General_Sepsis_V11/scripts/04_evaluate.py --expression-path General_Sepsis_V11/results/expression_raw_selected.csv`
- `python General_Sepsis_V11/scripts/03_train_v11_general_sepsis.py --expression-path General_Sepsis_V11/results/expression_raw_selected.csv --epochs 60 --patience 12 --feature-select-top-k 1200 --lambda-dann 0.0 --dropout 0.2`
- `python General_Sepsis_V11/scripts/04_evaluate.py --expression-path General_Sepsis_V11/results/expression_raw_selected.csv`
- `python General_Sepsis_V11/scripts/06_metrics_and_plots.py --expression-path General_Sepsis_V11/results/expression_raw_selected.csv`
- `python ACSEF_Final_Submission/scripts/05_generate_general_sepsis_v11_publication.py`

## Outputs Regenerated
- Core metrics:
  - `General_Sepsis_V11/results/cv_metrics_raw.json`
  - `General_Sepsis_V11/results/general_sepsis_v11_results.json`
  - `General_Sepsis_V11/results/baseline_comparison.json`
  - `General_Sepsis_V11/results/validation_audit_report.md`
  - `General_Sepsis_V11/results/metrics_overall.csv`
  - `General_Sepsis_V11/results/metrics_external.csv`
- Visual package:
  - `General_Sepsis_V11/results/plots/roc_cv_model_comparison.png`
  - `General_Sepsis_V11/results/plots/pr_cv_model_comparison.png`
  - `General_Sepsis_V11/results/plots/roc_external_model_comparison.png`
  - `General_Sepsis_V11/results/plots/metrics_heatmap_cv.png`
  - `General_Sepsis_V11/results/plots/metrics_heatmap_external.png`
  - `General_Sepsis_V11/results/plots/auroc_heatmap_by_dataset_cv.png`
  - `General_Sepsis_V11/results/plots/relation_attention_heatmap.png`
  - `General_Sepsis_V11/results/plots/gnn_topology_3d.png`
  - `General_Sepsis_V11/results/plots/shap_summary_top20.png`
  - `General_Sepsis_V11/results/plots/shap_heatmap_top20.png`
  - `General_Sepsis_V11/results/plots/shap_top20_features.csv`
- Publication assets:
  - `ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_poster.png`
  - `ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_poster.svg`
  - `ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_poster.pdf`
  - `ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_figure_manifest.md`
  - `ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_claim_traceability.csv`

## Verification Notes
- Syntax checks executed with `python -m py_compile` for each modified script.
- Leakage checks in `general_sepsis_v11_results.json` report pass under robust fold constraints.
- Model and baseline comparisons now show non-trivial, non-saturated performance under robust protocol.

## Changed Files
- `General_Sepsis_V11/scripts/01_download_and_preprocess.py`
- `General_Sepsis_V11/scripts/03_train_v11_general_sepsis.py`
- `General_Sepsis_V11/scripts/04_evaluate.py`
- `General_Sepsis_V11/scripts/05_build_master_notebook.py`
- `General_Sepsis_V11/scripts/06_metrics_and_plots.py`
- `ACSEF_Final_Submission/scripts/05_generate_general_sepsis_v11_publication.py`
- `requirements.txt`
