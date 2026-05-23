# 08 Reproducibility, File Map, and Execution Order

Date: 2026-02-24

## Execution Order Used for Final Submission
1. Rebuild and train sepsis backbone artifacts (`10_rebuild_and_train_a1.py`).
2. Run robust XAI biomarker pipeline (`ACSEF_Final_Submission/scripts/01_run_robust_xai_biomarkers.py`).
3. Aggregate metrics and generate required figures (`ACSEF_Final_Submission/scripts/02_compile_metrics_and_make_figures.py`).
4. Build publication assets and interactive visuals (`ACSEF_Final_Submission/scripts/03_build_publication_assets.py`).

## Core Output Locations
- Data: `ACSEF_Final_Submission/data`
- Models: `ACSEF_Final_Submission/models`
- Figures: `ACSEF_Final_Submission/figures`
- Results: `ACSEF_Final_Submission/results`
- Documents: `ACSEF_Final_Submission/acsef_documents`
- Logs: `ACSEF_Final_Submission/logs`

## Determinism Notes
- Seeds were fixed where scripts exposed seed control.
- Small run-to-run floating-point variation can occur on GPU execution.
- Reported metrics are grounded in generated JSON outputs tracked in the submission folder.

## Submission Caveat Notes
- Baseline sources were historical outputs with non-identical protocols.
- The robust-XAI run is a faithful architectural rebuild, not a byte-identical replay of every archived checkpoint state.
