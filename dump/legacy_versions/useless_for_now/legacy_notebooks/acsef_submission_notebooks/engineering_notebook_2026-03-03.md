# ACSEF Engineering Notebook (2026-03-03)

## Major Overhaul Summary
- Validation protocol migrated to robust cohort-aware mode (`lodo`) with fold-internal feature selection.
- Fold-specific normalization and graph construction were enforced to reduce optimistic validation artifacts.
- Comparative visualization stack was rebuilt to include model-by-model curves and heatmaps.
- 3D topology rendering and SHAP explainability were added to the sepsis package.
- Poster and publication manifest were regenerated from the new artifacts.

## Primary Artifacts
- `General_Sepsis_V11/results/overhaul_execution_log.md`
- `General_Sepsis_V11/results/general_sepsis_v11_results.json`
- `General_Sepsis_V11/results/metrics_report.md`
- `ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_poster.pdf`
- `ACSEF_Final_Submission/acsef_documents/engineering_notebook_master.tex`

## Notes
- Current robust results remove near-saturated CV behavior but do not show universal dominance over every baseline; claims should stay tied to specific metrics/splits in the generated traceability table.
