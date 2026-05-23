# Final Package QA Checklist

Generated: 2026-03-02T19:55:26.645593

Artifacts checked: 17
Missing artifacts: 0

## Artifact Presence
- PASS c:/Users/terry/Downloads/Projects/ISEF/General_Sepsis_V11/results/expression_combat.csv
- PASS c:/Users/terry/Downloads/Projects/ISEF/General_Sepsis_V11/results/metadata.csv
- PASS c:/Users/terry/Downloads/Projects/ISEF/General_Sepsis_V11/results/gene_list.json
- PASS c:/Users/terry/Downloads/Projects/ISEF/General_Sepsis_V11/results/cohort_manifest.json
- PASS c:/Users/terry/Downloads/Projects/ISEF/General_Sepsis_V11/results/pathway_info.json
- PASS c:/Users/terry/Downloads/Projects/ISEF/General_Sepsis_V11/results/cv_metrics_raw.json
- PASS c:/Users/terry/Downloads/Projects/ISEF/General_Sepsis_V11/results/general_sepsis_v11_results.json
- PASS c:/Users/terry/Downloads/Projects/ISEF/General_Sepsis_V11/results/baseline_comparison.json
- PASS c:/Users/terry/Downloads/Projects/ISEF/General_Sepsis_V11/results/validation_audit_report.md
- PASS c:/Users/terry/Downloads/Projects/ISEF/General_Sepsis_V11/models/general_sepsis_v11_best.pt
- PASS c:/Users/terry/Downloads/Projects/ISEF/ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_poster.pdf
- PASS c:/Users/terry/Downloads/Projects/ISEF/ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_poster.svg
- PASS c:/Users/terry/Downloads/Projects/ISEF/ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_poster.png
- PASS c:/Users/terry/Downloads/Projects/ISEF/ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_figure_manifest.md
- PASS c:/Users/terry/Downloads/Projects/ISEF/ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_claim_traceability.csv
- PASS c:/Users/terry/Downloads/Projects/ISEF/ACSEF_Final_Submission/acsef_documents/engineering_notebook_master.tex
- PASS c:/Users/terry/Downloads/Projects/ISEF/ACSEF_Final_Submission/acsef_documents/engineering_notebook_master.pdf

## Claim-to-Artifact Mapping
- Poster claims traceability: ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_claim_traceability.csv
- Figure source manifest: ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_figure_manifest.md
- Model metrics source: General_Sepsis_V11/results/general_sepsis_v11_results.json

## Gate Snapshot (non-artifact quality gates)
- leakage_checks_pass: True
- cv_mean_auroc_ge_0_75: True
- external_auroc_ge_0_70: True
- external_auroc_ci_lower_gt_0_60: True
- model_auc_improvement_ge_0_05: False
- permutation_p_lt_0_05: False
- all_passed: False

## Critical Failures: 0
- Zero critical package-assembly failures (all required deliverable files present).