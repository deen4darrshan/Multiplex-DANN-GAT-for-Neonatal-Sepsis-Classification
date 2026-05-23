# Final ACSEF Visual Set (12 Figures)

1. **Cross Disease Metric Scorecard**
   - File: `ACSEF_Final_Submission/final_visuals/01_cross_disease_metric_scorecard.png`
   - Source: `updated sepsis metrics + current disease summaries`
   - Why included: Master comparison view showing architecture and baseline performance for all diseases across accuracy, AUROC, and F1.
2. **All Model Landscape**
   - File: `ACSEF_Final_Submission/final_visuals/02_all_model_landscape.png`
   - Source: `General_Sepsis_V11/results/metrics_overall.csv + ACSEF compiled baselines + disease summaries`
   - Why included: Master graph comparing every available architecture and baseline point in one accuracy-AUROC-F1 space with readable model names.
3. **Architecture Gain Over Baseline**
   - File: `ACSEF_Final_Submission/final_visuals/03_architecture_gain_over_baseline.png`
   - Source: `derived from updated cross-disease table`
   - Why included: Hybrid model gain heatmap with explicit baseline architecture names by disease.
4. **Sepsis ROC Evidence Panel**
   - File: `ACSEF_Final_Submission/final_visuals/04_sepsis_roc_evidence_panel.png`
   - Source: `General_Sepsis_V11/results/plots/roc_cv_model_comparison.png + roc_external_model_comparison.png`
   - Why included: Side-by-side ROC evidence panel for CV and external settings.
5. **3D Graph Topology**
   - File: `ACSEF_Final_Submission/final_visuals/05_3d_graph_topology.png`
   - Source: `General_Sepsis_V11/results/plots/gnn_topology_3d.png`
   - Why included: Keeps the strongest existing topology visual for structural intuition.
6. **SHAP Summary Top 20**
   - File: `ACSEF_Final_Submission/final_visuals/06_shap_summary_top_20.png`
   - Source: `General_Sepsis_V11/results/plots/shap_summary_top20.png`
   - Why included: Keeps the most interpretable biomarker ranking visual in the package.
7. **Sepsis Cohort Policy Flow**
   - File: `ACSEF_Final_Submission/final_visuals/07_sepsis_cohort_policy_flow.png`
   - Source: `General_Sepsis_V11/results/cohort_manifest.json`
   - Why included: Updated workflow diagram with larger cohort cards and cleaner split communication.
8. **Biomarker Fingerprint Across Cohorts**
   - File: `ACSEF_Final_Submission/final_visuals/08_biomarker_fingerprint_across_cohorts.png`
   - Source: `General_Sepsis_V11/results/expression_combat.csv + metadata.csv + shap_top20_features.csv`
   - Why included: Shows whether the top SHAP genes behave consistently across datasets and conditions instead of only in aggregate.
9. **Graph Prior Coverage and Scale**
   - File: `ACSEF_Final_Submission/final_visuals/09_graph_prior_coverage_and_scale.png`
   - Source: `General_Sepsis_V11/results/pathway_info.json + cv_metrics_raw.json`
   - Why included: Quantifies how much biological structure is injected into the sepsis model and how large the runtime graph actually is.
10. **Sepsis Validation Dashboard**
   - File: `ACSEF_Final_Submission/final_visuals/10_sepsis_validation_dashboard.png`
   - Source: `General_Sepsis_V11/results/metrics_overall.csv + metrics_external.csv + general_sepsis_v11_results.json`
   - Why included: Expanded one-page sepsis panel with larger cards and a dedicated Best Model section.
11. **Rare Disease External Summary**
   - File: `ACSEF_Final_Submission/final_visuals/11_rare_disease_external_summary.png`
   - Source: `results/osteogenesis/osteogenesis_metrics_summary.json`
   - Why included: Replaces holdout matrix with architecture-included rare-disease summary panels.
12. **Sepsis Hybrid vs Baseline Panel**
   - File: `ACSEF_Final_Submission/final_visuals/12_sepsis_hybrid_vs_baseline_panel.png`
   - Source: `General_Sepsis_V11/results/metrics_overall.csv + ACSEF compiled baselines`
   - Why included: Focused sepsis comparison panel against Logistic Regression, GAT only, GCN only, and MLP only.

## Companion Prompt
- Nano Banana architecture prompt: `ACSEF_Final_Submission/final_visuals/nano_banana_master_prompt_hybrid_gcn_mlp_dann.md`
