# ACSEF Quad Chart Content

## Q1: Scientific Question (<=75 words)
Can a biologically informed, domain-robust graph model diagnose neonatal sepsis from blood transcriptomes more accurately than standard GCN/GAT/HGCN baselines while remaining interpretable? We tested whether multiplex biological priors (KEGG, STRING, co-expression) plus domain adversarial training improve generalization across cohorts and whether gene-level explainability identifies plausible biomarkers relevant to immune dysregulation.

## Q2: Methodology (<=75 words)
Pipeline: GEO data acquisition (GSE25504, GSE69686), preprocessing, gene alignment, ComBat correction, and graph construction from three relations. Model: Multiplex-Hypergraph-DANN-MLP (hypergraph conv -> relation attention -> gene scorer -> MLP classifier + domain head). Validation: stratified 5-fold CV and external validation on GSE26440. Architecture visual: `fig_architecture_flowchart.png`.

## Q3: Data Analysis and Results (<=75 words)
Main sepsis model achieved AUROC 0.9796, accuracy 0.9779, F1 0.9733 (5-fold). External cohort GSE26440 (n=104) achieved AUROC 0.9856, accuracy 0.9519, F1 0.9697. Baselines underperformed (HGCN AUROC 0.6842; GCN 0.6706; GAT 0.6819). XAI identified top biomarkers (TNFAIP6, S100A12, RETN, CD52). Visuals: `fig_roc_comparisons.png`, `fig_external_validation_gse26440.png`, `fig_biomarker_attributions.png`.

## Q4: Interpretation and Conclusions (<=75 words)
Biologically constrained multiplex graphs plus non-linear MLP integration materially improved robustness. Ablation without MLP collapsed toward chance, indicating linear graph aggregation alone was insufficient. Biomarker attributions aligned with inflammatory biology and supported interpretability. Transfer experiments in osteogenesis imperfecta preserved performance trends under grouped validation, demonstrating cross-disease scalability. Conclusion: domain-aware graph architectures can support clinically relevant and explainable rare-disease genomics.
