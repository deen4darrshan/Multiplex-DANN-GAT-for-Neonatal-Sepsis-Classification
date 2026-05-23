# ACSEF Project Presentation (12 Slides Max)

Font guidance: Arial or Times New Roman, minimum 24 pt body text, high-contrast colors.

## Slide 1: Title
Title: Graph-Guided Domain-Adversarial Learning for Neonatal Sepsis Biomarker Discovery
Subtitle: ACSEF Final Project
Presenter: Student Researcher
Date: 2026-02-24

## Slide 2: Clinical Problem and Hypothesis
- Neonatal sepsis has high morbidity and delayed diagnosis.
- Transcriptomic models often fail across cohorts due to domain shift.
- Hypothesis: Multiplex biological priors + domain-adversarial training improves accuracy and transferability.

## Slide 3: Datasets and Study Design
- Development cohorts: GSE25504 and GSE69686.
- External validation cohort: GSE26440 (n=104).
- Features harmonized to common gene space; top variance genes retained.
- Validation protocol: 5-fold cross-validation + external holdout.
Citation (APA): Barrett, T., et al. (2013). NCBI GEO: archive for functional genomics data sets. Nucleic Acids Research, 41(D1), D991-D995.

## Slide 4: Preprocessing and Quality Control
- Gene ID harmonization and duplicate collapse.
- Log/scale normalization and missingness checks.
- Batch correction with ComBat across studies.
- Distribution and PCA checks after correction.
Graphic: `fig_normalization_distributions.png`
Graphic citation (APA): Student Researcher. (2026, February 24). Normalization distribution diagnostics [Figure].

## Slide 5: Final Architecture
- Multiplex hypergraph convolution on KEGG, STRING, and co-expression relations.
- Relation attention fuses relation-specific embeddings.
- Gene scorer produces biologically constrained feature mask.
- MLP classifier predicts sepsis; adversarial head penalizes domain leakage.
Graphic: `fig_architecture_flowchart.png`
Graphic citation (APA): Student Researcher. (2026, February 24). Multiplex-Hypergraph-DANN-MLP architecture [Figure].

## Slide 6: Training Strategy and Mathematical Objective
- Objective: L_total = L_class + alpha * L_domain + beta * L_reg.
- Class loss: binary cross-entropy.
- Domain loss: adversarial binary cross-entropy on batch/domain labels.
- Regularization: weight decay and dropout to reduce overfitting.
- Early stopping on validation performance.

## Slide 7: Baseline Comparison Results
- Baselines: HGCN, GCN, GAT.
- Final model exceeded all baselines by large margin in AUROC and accuracy.
- Key result: final AUROC 0.9796 vs baseline AUROC approximately 0.67 to 0.68.
Graphic: `fig_roc_comparisons.png`
Graphic citation (APA): Student Researcher. (2026, February 24). AUROC comparison across baseline and final models [Figure].

## Slide 8: External Validation (Generalization)
- Evaluated locked model on GSE26440 external cohort.
- Accuracy 0.9519, AUROC 0.9856, F1 0.9697.
- Supports true cross-cohort generalization, not only in-sample fitting.
Graphic: `fig_external_validation_gse26440.png`
Graphic citation (APA): Student Researcher. (2026, February 24). External validation performance on GSE26440 [Figure].

## Slide 9: Explainable AI and Biomarkers
- Implemented custom Integrated Gradients to avoid environment deadlocks.
- Ranked 100 biomarkers using signed attribution and gene-score fusion.
- Top signals included TNFAIP6, S100A12, RETN, and CD52.
Graphics: `fig_biomarker_attributions.png`, `fig_biomarker_correlation_heatmap.png`
Graphic citation (APA): Student Researcher. (2026, February 24). Top biomarker integrated gradient attributions [Figure].
Graphic citation (APA): Student Researcher. (2026, February 24). Biomarker correlation structure [Figure].

## Slide 10: Engineering Failures and Recovery
- SciPy statistical helpers triggered instability under high-memory Windows runs.
- Loading full STRING table caused swap-thrashing.
- Unicode console encoding crashed one pipeline execution.
- Solutions: pure NumPy ranking/MAD, chunked STRING scan, UTF-8 execution context.
Reference file: `logs/failure_analysis_log.md`

## Slide 11: Rare-Disease Scaling (Osteogenesis Imperfecta)
- Applied core graph-guided methodology to osteogenesis imperfecta datasets.
- Human grouped 5-fold best accuracy 0.8235, AUROC 0.8022.
- Demonstrated architecture portability with realistic domain-shift behavior.
Graphic: `fig_osteogenesis_scaling_summary.png`
Graphic citation (APA): Student Researcher. (2026, February 24). Rare-disease scaling summary [Figure].

## Slide 12: Conclusions and References
Conclusions:
- Multiplex-Hypergraph-DANN-MLP improved sepsis prediction and external generalization.
- MLP integration was essential; pure graph ablation collapsed toward chance.
- XAI biomarkers provided biologically interpretable signals.
- Framework generalized to a second rare disease setting.

References (APA):
- Barrett, T., et al. (2013). NCBI GEO: archive for functional genomics data sets. Nucleic Acids Research, 41(D1), D991-D995.
- Leek, J. T., Johnson, W. E., Parker, H. S., Jaffe, A. E., & Storey, J. D. (2012). The sva package for removing batch effects and other unwanted variation in high-throughput experiments. Bioinformatics, 28(6), 882-883.
- Szklarczyk, D., et al. (2023). STRING v12: protein-protein association networks. Nucleic Acids Research, 51(D1), D638-D646.
- Student Researcher. (2026). All project-specific figures and analysis outputs generated in ACSEF_Final_Submission.
- OpenAI. (2026). AI-assisted coding and documentation support acknowledged for engineering acceleration; all scientific interpretation and final validation decisions were performed by the student.
