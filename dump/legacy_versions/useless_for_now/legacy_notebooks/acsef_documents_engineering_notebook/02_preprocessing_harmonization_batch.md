# 02 Preprocessing, Harmonization, and Batch Control

Date: 2026-02-24

## Gene-Space Harmonization
Expression matrices from different cohorts were aligned to shared gene symbols/IDs. Duplicate mappings were collapsed at the gene level to avoid multi-mapping leakage.

## Transformation and Filtering
- Applied normalization steps suitable for cross-study transcriptomics.
- Selected high-variance genes for model input dimensionality control (final explainability run used 2,000 genes).

## Batch Correction
ComBat-style adjustment was applied to reduce study-specific offsets while preserving biological class signal.

Conceptual model per gene:
- x_ij = alpha_j + beta_j * covariates_i + gamma_{b(i),j} + delta_{b(i),j} * epsilon_ij
- Remove estimated batch terms (gamma, delta) while retaining class-related structure in beta_j.

## Verification
Post-correction diagnostics used:
- Distribution overlays (`fig_normalization_distributions.png`).
- PCA-level separability checks (including top biomarker projections).

## Rationale
Without explicit batch correction and harmonized feature space, domain-adversarial learning alone can overfit to unresolved platform artifacts.
