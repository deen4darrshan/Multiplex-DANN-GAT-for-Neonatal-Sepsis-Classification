# Slide 03 - Methods (Data and Preprocessing)
## Section
Methods

## Text to display
- Data sources (public GEO cohorts):
- Sepsis development cohorts (n=319 scored samples) plus independent external cohort GSE26440 (n=104).
- Alzheimer's disease brain cohorts (n=222).
- Osteogenesis imperfecta cohorts (total n=34 across 4 holdout cohorts).
- Controls and variables:
- Control group in each disease = healthy/non-disease samples.
- Predictor variables = transcriptomic gene expression features; outcome variable = disease class.
- Preprocessing workflow:
- Probe-to-gene harmonization, duplicate collapse, per-sample normalization, fold-local feature selection (top 2,000 MAD genes).
- Batch/domain correction and strict split-local processing to avoid leakage.

## Images to display
- `figures/pca_before_combat.png`
- `figures/pca_after_combat.png`

## Graphic credits (APA)
- Student Researcher. (2026). PCA structure before batch correction [Figure]. Generated from project preprocessing outputs.
- Student Researcher. (2026). PCA structure after batch correction [Figure]. Generated from project preprocessing outputs.
