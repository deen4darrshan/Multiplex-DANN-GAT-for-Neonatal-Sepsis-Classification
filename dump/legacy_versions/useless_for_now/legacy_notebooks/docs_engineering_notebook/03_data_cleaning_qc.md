# Data Cleaning and QC

Gene symbol normalization
- All gene symbols are uppercased and stripped of extra delimiters.
- Duplicate symbols are aggregated by mean.

Numeric coercion and missing values
- Expression values are coerced to numeric with errors set to NaN.
- NaN values are imputed using per gene median or mean depending on the dataset script.
- Genes with near zero variance are dropped to avoid numerical artifacts.

Transformations
- Log transform uses log2(x + 1) for count or count like matrices.
- Standardization occurs per sample in model specific pipelines to stabilize model training.

Label handling
- Samples with Unknown label are removed before training.
- The human only evaluation uses only human datasets and excludes the mouse cohort.

Quality control visuals
- PCA plots before and after ComBat are generated to verify batch correction.

Key scripts
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\01_prepare_expression.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\02_combat_correction.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\07_prepare_multicohort_real.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\09_prepare_expanded_real.py`
