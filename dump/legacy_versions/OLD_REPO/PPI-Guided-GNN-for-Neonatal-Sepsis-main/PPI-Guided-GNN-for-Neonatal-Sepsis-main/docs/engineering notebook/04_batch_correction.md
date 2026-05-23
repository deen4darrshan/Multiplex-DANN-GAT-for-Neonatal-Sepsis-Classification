# Batch Correction

Method
ComBat was used to remove batch effects while preserving the Condition signal.

Model intuition
ComBat assumes that gene expression can be decomposed into a biological component, a batch effect, and noise, and it estimates and subtracts batch specific adjustments.

Implementation
- Input matrix is genes by samples after log2 transform.
- Batch label is the dataset identifier.
- Condition is passed as a covariate to preserve signal.

Verification
- PCA plots are generated before and after ComBat.
- Visual inspection ensures batch separation is reduced while biological separation is not erased.

Key script
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\02_combat_correction.py`

Artifacts
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\data\processed\expression_combat.csv`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\data\processed\metadata_combat.csv`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\figures\pca_before_combat.png`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\figures\pca_after_combat.png`
