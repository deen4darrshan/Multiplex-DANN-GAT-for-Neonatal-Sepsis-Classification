# Data Sources

Primary human cohorts
- GSE160207: human fibroblast RNA seq counts for OI and control samples. Counts file plus series matrix for metadata.
- GSE163812: human donor fibroblast RNA seq counts with treatments. Only baseline GFP treated samples were retained to avoid intervention confounding.
- GSE180838: human FKBP10 related dataset, expression provided in an xlsx sheet with sample information.
- GSE186141: human dataset with OI versus control labels in an xlsx sheet with series matrix metadata.

Expanded cohort
- GSE154748: mouse OI dataset used only for expanded sensitivity testing. Human only evaluation remains the main benchmark.

Biological network
- STRING v12 protein protein interaction file for Homo sapiens, filtered by a confidence threshold.
- MyGene API for mapping Ensembl protein IDs to gene symbols.

Where they are stored
- Raw downloads: `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\data\raw`
- Per cohort processed files: `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\data\processed\datasets`
- Combined datasets: `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\data\processed`

Key scripts
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\00_download_data.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\07_prepare_multicohort_real.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\09_prepare_expanded_real.py`
