# Project Origin

Goal
Build a reproducible, real data based classifier for Osteogenesis Imperfecta (OI) using gene expression and a graph neural network (GNN) architecture that incorporates protein interaction structure.

Why this direction
- OI is a rare disease with small, multi cohort RNA sequencing datasets that are vulnerable to batch effects and overfitting.
- A graph based model allows expression values to be contextualized by protein interaction topology, which is a biologically grounded inductive bias.
- A strict evaluation regime was required to avoid inflated accuracy from synthetic or overly curated data.

How it started
- Initial work used two public human RNA seq cohorts, GSE160207 and GSE163812, with a basic baseline classifier.
- Batch effects were addressed via ComBat and a standard log2 transform.
- A graph was built from STRING interactions to enable GNN training.

How it expanded
- Additional human cohorts were added for real world validation and to reduce cohort specific overfitting: GSE180838 and GSE186141.
- A separate expanded dataset with mouse OI (GSE154748) was constructed for sensitivity testing, while the primary human only evaluation remained the main benchmark.
- A leave one dataset out external evaluation and a human only StratifiedGroupKFold protocol were implemented.

Key files
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\00_download_data.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\01_prepare_expression.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\02_combat_correction.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\03_build_graphs.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\05_train_gnn.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\07_prepare_multicohort_real.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\08_run_real_external_eval.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\11_human_grouped5_eval.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\14_human_grouped5_l2_lr_tuning.py`
