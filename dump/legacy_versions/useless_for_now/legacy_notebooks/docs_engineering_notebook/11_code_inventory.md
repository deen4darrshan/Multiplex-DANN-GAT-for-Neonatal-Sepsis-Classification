# Code Inventory

Scripts in the Osteogenesis imperfecta pipeline
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\00_download_data.py` downloads initial cohorts and STRING.
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\01_prepare_expression.py` prepares two cohort expression matrices and metadata.
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\02_combat_correction.py` runs ComBat and PCA QC.
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\03_build_graphs.py` builds graph objects from STRING and expression features.
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\04_baselines.py` trains baseline tabular models.
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\05_train_gnn.py` trains GAT and GCN on graph data.
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\06_summarize_results.py` writes a short results summary.
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\07_prepare_multicohort_real.py` builds a strict human multicohort dataset.
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\08_run_real_external_eval.py` performs leave one dataset out external validation.
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\09_prepare_expanded_real.py` adds mouse OI for expanded analysis.
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\10_tune_5fold_combined.py` runs pooled 5 fold tuning on expanded data.
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\11_human_grouped5_eval.py` human only grouped 5 fold evaluation.
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\12_human_grouped5_tune_fast.py` fast sweep prototype.
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\13_human_grouped5_lr_only_tuning.py` LR only tuning prototype.
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\14_human_grouped5_l2_lr_tuning.py` L2 LR full tuning.

Results and figures
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\results` stores JSON and markdown summaries.
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\figures` stores ROC, PCA, and accuracy plots.
