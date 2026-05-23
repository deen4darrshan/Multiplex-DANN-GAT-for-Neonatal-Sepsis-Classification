# Validation Audit Report

- Generated: 2026-03-05T22:39:27.478796
- Holdout dataset: `GSE26378`

## Leakage Checks
- Overall pass: **True**
- `PASS` fold_1_train_val_sample_disjoint: overlap=0
- `PASS` fold_1_group_disjoint_patient_id: patient_overlap=0
- `PASS` fold_1_coexpr_train_ids_match_fold_train: coexpr_ids=106 train_ids=106
- `PASS` fold_1_no_holdout_in_cv: intersection_with_holdout=0 expected
- `PASS` fold_1_dataset_disjoint_lodo: train_datasets=['GSE54514', 'GSE57065'] val_datasets=['GSE134347']
- `PASS` fold_2_train_val_sample_disjoint: overlap=0
- `PASS` fold_2_group_disjoint_patient_id: patient_overlap=0
- `PASS` fold_2_coexpr_train_ids_match_fold_train: coexpr_ids=292 train_ids=292
- `PASS` fold_2_no_holdout_in_cv: intersection_with_holdout=0 expected
- `PASS` fold_2_dataset_disjoint_lodo: train_datasets=['GSE134347', 'GSE57065'] val_datasets=['GSE54514']
- `PASS` fold_3_train_val_sample_disjoint: overlap=0
- `PASS` fold_3_group_disjoint_patient_id: patient_overlap=0
- `PASS` fold_3_coexpr_train_ids_match_fold_train: coexpr_ids=292 train_ids=292
- `PASS` fold_3_no_holdout_in_cv: intersection_with_holdout=0 expected
- `PASS` fold_3_dataset_disjoint_lodo: train_datasets=['GSE134347', 'GSE54514'] val_datasets=['GSE57065']
- `PASS` holdout_dataset_not_in_train_split: train_datasets=['GSE134347', 'GSE54514', 'GSE57065'] holdout=GSE26378
- `PASS` holdout_split_all_from_target_dataset: holdout_datasets=['GSE26378']

## Statistical Checks
- CV mode: lodo
- Operating threshold (from OOF Youden-J): 0.0109
- CV fold-mean AUROC: 0.8338
- CV pooled OOF AUROC: 0.8477
- External AUROC: 0.9913
- External AUROC 95% CI: 0.9764705882352941 to 1.0
- Best baseline: mlp_only (AUROC=0.8015)
- Model-baseline AUROC delta: 0.046260056534029204
- Permutation p-value: 0.11188811188811189

## Hard Gates
- `leakage_checks_pass`: **True**
- `cv_mean_auroc_ge_0_75`: **True**
- `cv_pooled_oof_auroc_ge_0_75`: **True**
- `external_auroc_ge_0_70`: **True**
- `external_auroc_ci_lower_gt_0_60`: **True**
- `model_auc_improvement_ge_0_05`: **False**
- `permutation_p_lt_0_05`: **False**
- `all_passed`: **False**

## Residual Risks
- GSE95233 fallback policy may alter adult-cohort composition when strict admission parsing yields no sepsis D00 samples.
- External holdout is pediatric; adult-to-pediatric domain shift remains a known risk despite adversarial training.
- Full multi-seed variance analysis is recommended for final claim hardening.