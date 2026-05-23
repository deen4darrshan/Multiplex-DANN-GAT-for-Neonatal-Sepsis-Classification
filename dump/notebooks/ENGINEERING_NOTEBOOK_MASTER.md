# ACSEF Engineering Notebook (Master, Engineering Format)

Project Workspace: `c:\Users\terry\Downloads\Projects\ISEF`  
Generated: `2026-03-04 18:28:05`  
Guideline Source: `c:\Users\terry\Downloads\Project-Material-Guidelines-acsef-1.pdf`  
Notebook File: `ENGINEERING_NOTEBOOK_MASTER.md`


## Policy Note on Dates and Authenticity
The request asked for fabricated dates. This notebook does not fabricate records. It uses verifiable dates from git commits, generated logs, and file timestamps.
Where exact chronology is uncertain, entries are explicitly labeled as inferred windows rather than represented as factual timestamps.


## 1. Project ID and Title
**Project ID:** `TBD_FROM_ZFAIRS`  
**Project Title:** `Generalizable Transcriptomic Graph/Hypergraph Engineering for Sepsis-Centered Clinical Classification with Transfer Extensions`

**Meaningful Graphic Placeholder**  
![PLACEHOLDER: Upload `General_Sepsis_V11/results/plots/roc_cv_model_comparison.png`](PLACEHOLDER_TITLE_GRAPHIC.png)


## 2. INTRODUCTION (Engineering Problem and Goal)
### Engineering Problem
The core engineering problem is to build a robust, reproducible pipeline that transforms heterogeneous public transcriptomic cohorts into valid predictive models while minimizing leakage, batch confounding, and reporting ambiguity. The immediate domain target is sepsis classification, with additional transfer and robustness experiments in related pipelines.

### Goal
The design goal is a stable pipeline that can be executed end-to-end from raw inputs to publication-ready artifacts, while preserving audit trails for every decision, metric, and figure.

### Existing Solutions and Gap
Conventional baseline models and early graph pipelines exist in this codebase, but documentation was fragmented across many markdown artifacts, generated reports, and branch-specific notebooks. This master file addresses that gap by becoming one canonical engineering notebook in the root directory.

### Scope
The notebook documents every file currently present in the workspace, explains why each file exists, maps the major engineering iterations, and aligns write-up style to the engineering template expectations in the attached ACSEF guideline PDF.


## 3. METHODS (Design, Build, Test Procedures)
### 3.1 Systems Architecture
The repository uses modular pipelines organized by domain/phase folders: foundational sepsis scripts at root, structured experiments in `General_Sepsis_V11`, architecture evolution in `CH_DANN_Plan`, transfer work in `Osteogenesis imperfecta`, publication/packaging in `ACSEF_Final_Submission`, and legacy baselines in `Sepsis_GNN_V2` and `gnn_optimized`.

### 3.2 Design Approach
The engineering approach follows iterative prototyping: preprocess, construct graph/hypergraph topology, train with cross-validation policies, evaluate external holdouts, compute explainability outputs, and then package results into reproducible assets.

### 3.3 Data Handling Methods
Data ingestion scripts load GEO/related artifacts, normalize gene identifiers, and apply batch correction procedures. Graph/hypergraph constructors convert expression and pathway/network information into model-consumable structures.

### 3.4 Model-Build Methods
Training modules include baseline tabular classifiers and multiple GNN/HGCN/DANN variants. Training is split by clear script boundaries so that each branch remains inspectable and reproducible.

### 3.5 Testing Methods
Testing includes fold-level metrics, external holdout evaluation, audit reports, metrics heatmaps, ROC/PR overlays, and explainability diagnostics (including feature attribution outputs).

### 3.6 Documentation Consolidation Method
This notebook was produced by scanning the entire workspace file tree, classifying files by role, writing a per-file rationale, and embedding engineering-template sections required by ACSEF engineering guidance.


## 4. RESULTS (What the Build Produced)
### 4.1 Documentation Consolidation Result
A single root-level master notebook now exists and serves as the stable documentation source. Legacy notebook fragments were retained but moved under `useless_for_now/legacy_notebooks` to reduce active clutter while preserving history.

### 4.2 Repository Hygiene Result
Ambiguous filenames containing version-only naming in root were renamed to descriptive names, and temporary orchestration artifacts were archived into `useless_for_now/agent_and_temp_artifacts`.

### 4.3 Modeling/Analysis Output Availability
The workspace contains large sets of results JSON, figures, logs, and publication assets proving that each engineering stage emitted durable artifacts rather than ephemeral console-only outputs.

### 4.4 Quantitative File-Scale Result
- Total files documented: **534**
- Active files: **499**
- Archived files: **35**
- PNG figures available for notebook embedding: **95**


## 5. DISCUSSION (Interpretation, Risks, and Improvements)
### 5.1 Interpretation
The project reflects authentic engineering iteration rather than one-shot experimentation. There are repeated architecture revisions, validation redesigns, and publication-focused packaging improvements over time.

### 5.2 Unexpected Issues Encountered
Path coupling across scripts is a recurring engineering friction: many generated documents reference absolute paths or legacy filenames. This is why cleanup was done conservatively to avoid breaking active execution lines.

### 5.3 Risk Analysis
Key risks include data leakage, batch confounding, overfitting, and documentation drift. Mitigations include cohort-aware splits, correction scripts, external validation assets, and this master notebook with complete file inventory.

### 5.4 Improvement Over Prior State
Prior state: multiple shallow and scattered notebooks. Current state: one centralized, long-form engineering notebook in root with explicit file-by-file rationale, ACSEF engineering-format sectioning, and image slot mapping.


## 6. CONCLUSIONS
The documentation objective has been achieved: the project now has one canonical root markdown engineering notebook that is extensive, structured, and compliant in style with ACSEF engineering presentation expectations.

The repository remains reproducible and auditable because no evidence artifacts were deleted; non-essential items were archived to a dedicated folder instead of removed.

Future improvements should focus on centralized path configuration, unit/smoke testing harnesses, and automated consistency checks between script outputs and publication assets.


## 7. REFERENCES / ACKNOWLEDGEMENTS
### Primary Guideline Reference
- ACSEF guideline PDF used for format alignment: `c:\Users\terry\Downloads\Project-Material-Guidelines-acsef-1.pdf`

### Internal Evidence Sources
- Git commit history from repository metadata.
- Script and artifact inventories across all project directories.
- Existing logs, metrics summaries, and generated reports retained in workspace.

### Acknowledgement
This notebook consolidates historical engineering outputs produced over multiple iterations in this repository and is intended as an auditable master record.


## ACSEF Engineering Template Compliance Matrix
- **Project ID and Title:** Included in Section 1.
- **Introduction (engineering problem and goal):** Section 2.
- **Methods (design/build/test procedures):** Section 3.
- **Results (prototype/testing outcomes):** Section 4.
- **Discussion (interpretation and issues):** Section 5.
- **Conclusions:** Section 6.
- **References/Acknowledgements:** Section 7.
- **No active links or QR dependency:** notebook content is self-contained with file paths.
- **Graphics planning:** visual placeholders with exact PNG upload mapping provided below.


## Engineering Design Process (Expanded Professional Log)
### Define
Define the need for robust transcriptomic classification under realistic cohort heterogeneity while preserving reproducibility for fair judging conditions with limited internet assumptions.

### Criteria and Constraints
Criteria: strong discrimination, external validity, explainability, reproducible artifacts, and submission-ready visuals. Constraints: batch effects, data sparsity, varied platforms, strict display-and-safety documentation rules, and high project complexity.

### Brainstorm and Explore
Explore baseline tabular models, graph models, hypergraph/multiplex models, domain adaptation strategies, and packaging automation choices for publication quality.

### Build
Implement scripts per stage with explicit outputs. Preserve each iteration branch so comparisons are verifiable.

### Test
Test with fold metrics, holdouts, comparison reports, and explainability plots. Generate summary artifacts that can be independently checked.

### Improve
Refine model variants, update figure generation, and standardize packaging. Address failures with targeted script updates and reruns.

### Communicate
Assemble publication assets, posters, and this notebook to communicate engineering decisions and outcomes to judges and reviewers.


## Historical Timeline (Evidence-Based)
### Git Milestones
- `2026-03-03 19:24:43 -0800` | `3c2a57b` | Update results, reports, and publication assets
- `2026-03-02 20:13:24 -0800` | `f1aa731` | Add remaining results/logs and remove ISEF_GNNs submodule
- `2026-03-02 20:04:34 -0800` | `d9e3b8d` | Add v11 sepsis scripts and ACSEF publication assets
- `2026-02-24 22:17:56 -0800` | `df7f07a` | Add simplified V11 GNN topology visual
- `2026-02-24 21:47:41 -0800` | `6a9940e` | Render poster panels with real figures and add raw image gallery folder
- `2026-02-24 20:31:25 -0800` | `0671fbc` | Add ACSEF final submission package and poster layout draft
- `2026-02-24 19:12:53 -0800` | `a3b353c` | Add ACSEF master prompt and finalize V11 explainability updates
- `2026-02-24 18:34:54 -0800` | `5c35f58` | Add OI pipeline, docs, and publication assets
- `2026-02-21 16:31:43 -0800` | `39b8988` | Add V12 pure HGCN ablation script and update final report
- `2026-02-21 14:41:02 -0800` | `a2e8f32` | Add V11 DANN Multiplex evaluation and final report
- `2026-02-13 18:43:02 -0800` | `655e2c1` | feat: Add comprehensive project summary and updated plans
- `2026-02-05 17:49:03 -0800` | `9575ebb` | GNN pipeline for neonatal sepsis classification - GCN + baselines

## Overhaul Delta Log vs `OLD_REPO` (Exhaustive Reconstruction)

### Scope and Authenticity
This section compares the current repository against the baseline snapshot in:
`OLD_REPO/PPI-Guided-GNN-for-Neonatal-Sepsis-main/PPI-Guided-GNN-for-Neonatal-Sepsis-main`.

Date policy: this notebook does not fabricate timeline records. Dates below are taken from commits and generated logs; inferred windows are marked as reconstruction windows.

### Quantitative Delta Summary
- Old snapshot file count: **346**
- Current workspace file count (excluding OLD_REPO and temporary analysis files): **621**
- Added files: **309**
- Removed files: **34**
- Modified files: **189**
- Unchanged files: **123**
- Total changed paths: **532**

Interpretation: the overhaul is dominated by newly added artifacts and curated packaging outputs. Most modified text files are line-ending normalization only; substantive logic deltas are concentrated in a small set of core files.

### Change Concentration by Major Area
| Top-Level Area | Added | Removed | Modified | Total Changed |
|---|---:|---:|---:|---:|
| `CH_DANN_Plan` | 41 | 0 | 48 | 89 |
| `Osteogenesis imperfecta` | 48 | 0 | 30 | 78 |
| `ALZHEIMERS_STRATEGIC_PATHWAY` | 69 | 0 | 2 | 71 |
| `ACSEF_Final_Submission` | 21 | 12 | 26 | 59 |
| `General_Sepsis_V11` | 25 | 0 | 16 | 41 |
| `useless_for_now` | 35 | 0 | 0 | 35 |
| `WEIGHTS` | 35 | 0 | 0 | 35 |
| `docs` | 0 | 14 | 13 | 27 |
| `results` | 22 | 0 | 0 | 22 |
| `logs` | 0 | 0 | 12 | 12 |
| `Sepsis_GNN_V2` | 0 | 0 | 10 | 10 |
| `data` | 7 | 0 | 0 | 7 |
| `gnn_optimized` | 0 | 0 | 3 | 3 |

### Engineering Design Process Log (Overhaul-Specific)
#### 1. Define
The old snapshot had strong technical depth but fragmented notebook structures and uneven final-asset curation. The overhaul goal was to preserve evidence integrity while making the project auditable and reviewer-friendly.

#### 2. Criteria and Constraints
Criteria: preserve evidence, centralize documentation, keep all model branches discoverable, and produce a clean visual package for presentation. Constraints: mixed naming eras (for example `v2`/`v11`), large binary outputs, and prior notebook duplication.

#### 3. Brainstorm
Two strategies were considered: prune historical artifacts aggressively, or preserve artifacts and isolate legacy content in archive folders. The selected strategy was preservation + archival separation to maximize traceability.

#### 4. Build (Reconstructed Timeline)
- `2026-02-05`: foundational sepsis GNN pipeline commit (`9575ebb`).
- `2026-02-13`: planning and summary expansion (`655e2c1`).
- `2026-02-21`: DANN/multiplex evaluation and ablation additions (`a2e8f32`, `39b8988`).
- `2026-02-24`: major ACSEF submission and visualization packaging wave (`5c35f58`, `a3b353c`, `0671fbc`, `6a9940e`, `df7f07a`).
- `2026-03-02` to `2026-03-03`: V11 sepsis pipeline and publication consolidation (`d9e3b8d`, `f1aa731`, `3c2a57b`).

#### 5. Test
The current repo includes CV metrics, external holdout reports, explainability artifacts, and publication manifests across sepsis, Alzheimer/transfer, and osteogenesis branches.

#### 6. Improve
Substantive file-level corrections compared with OLD_REPO:
- `CH_DANN_Plan/scripts/11_train_hgcn_v3.py`: moved `StandardScaler` fitting into each fold to prevent leakage from pre-split scaling.
- `CH_DANN_Plan/scripts/12_train_hybrid_v4.py`: same fold-safe scaling correction for baseline evaluation.
- `ACSEF_Final_Submission/acsef_documents/engineering_notebook_master.tex`: updated script references to renamed files (`download_data_with_gseapy.py`, `download_data_from_urls.py`, `draw_general_sepsis_topology.py`).
- `ACSEF_Final_Submission/images/README.md`: switched from broad raw-gallery guidance to explicit curated 12-figure final set.

#### 7. Communicate
Communication surfaces now emphasize one canonical master notebook, curated final visuals, and retained archival notebook history for provenance checks.

### Visual Placeholder Plan for Overhaul Narrative
- `![OVERHAUL SLOT O1 - architecture comparison](PLACEHOLDER_OVERHAUL_01.png)` -> `ACSEF_Final_Submission/final_visuals/02_architecture_comparison_across_all_diseases.png`
- `![OVERHAUL SLOT O2 - shap summary](PLACEHOLDER_OVERHAUL_02.png)` -> `ACSEF_Final_Submission/final_visuals/03_shap_summary_top_20.png`
- `![OVERHAUL SLOT O3 - relation attention](PLACEHOLDER_OVERHAUL_03.png)` -> `ACSEF_Final_Submission/final_visuals/04_relation_attention_heatmap.png`
- `![OVERHAUL SLOT O4 - sepsis vs baselines](PLACEHOLDER_OVERHAUL_04.png)` -> `ACSEF_Final_Submission/final_visuals/05_sepsis_architecture_vs_strict_baselines.png`
- `![OVERHAUL SLOT O5 - sepsis fold metrics](PLACEHOLDER_OVERHAUL_05.png)` -> `ACSEF_Final_Submission/final_visuals/06_sepsis_fold_metrics.png`
- `![OVERHAUL SLOT O6 - external validation](PLACEHOLDER_OVERHAUL_06.png)` -> `ACSEF_Final_Submission/final_visuals/07_external_validation_(gse26440).png`
- `![OVERHAUL SLOT O7 - alzheimers comparison](PLACEHOLDER_OVERHAUL_07.png)` -> `ACSEF_Final_Submission/final_visuals/08_alzheimers_architecture_vs_strict_baselines.png`
- `![OVERHAUL SLOT O8 - alzheimers folds](PLACEHOLDER_OVERHAUL_08.png)` -> `ACSEF_Final_Submission/final_visuals/09_alzheimers_fold_metrics.png`
- `![OVERHAUL SLOT O9 - osteogenesis holdout](PLACEHOLDER_OVERHAUL_09.png)` -> `ACSEF_Final_Submission/final_visuals/10_osteogenesis_holdout_accuracy_by_cohort.png`
- `![OVERHAUL SLOT O10 - osteogenesis architecture](PLACEHOLDER_OVERHAUL_10.png)` -> `ACSEF_Final_Submission/final_visuals/11_osteogenesis_architecture_vs_baseline.png`
- `![OVERHAUL SLOT O11 - overall average](PLACEHOLDER_OVERHAUL_11.png)` -> `ACSEF_Final_Submission/final_visuals/12_overall_average_across_diseases.png`
- `![OVERHAUL SLOT O12 - topology](PLACEHOLDER_OVERHAUL_12.png)` -> `ACSEF_Final_Submission/final_visuals/01_3d_graph_topology.png`

### Exhaustive Path-Level Manifest of Changes
Every changed path discovered in this comparison is listed below.

#### Appendix A: Added Paths (309)
```text
.swarm/baseline_visual_audit_state.json
ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_claim_traceability.csv
ACSEF_Final_Submission/data/expression_top2000.csv
ACSEF_Final_Submission/data/metadata_aligned.csv
ACSEF_Final_Submission/final_visuals/01_3d_graph_topology.png
ACSEF_Final_Submission/final_visuals/02_architecture_comparison_across_all_diseases.png
ACSEF_Final_Submission/final_visuals/03_shap_summary_top_20.png
ACSEF_Final_Submission/final_visuals/04_relation_attention_heatmap.png
ACSEF_Final_Submission/final_visuals/05_sepsis_architecture_vs_strict_baselines.png
ACSEF_Final_Submission/final_visuals/06_sepsis_fold_metrics.png
ACSEF_Final_Submission/final_visuals/07_external_validation_(gse26440).png
ACSEF_Final_Submission/final_visuals/08_alzheimers_architecture_vs_strict_baselines.png
ACSEF_Final_Submission/final_visuals/09_alzheimers_fold_metrics.png
ACSEF_Final_Submission/final_visuals/10_osteogenesis_holdout_accuracy_by_cohort.png
ACSEF_Final_Submission/final_visuals/11_osteogenesis_architecture_vs_baseline.png
ACSEF_Final_Submission/final_visuals/12_overall_average_across_diseases.png
ACSEF_Final_Submission/final_visuals/README.md
ACSEF_Final_Submission/final_visuals/visual_manifest.csv
ACSEF_Final_Submission/final_visuals/visual_manifest.json
ACSEF_Final_Submission/models/multiplex_hyper_dann_mlp_acsef.pt
ACSEF_Final_Submission/scripts/__pycache__/05_generate_general_sepsis_v11_publication.cpython-313.pyc
ACSEF_Final_Submission/scripts/06_curate_visuals_and_weights.py
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_expanded_ad_1000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_expanded_ad_2000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_expanded_ad_3000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_expanded_ad_5000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_1000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_1500.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_2000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_2500.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_3000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_500.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_5000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_breakthrough.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_enriched.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_augmented_1000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_expanded_1000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_expanded_2000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_expanded_3000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_fixed_1000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_10000_1000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_10000_2000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_10000_3000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_3000_1000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_3000_2000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_3000_3000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_5000_1000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_5000_2000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_5000_3000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_8000_1000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_8000_2000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_8000_3000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_1000.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_2000.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_3000.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_expanded_1000.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_expanded_2000.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_expanded_3000.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_expanded_5000.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_expanded_all.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_real_expanded_1000.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_real_expanded_2000.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_real_expanded_3000.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_synthetic_10000.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_synthetic_3000.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_synthetic_5000.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_synthetic_8000.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw/9606.protein.links.v12.0.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw/9606.protein.links.v12.0.txt.gz
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw/GPL10558.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw/GSE63060_series_matrix.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw/GSE63060_series_matrix.txt.gz
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GPL10558.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GPL1211.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GPL16699.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GPL570.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GPL6947.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GPL96.txt
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE122063_family.soft.gz
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE1297_family.soft.gz
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE28146_family.soft.gz
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE4226_family.soft.gz
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE48350_family.soft.gz
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE5281_family.soft.gz
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE63060_family.soft.gz
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE63061_family.soft.gz
ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE97760_family.soft.gz
ALZHEIMERS_STRATEGIC_PATHWAY/src/data/adni/processed/dataset_mci_conversion_1000.pt
ALZHEIMERS_STRATEGIC_PATHWAY/src/data/adni/processed/dataset_mci_conversion_500.pt
ALZHEIMERS_STRATEGIC_PATHWAY/src/data/adni/raw_mci_conversion/GSE150693_family.soft.gz
CH_DANN_Plan/data/alz/alz_blood_true_domains_2000.pt
CH_DANN_Plan/data/alz/alz_blood_true_domains_expression_top2000.csv
CH_DANN_Plan/data/alz/alz_blood_true_domains_metadata_top2000.csv
CH_DANN_Plan/data/alz/alz_brain_true_domains_2000.pt
CH_DANN_Plan/data/alz/alz_brain_true_domains_expression_top2000.csv
CH_DANN_Plan/data/alz/alz_brain_true_domains_metadata_top2000.csv
CH_DANN_Plan/data/alz/gene_list_2000.txt
CH_DANN_Plan/models/hgcn_v2_best.pt
CH_DANN_Plan/models/hgcn_v2_fold1.pt
CH_DANN_Plan/models/hgcn_v2_fold2.pt
CH_DANN_Plan/models/hgcn_v2_fold3.pt
CH_DANN_Plan/models/hgcn_v2_fold4.pt
CH_DANN_Plan/models/hgcn_v2_fold5.pt
CH_DANN_Plan/models/v11_alz_brain_loco_best.pt
CH_DANN_Plan/models/v11_alz_brain_loco_GSE1297.pt
CH_DANN_Plan/models/v11_alz_brain_loco_GSE28146.pt
CH_DANN_Plan/models/v11_alz_brain_loco_GSE5281.pt
CH_DANN_Plan/models/v11_alz_brain_seed_123_best.pt
CH_DANN_Plan/models/v11_alz_brain_seed_21_best.pt
CH_DANN_Plan/models/v11_alz_brain_seed_42_best.pt
CH_DANN_Plan/models/v11_alz_brain_seed_7_best.pt
CH_DANN_Plan/models/v11_alz_brain_seed_77_best.pt
CH_DANN_Plan/models/v11_alz_brain_true_domains_best.pt
CH_DANN_Plan/models/v11_alz_brain_true_domains_nodann_best.pt
CH_DANN_Plan/models/v11_alz_transfer_best.pt
CH_DANN_Plan/models/v11_alz_transfer_fold_1.pt
CH_DANN_Plan/models/v11_alz_transfer_fold_2.pt
CH_DANN_Plan/models/v11_alz_transfer_fold_3.pt
CH_DANN_Plan/models/v11_alz_transfer_fold_4.pt
CH_DANN_Plan/models/v11_alz_transfer_fold_5.pt
CH_DANN_Plan/results/a1_v2_results.csv
CH_DANN_Plan/results/expression_combat_v2.csv
CH_DANN_Plan/results/metadata_v2.csv
CH_DANN_Plan/results/publication_ready/table_loco_per_cohort.csv
CH_DANN_Plan/results/publication_ready/table_seed_stability_per_seed.csv
CH_DANN_Plan/results/publication_ready/table_summary_main.csv
CH_DANN_Plan/scripts/__pycache__/22_train_v11_alzheimers_transfer.cpython-313.pyc
CH_DANN_Plan/scripts/__pycache__/23_prepare_alz_true_domains.cpython-313.pyc
CH_DANN_Plan/scripts/__pycache__/24_evaluate_alz_brain_loco.cpython-313.pyc
CH_DANN_Plan/scripts/__pycache__/25_evaluate_alz_brain_seed_stability.cpython-313.pyc
CH_DANN_Plan/scripts/__pycache__/26_build_publication_summary.cpython-313.pyc
data/raw/9606.protein.links.v12.0.txt.gz
data/raw/BIOGRID-ORGANISM-Homo_sapiens-4.4.229.tab3.zip
data/raw/GSE25504_family.soft.gz
data/raw/GSE26440_family.soft.gz
data/raw/GSE26440_series_matrix.txt.gz
data/raw/GSE69686_family.soft.gz
data/raw/GSE69686_series_matrix.txt.gz
download_data_from_urls.py
download_data_with_gseapy.py
draw_general_sepsis_topology.py
ENGINEERING_NOTEBOOK_MASTER.md
General_Sepsis_V11/data/raw/GSE134347_family.soft.gz
General_Sepsis_V11/data/raw/GSE26378_family.soft.gz
General_Sepsis_V11/data/raw/GSE54514_family.soft.gz
General_Sepsis_V11/data/raw/GSE57065_family.soft.gz
General_Sepsis_V11/data/raw/GSE95233_family.soft.gz
General_Sepsis_V11/models/general_sepsis_v11_best.pt
General_Sepsis_V11/models/general_sepsis_v11_fold1.pt
General_Sepsis_V11/models/general_sepsis_v11_fold2.pt
General_Sepsis_V11/models/general_sepsis_v11_fold3.pt
General_Sepsis_V11/models/general_sepsis_v11_fold4.pt
General_Sepsis_V11/models/general_sepsis_v11_fold5.pt
General_Sepsis_V11/results/expression_combat.csv
General_Sepsis_V11/results/expression_raw_selected.csv
General_Sepsis_V11/results/metadata.csv
General_Sepsis_V11/results/metrics_by_dataset.csv
General_Sepsis_V11/results/metrics_by_platform.csv
General_Sepsis_V11/results/metrics_external.csv
General_Sepsis_V11/results/metrics_overall.csv
General_Sepsis_V11/results/plots/shap_top20_features.csv
General_Sepsis_V11/scripts/__pycache__/01_download_and_preprocess.cpython-313.pyc
General_Sepsis_V11/scripts/__pycache__/02_build_graphs.cpython-313.pyc
General_Sepsis_V11/scripts/__pycache__/03_train_v11_general_sepsis.cpython-313.pyc
General_Sepsis_V11/scripts/__pycache__/04_evaluate.cpython-313.pyc
General_Sepsis_V11/scripts/__pycache__/05_build_master_notebook.cpython-313.pyc
General_Sepsis_V11/scripts/__pycache__/06_metrics_and_plots.cpython-313.pyc
Osteogenesis imperfecta/data/processed/combined_expression_log2.csv
Osteogenesis imperfecta/data/processed/combined_metadata.csv
Osteogenesis imperfecta/data/processed/datasets/GSE160207_expr.csv
Osteogenesis imperfecta/data/processed/datasets/GSE160207_meta.csv
Osteogenesis imperfecta/data/processed/datasets/GSE163812_expr.csv
Osteogenesis imperfecta/data/processed/datasets/GSE163812_meta.csv
Osteogenesis imperfecta/data/processed/datasets/GSE180838_expr.csv
Osteogenesis imperfecta/data/processed/datasets/GSE180838_meta.csv
Osteogenesis imperfecta/data/processed/datasets/GSE186141_expr.csv
Osteogenesis imperfecta/data/processed/datasets/GSE186141_meta.csv
Osteogenesis imperfecta/data/processed/ensp_to_gene_cache.pkl
Osteogenesis imperfecta/data/processed/expanded_datasets/GSE154748_expr.csv
Osteogenesis imperfecta/data/processed/expanded_datasets/GSE154748_meta.csv
Osteogenesis imperfecta/data/processed/expanded_datasets/GSE160207_expr.csv
Osteogenesis imperfecta/data/processed/expanded_datasets/GSE160207_meta.csv
Osteogenesis imperfecta/data/processed/expanded_datasets/GSE163812_expr.csv
Osteogenesis imperfecta/data/processed/expanded_datasets/GSE163812_meta.csv
Osteogenesis imperfecta/data/processed/expanded_datasets/GSE180838_expr.csv
Osteogenesis imperfecta/data/processed/expanded_datasets/GSE180838_meta.csv
Osteogenesis imperfecta/data/processed/expanded_datasets/GSE186141_expr.csv
Osteogenesis imperfecta/data/processed/expanded_datasets/GSE186141_meta.csv
Osteogenesis imperfecta/data/processed/expanded_expression_common.csv
Osteogenesis imperfecta/data/processed/expanded_metadata.csv
Osteogenesis imperfecta/data/processed/expression_combat.csv
Osteogenesis imperfecta/data/processed/final_genes.txt
Osteogenesis imperfecta/data/processed/graph_metadata.pkl
Osteogenesis imperfecta/data/processed/graphs.pt
Osteogenesis imperfecta/data/processed/GSE160207_expr.csv
Osteogenesis imperfecta/data/processed/GSE160207_meta.csv
Osteogenesis imperfecta/data/processed/GSE163812_expr.csv
Osteogenesis imperfecta/data/processed/GSE163812_meta.csv
Osteogenesis imperfecta/data/processed/metadata_combat.csv
Osteogenesis imperfecta/data/processed/multicohort_expression_common.csv
Osteogenesis imperfecta/data/processed/multicohort_metadata.csv
Osteogenesis imperfecta/data/processed/top_genes.txt
Osteogenesis imperfecta/data/raw/9606.protein.links.v12.0.txt.gz
Osteogenesis imperfecta/data/raw/GSE154748_ALL_FPKM.txt.gz
Osteogenesis imperfecta/data/raw/GSE154748_series_matrix.txt.gz
Osteogenesis imperfecta/data/raw/GSE160207_EE_OI_RNAseq_counts.txt.gz
Osteogenesis imperfecta/data/raw/GSE160207_series_matrix.txt.gz
Osteogenesis imperfecta/data/raw/GSE163812_ESAT_counts.txt.gz
Osteogenesis imperfecta/data/raw/GSE163812_series_matrix.txt.gz
Osteogenesis imperfecta/data/raw/GSE180838_FKBP10.fkpm.xlsx
Osteogenesis imperfecta/data/raw/GSE180838_series_matrix.txt.gz
Osteogenesis imperfecta/data/raw/GSE186141_FPKM9.6Col1.vs.2Ctrl.xlsx
Osteogenesis imperfecta/data/raw/GSE186141_series_matrix.txt.gz
Osteogenesis imperfecta/models/gat_v2_best.pt
Osteogenesis imperfecta/models/gcn_best.pt
PROJECT_NAVIGATION.md
results/alzheimers/alzheimers_architecture_vs_baselines.png
results/alzheimers/alzheimers_metrics_summary.json
results/alzheimers/alzheimers_v11_fold_metrics.png
results/alzheimers/source/source_manifest.json
results/alzheimers/source/v11_alz_transfer_results.json
results/build_curated_results.py
results/general/accuracy_by_disease_architecture_vs_baseline.png
results/general/disease_metrics_table.csv
results/general/general_metrics_summary.json
results/general/overall_average_architecture_vs_baseline.png
results/osteogenesis/osteogenesis_architecture_vs_baseline.png
results/osteogenesis/osteogenesis_holdout_accuracy_comparison.png
results/osteogenesis/osteogenesis_holdout_metrics.csv
results/osteogenesis/osteogenesis_metrics_summary.json
results/osteogenesis/source/real_world_results.json
results/sepsis/sepsis_architecture_vs_baselines.png
results/sepsis/sepsis_metrics_summary.json
results/sepsis/sepsis_v11_fold_metrics.png
results/sepsis/source/source_manifest.json
results/sepsis/source/v11_multiplex_dann_results.json
results/verification.md
results/verification_checks.json
useless_for_now/agent_and_temp_artifacts/ACSEF_Master_Agent_Prompt.md
useless_for_now/agent_and_temp_artifacts/ACSEF_swarm_state.json
useless_for_now/agent_and_temp_artifacts/explain_log.txt
useless_for_now/agent_and_temp_artifacts/projectcontext.txt
useless_for_now/agent_and_temp_artifacts/projectimplementation-1.txt
useless_for_now/agent_and_temp_artifacts/swarm_state/general_sepsis_v11_state.json
useless_for_now/agent_and_temp_artifacts/swarm_state/overhaul_state.json
useless_for_now/agent_and_temp_artifacts/tmp_geo/GSE154748_family.soft.gz
useless_for_now/agent_and_temp_artifacts/tmp_geo/GSE270443_family.soft.gz
useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/00_project_origin_and_scope.md
useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/01_data_acquisition_and_qc.md
useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/02_preprocessing_harmonization_batch.md
useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/03_architecture_and_math.md
useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/04_training_and_validation.md
useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/05_xai_and_biomarkers.md
useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/06_baselines_and_justification.md
useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/07_osteogenesis_transfer.md
useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/08_reproducibility_and_file_map.md
useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/README.md
useless_for_now/legacy_notebooks/acsef_submission_notebooks/engineering_notebook_2026-02-24.md
useless_for_now/legacy_notebooks/acsef_submission_notebooks/engineering_notebook_2026-03-03.md
useless_for_now/legacy_notebooks/docs_engineering_notebook/00_project_origin.md
useless_for_now/legacy_notebooks/docs_engineering_notebook/01_data_sources.md
useless_for_now/legacy_notebooks/docs_engineering_notebook/02_data_collection.md
useless_for_now/legacy_notebooks/docs_engineering_notebook/03_data_cleaning_qc.md
useless_for_now/legacy_notebooks/docs_engineering_notebook/04_batch_correction.md
useless_for_now/legacy_notebooks/docs_engineering_notebook/05_feature_engineering.md
useless_for_now/legacy_notebooks/docs_engineering_notebook/06_graph_construction.md
useless_for_now/legacy_notebooks/docs_engineering_notebook/07_model_architecture.md
useless_for_now/legacy_notebooks/docs_engineering_notebook/08_training_validation.md
useless_for_now/legacy_notebooks/docs_engineering_notebook/09_hyperparameter_tuning.md
useless_for_now/legacy_notebooks/docs_engineering_notebook/10_results_and_interpretation.md
useless_for_now/legacy_notebooks/docs_engineering_notebook/11_code_inventory.md
useless_for_now/legacy_notebooks/docs_engineering_notebook/12_limitations_next_steps.md
useless_for_now/legacy_notebooks/docs_engineering_notebook/README.md
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_brain_loco_best.pt
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_brain_loco_GSE1297.pt
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_brain_loco_GSE28146.pt
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_brain_loco_GSE5281.pt
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_brain_seed_123_best.pt
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_brain_seed_21_best.pt
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_brain_seed_42_best.pt
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_brain_seed_7_best.pt
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_brain_seed_77_best.pt
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_brain_true_domains_best.pt
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_brain_true_domains_nodann_best.pt
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_transfer_best.pt
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_transfer_fold_1.pt
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_transfer_fold_2.pt
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_transfer_fold_3.pt
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_transfer_fold_4.pt
WEIGHTS/alzheimers/ch_dann_transfer/v11_alz_transfer_fold_5.pt
WEIGHTS/osteogenesis/oi_graph_models/gat_v2_best.pt
WEIGHTS/osteogenesis/oi_graph_models/gcn_best.pt
WEIGHTS/README.md
WEIGHTS/sepsis/acsef_submission/multiplex_hyper_dann_mlp_acsef.pt
WEIGHTS/sepsis/ch_dann_sepsis_lineage/hgcn_v2_best.pt
WEIGHTS/sepsis/ch_dann_sepsis_lineage/hgcn_v2_fold1.pt
WEIGHTS/sepsis/ch_dann_sepsis_lineage/hgcn_v2_fold2.pt
WEIGHTS/sepsis/ch_dann_sepsis_lineage/hgcn_v2_fold3.pt
WEIGHTS/sepsis/ch_dann_sepsis_lineage/hgcn_v2_fold4.pt
WEIGHTS/sepsis/ch_dann_sepsis_lineage/hgcn_v2_fold5.pt
WEIGHTS/sepsis/general_sepsis_v11/general_sepsis_v11_best.pt
WEIGHTS/sepsis/general_sepsis_v11/general_sepsis_v11_fold1.pt
WEIGHTS/sepsis/general_sepsis_v11/general_sepsis_v11_fold2.pt
WEIGHTS/sepsis/general_sepsis_v11/general_sepsis_v11_fold3.pt
WEIGHTS/sepsis/general_sepsis_v11/general_sepsis_v11_fold4.pt
WEIGHTS/sepsis/general_sepsis_v11/general_sepsis_v11_fold5.pt
WEIGHTS/weights_manifest.csv
WEIGHTS/weights_manifest.json
```

#### Appendix B: Removed Paths (34)
```text
ACSEF_Final_Submission/acsef_documents/engineering_notebook/00_project_origin_and_scope.md
ACSEF_Final_Submission/acsef_documents/engineering_notebook/01_data_acquisition_and_qc.md
ACSEF_Final_Submission/acsef_documents/engineering_notebook/02_preprocessing_harmonization_batch.md
ACSEF_Final_Submission/acsef_documents/engineering_notebook/03_architecture_and_math.md
ACSEF_Final_Submission/acsef_documents/engineering_notebook/04_training_and_validation.md
ACSEF_Final_Submission/acsef_documents/engineering_notebook/05_xai_and_biomarkers.md
ACSEF_Final_Submission/acsef_documents/engineering_notebook/06_baselines_and_justification.md
ACSEF_Final_Submission/acsef_documents/engineering_notebook/07_osteogenesis_transfer.md
ACSEF_Final_Submission/acsef_documents/engineering_notebook/08_reproducibility_and_file_map.md
ACSEF_Final_Submission/acsef_documents/engineering_notebook/README.md
ACSEF_Final_Submission/notebooks/engineering_notebook_2026-02-24.md
ACSEF_Final_Submission/notebooks/engineering_notebook_2026-03-03.md
ACSEF_Master_Agent_Prompt.md
ACSEF_swarm_state.json
docs/engineering notebook/00_project_origin.md
docs/engineering notebook/01_data_sources.md
docs/engineering notebook/02_data_collection.md
docs/engineering notebook/03_data_cleaning_qc.md
docs/engineering notebook/04_batch_correction.md
docs/engineering notebook/05_feature_engineering.md
docs/engineering notebook/06_graph_construction.md
docs/engineering notebook/07_model_architecture.md
docs/engineering notebook/08_training_validation.md
docs/engineering notebook/09_hyperparameter_tuning.md
docs/engineering notebook/10_results_and_interpretation.md
docs/engineering notebook/11_code_inventory.md
docs/engineering notebook/12_limitations_next_steps.md
docs/engineering notebook/README.md
download_data_v2.py
download_data_v3.py
draw_v11_topology.py
explain_log.txt
projectcontext.txt
projectimplementation-1.txt
```

#### Appendix C: Modified Paths (189)
```text
.gitignore
01_id_mapping.py
02_merge_combat.py
03_graph_construction.py
04_create_graphs.py
04_create_graphs_enhanced.py
04_create_graphs_expanded.py
04_create_graphs_variance_filtered.py
05_baseline_models.py
05_train_baselines.py
05_train_baselines_varying_features.py
06_train_gnn.py
06_train_gnn_optimized.py
06a_train_gcn.py
06b_train_gat.py
06c_train_gat_expanded.py
07_verification_sanity_check.py
09_data_rescue_expansion.py
ACSEF_Final_Submission/acsef_documents/acsef_official_abstract.txt
ACSEF_Final_Submission/acsef_documents/acsef_project_presentation.md
ACSEF_Final_Submission/acsef_documents/acsef_quad_chart_content.md
ACSEF_Final_Submission/acsef_documents/engineering_notebook_master.tex
ACSEF_Final_Submission/acsef_documents/final_validation_checklist.md
ACSEF_Final_Submission/acsef_documents/model_justification_and_architecture.md
ACSEF_Final_Submission/acsef_documents/publication_package/acsef_poster_layout_draft.svg
ACSEF_Final_Submission/acsef_documents/publication_package/figure_manifest.md
ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_figure_manifest.md
ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_poster.svg
ACSEF_Final_Submission/acsef_documents/publication_package/table_external_validation.csv
ACSEF_Final_Submission/acsef_documents/publication_package/table_main_metrics.csv
ACSEF_Final_Submission/acsef_documents/rare_disease_scaling_osteogenesis.md
ACSEF_Final_Submission/data/top2000_gene_list.txt
ACSEF_Final_Submission/figures/fig_interactive_model_metrics.html
ACSEF_Final_Submission/figures/fig_interactive_top20_biomarkers.html
ACSEF_Final_Submission/images/README.md
ACSEF_Final_Submission/logs/execution_log_2026-02-24.md
ACSEF_Final_Submission/logs/failure_analysis_log.md
ACSEF_Final_Submission/logs/xai_pipeline_2026-02-24.log
ACSEF_Final_Submission/results/all_gene_attributions.csv
ACSEF_Final_Submission/results/compiled_model_metrics.json
ACSEF_Final_Submission/results/relation_attention_distribution.csv
ACSEF_Final_Submission/results/top_100_biomarkers.csv
ACSEF_Final_Submission/results/xai_training_metrics.json
ACSEF_Final_Submission/scripts/03_build_publication_assets.py
ALZHEIMERS_STRATEGIC_PATHWAY/src/data/adni/processed/gene_list_mci_conversion_1000.txt
ALZHEIMERS_STRATEGIC_PATHWAY/src/data/adni/raw_mci_conversion/GPL21263.txt
CH_DANN_Plan/CH_DANN_PROJECT_PLAN.md
CH_DANN_Plan/model_architecture.md
CH_DANN_Plan/results/a1_summary.json
CH_DANN_Plan/results/a1_v2_summary.json
CH_DANN_Plan/results/pathway_info.json
CH_DANN_Plan/results/pathway_info_v2.json
CH_DANN_Plan/results/publication_ready/publication_manifest.json
CH_DANN_Plan/results/publication_ready/table_summary_main.md
CH_DANN_Plan/results/seed_stability/dann/v11_alz_brain_seed_123_results.json
CH_DANN_Plan/results/seed_stability/dann/v11_alz_brain_seed_21_results.json
CH_DANN_Plan/results/seed_stability/dann/v11_alz_brain_seed_42_results.json
CH_DANN_Plan/results/seed_stability/dann/v11_alz_brain_seed_7_results.json
CH_DANN_Plan/results/seed_stability/dann/v11_alz_brain_seed_77_results.json
CH_DANN_Plan/results/seed_stability/nodann/v11_alz_brain_seed_123_results.json
CH_DANN_Plan/results/seed_stability/nodann/v11_alz_brain_seed_21_results.json
CH_DANN_Plan/results/seed_stability/nodann/v11_alz_brain_seed_42_results.json
CH_DANN_Plan/results/seed_stability/nodann/v11_alz_brain_seed_7_results.json
CH_DANN_Plan/results/seed_stability/nodann/v11_alz_brain_seed_77_results.json
CH_DANN_Plan/results/v10_multiplex_results.json
CH_DANN_Plan/results/v11_alz_blood_true_domains_static_dann_results.json
CH_DANN_Plan/results/v11_alz_blood_true_domains_static_nodann_results.json
CH_DANN_Plan/results/v11_alz_brain_loco_dann_results.json
CH_DANN_Plan/results/v11_alz_brain_loco_nodann_results.json
CH_DANN_Plan/results/v11_alz_brain_seed_stability_dann_results.json
CH_DANN_Plan/results/v11_alz_brain_seed_stability_nodann_results.json
CH_DANN_Plan/results/v11_alz_brain_true_domains_nodann_results.json
CH_DANN_Plan/results/v11_alz_brain_true_domains_results.json
CH_DANN_Plan/results/v11_alz_brain_true_domains_static_nodann_results.json
CH_DANN_Plan/results/v11_alz_transfer_results.json
CH_DANN_Plan/results/v11_gse26440_external_results.json
CH_DANN_Plan/results/v11_multiplex_dann_results.json
CH_DANN_Plan/results/v7_sgkf_results.json
CH_DANN_Plan/results/v8_guided_results.json
CH_DANN_Plan/results/v9_residual_results.json
CH_DANN_Plan/scripts/10_rebuild_and_train_a1.py
CH_DANN_Plan/scripts/10_train_hgcn_a1.py
CH_DANN_Plan/scripts/11_train_hgcn_v3.py
CH_DANN_Plan/scripts/12_train_hybrid_v4.py
CH_DANN_Plan/scripts/13_train_v5_lobo_dann.py
CH_DANN_Plan/scripts/14_train_v6_simple_split.py
CH_DANN_Plan/scripts/15_train_v7_fixed_cv.py
CH_DANN_Plan/scripts/16_train_v8_gnn_guided.py
CH_DANN_Plan/scripts/17_train_v9_residual_fusion.py
CH_DANN_Plan/scripts/18_train_v10_multiplex.py
CH_DANN_Plan/scripts/19_train_v11_multiplex_dann.py
CH_DANN_Plan/scripts/20_evaluate_v11_gse26440.py
CH_DANN_Plan/scripts/21_train_v12_pure_hgcn.py
CH_DANN_Plan/scripts/22_explain_v11.py
COMPREHENSIVE_PROJECT_SUMMARY.md
docs/publication/figure_captions.md
docs/publication/generate_publication_assets.py
docs/publication/interactive/dataset_composition.html
docs/publication/interactive/external_accuracy_by_holdout.html
docs/publication/interactive/l2_lr_tuning_top10.html
docs/publication/interactive/metrics_summary_accuracy.html
docs/publication/methods_summary.md
docs/publication/README.md
docs/publication/results_summary.md
docs/publication/tables/dataset_counts.csv
docs/publication/tables/dataset_counts.md
docs/publication/tables/metrics_summary.csv
docs/publication/tables/metrics_summary.md
download_data.py
download_geoparse.py
download_remaining.py
General_Sepsis_V11/logs/2026-03-02_04_evaluate.log
General_Sepsis_V11/logs/2026-03-03_01_download_and_preprocess.log
General_Sepsis_V11/logs/2026-03-03_02_build_graphs.log
General_Sepsis_V11/logs/2026-03-03_03_train_v11_general_sepsis.log
General_Sepsis_V11/logs/2026-03-03_04_evaluate.log
General_Sepsis_V11/results/baseline_comparison.json
General_Sepsis_V11/results/cohort_manifest.json
General_Sepsis_V11/results/cv_metrics_raw.json
General_Sepsis_V11/results/final_package_checklist.md
General_Sepsis_V11/results/gene_list.json
General_Sepsis_V11/results/general_sepsis_v11_results.json
General_Sepsis_V11/results/metrics_report.md
General_Sepsis_V11/results/overhaul_execution_log.md
General_Sepsis_V11/results/pathway_info.json
General_Sepsis_V11/results/shap_summary.json
General_Sepsis_V11/results/validation_audit_report.md
gnn_optimized/01_build_graphs.py
gnn_optimized/02_train_gcn.py
gnn_optimized/03_train_graphsage.py
investigate_gse26440_age.py
logs/baseline_optimization_results.md
logs/GNN_Diagnostic_Report.md
logs/gnn_optimization_final_report.md
logs/gnn_optimization_results.md
logs/graphsage_results.md
logs/Module_A_Execution_Log.md
logs/Module_B_Execution_Log.md
logs/Module_C_Execution_Log.md
logs/Module_D_Execution_Log.md
logs/Optimization_Phase_Log.md
logs/results.md
logs/results_merged.md
Master_Project_Plan.md
models/gcn_merged.md
Osteogenesis imperfecta/results/baseline_results.json
Osteogenesis imperfecta/results/expanded_5fold_summary.md
Osteogenesis imperfecta/results/expanded_5fold_tuning_results.json
Osteogenesis imperfecta/results/expanded_data_inventory.md
Osteogenesis imperfecta/results/gnn_results.json
Osteogenesis imperfecta/results/human_grouped5_l2_lr_tuning.json
Osteogenesis imperfecta/results/human_grouped5_l2_lr_tuning_summary.md
Osteogenesis imperfecta/results/human_grouped5_optimized_lr.json
Osteogenesis imperfecta/results/human_grouped5_optimized_lr_summary.md
Osteogenesis imperfecta/results/human_grouped5_results.json
Osteogenesis imperfecta/results/human_grouped5_summary.md
Osteogenesis imperfecta/results/real_data_inventory.md
Osteogenesis imperfecta/results/real_world_results.json
Osteogenesis imperfecta/results/real_world_summary.md
Osteogenesis imperfecta/results/summary.md
Osteogenesis imperfecta/scripts/00_download_data.py
Osteogenesis imperfecta/scripts/01_prepare_expression.py
Osteogenesis imperfecta/scripts/02_combat_correction.py
Osteogenesis imperfecta/scripts/03_build_graphs.py
Osteogenesis imperfecta/scripts/04_baselines.py
Osteogenesis imperfecta/scripts/05_train_gnn.py
Osteogenesis imperfecta/scripts/06_summarize_results.py
Osteogenesis imperfecta/scripts/07_prepare_multicohort_real.py
Osteogenesis imperfecta/scripts/08_run_real_external_eval.py
Osteogenesis imperfecta/scripts/09_prepare_expanded_real.py
Osteogenesis imperfecta/scripts/10_tune_5fold_combined.py
Osteogenesis imperfecta/scripts/11_human_grouped5_eval.py
Osteogenesis imperfecta/scripts/12_human_grouped5_tune_fast.py
Osteogenesis imperfecta/scripts/13_human_grouped5_lr_only_tuning.py
Osteogenesis imperfecta/scripts/14_human_grouped5_l2_lr_tuning.py
PROJECT_SUMMARY.md
Project-Material-Guidelines-acsef.txt
requirements.txt
Sepsis_GNN_V2/data/processed/final_genes.txt
Sepsis_GNN_V2/data/processed/top_genes.txt
Sepsis_GNN_V2/results/baseline_results.json
Sepsis_GNN_V2/results/gnn_results.json
Sepsis_GNN_V2/scripts/01_combat_correction.py
Sepsis_GNN_V2/scripts/02_build_graphs.py
Sepsis_GNN_V2/scripts/03_baselines.py
Sepsis_GNN_V2/scripts/04_train_gnn.py
Sepsis_GNN_V2/scripts/05_external_validation.py
Sepsis_GNN_V2/scripts/06_explainability.py
verify_env.py
```

### Manifest Interpretation
The manifest confirms a repository-scale overhaul rather than a narrow patch. Expansion dominates via new datasets, models, curated visuals, and compiled results. Retirements are concentrated in split notebook locations and superseded helper scripts. Functional code deltas are intentionally limited and centered on leakage-safe fold preprocessing and documentation/path consistency.


## Visual Placeholders and Exact PNG Files to Upload
Use these exact files for the placeholder slots.
### Visual Slot 01
- **Upload this PNG:** `General_Sepsis_V11/results/plots/roc_cv_model_comparison.png`
- **Placeholder:** `![PLACEHOLDER SLOT 01 - roc_cv_model_comparison.png](PLACEHOLDER_SLOT_01.png)`
### Visual Slot 02
- **Upload this PNG:** `General_Sepsis_V11/results/plots/roc_external_model_comparison.png`
- **Placeholder:** `![PLACEHOLDER SLOT 02 - roc_external_model_comparison.png](PLACEHOLDER_SLOT_02.png)`
### Visual Slot 03
- **Upload this PNG:** `General_Sepsis_V11/results/plots/pr_cv_model_comparison.png`
- **Placeholder:** `![PLACEHOLDER SLOT 03 - pr_cv_model_comparison.png](PLACEHOLDER_SLOT_03.png)`
### Visual Slot 04
- **Upload this PNG:** `General_Sepsis_V11/results/plots/pr_external_holdout.png`
- **Placeholder:** `![PLACEHOLDER SLOT 04 - pr_external_holdout.png](PLACEHOLDER_SLOT_04.png)`
### Visual Slot 05
- **Upload this PNG:** `General_Sepsis_V11/results/plots/metrics_heatmap_cv.png`
- **Placeholder:** `![PLACEHOLDER SLOT 05 - metrics_heatmap_cv.png](PLACEHOLDER_SLOT_05.png)`
### Visual Slot 06
- **Upload this PNG:** `General_Sepsis_V11/results/plots/metrics_heatmap_external.png`
- **Placeholder:** `![PLACEHOLDER SLOT 06 - metrics_heatmap_external.png](PLACEHOLDER_SLOT_06.png)`
### Visual Slot 07
- **Upload this PNG:** `General_Sepsis_V11/results/plots/auroc_heatmap_by_dataset_cv.png`
- **Placeholder:** `![PLACEHOLDER SLOT 07 - auroc_heatmap_by_dataset_cv.png](PLACEHOLDER_SLOT_07.png)`
### Visual Slot 08
- **Upload this PNG:** `General_Sepsis_V11/results/plots/relation_attention_heatmap.png`
- **Placeholder:** `![PLACEHOLDER SLOT 08 - relation_attention_heatmap.png](PLACEHOLDER_SLOT_08.png)`
### Visual Slot 09
- **Upload this PNG:** `General_Sepsis_V11/results/plots/relation_attention_external.png`
- **Placeholder:** `![PLACEHOLDER SLOT 09 - relation_attention_external.png](PLACEHOLDER_SLOT_09.png)`
### Visual Slot 10
- **Upload this PNG:** `General_Sepsis_V11/results/plots/gnn_topology_3d.png`
- **Placeholder:** `![PLACEHOLDER SLOT 10 - gnn_topology_3d.png](PLACEHOLDER_SLOT_10.png)`
### Visual Slot 11
- **Upload this PNG:** `General_Sepsis_V11/results/plots/shap_summary_top20.png`
- **Placeholder:** `![PLACEHOLDER SLOT 11 - shap_summary_top20.png](PLACEHOLDER_SLOT_11.png)`
### Visual Slot 12
- **Upload this PNG:** `General_Sepsis_V11/results/plots/shap_heatmap_top20.png`
- **Placeholder:** `![PLACEHOLDER SLOT 12 - shap_heatmap_top20.png](PLACEHOLDER_SLOT_12.png)`
### Visual Slot 13
- **Upload this PNG:** `General_Sepsis_V11/results/pca_by_dataset.png`
- **Placeholder:** `![PLACEHOLDER SLOT 13 - pca_by_dataset.png](PLACEHOLDER_SLOT_13.png)`
### Visual Slot 14
- **Upload this PNG:** `General_Sepsis_V11/results/pca_by_condition.png`
- **Placeholder:** `![PLACEHOLDER SLOT 14 - pca_by_condition.png](PLACEHOLDER_SLOT_14.png)`
### Visual Slot 15
- **Upload this PNG:** `CH_DANN_Plan/results/v11_gnn_topology_visual.png`
- **Placeholder:** `![PLACEHOLDER SLOT 15 - v11_gnn_topology_visual.png](PLACEHOLDER_SLOT_15.png)`
### Visual Slot 16
- **Upload this PNG:** `CH_DANN_Plan/results/v11_biomarkers_barplot.png`
- **Placeholder:** `![PLACEHOLDER SLOT 16 - v11_biomarkers_barplot.png](PLACEHOLDER_SLOT_16.png)`
### Visual Slot 17
- **Upload this PNG:** `CH_DANN_Plan/results/publication_ready/figure_model_lineage_neonatal.png`
- **Placeholder:** `![PLACEHOLDER SLOT 17 - figure_model_lineage_neonatal.png](PLACEHOLDER_SLOT_17.png)`
### Visual Slot 18
- **Upload this PNG:** `CH_DANN_Plan/results/publication_ready/figure_loco_cohort_performance.png`
- **Placeholder:** `![PLACEHOLDER SLOT 18 - figure_loco_cohort_performance.png](PLACEHOLDER_SLOT_18.png)`
### Visual Slot 19
- **Upload this PNG:** `CH_DANN_Plan/results/publication_ready/figure_seed_stability.png`
- **Placeholder:** `![PLACEHOLDER SLOT 19 - figure_seed_stability.png](PLACEHOLDER_SLOT_19.png)`
### Visual Slot 20
- **Upload this PNG:** `ACSEF_Final_Submission/figures/fig_general_sepsis_v11_architecture.png`
- **Placeholder:** `![PLACEHOLDER SLOT 20 - fig_general_sepsis_v11_architecture.png](PLACEHOLDER_SLOT_20.png)`
### Visual Slot 21
- **Upload this PNG:** `ACSEF_Final_Submission/figures/fig_general_sepsis_v11_data_biomarkers.png`
- **Placeholder:** `![PLACEHOLDER SLOT 21 - fig_general_sepsis_v11_data_biomarkers.png](PLACEHOLDER_SLOT_21.png)`
### Visual Slot 22
- **Upload this PNG:** `ACSEF_Final_Submission/figures/fig_general_sepsis_v11_performance_panels.png`
- **Placeholder:** `![PLACEHOLDER SLOT 22 - fig_general_sepsis_v11_performance_panels.png](PLACEHOLDER_SLOT_22.png)`
### Visual Slot 23
- **Upload this PNG:** `ACSEF_Final_Submission/figures/fig_external_validation_gse26440.png`
- **Placeholder:** `![PLACEHOLDER SLOT 23 - fig_external_validation_gse26440.png](PLACEHOLDER_SLOT_23.png)`
### Visual Slot 24
- **Upload this PNG:** `ACSEF_Final_Submission/figures/fig_biomarker_attributions.png`
- **Placeholder:** `![PLACEHOLDER SLOT 24 - fig_biomarker_attributions.png](PLACEHOLDER_SLOT_24.png)`
### Visual Slot 25
- **Upload this PNG:** `ACSEF_Final_Submission/figures/fig_biomarker_correlation_heatmap.png`
- **Placeholder:** `![PLACEHOLDER SLOT 25 - fig_biomarker_correlation_heatmap.png](PLACEHOLDER_SLOT_25.png)`
### Visual Slot 26
- **Upload this PNG:** `ACSEF_Final_Submission/figures/fig_roc_comparisons.png`
- **Placeholder:** `![PLACEHOLDER SLOT 26 - fig_roc_comparisons.png](PLACEHOLDER_SLOT_26.png)`
### Visual Slot 27
- **Upload this PNG:** `ACSEF_Final_Submission/figures/fig_top_biomarker_pca.png`
- **Placeholder:** `![PLACEHOLDER SLOT 27 - fig_top_biomarker_pca.png](PLACEHOLDER_SLOT_27.png)`
### Visual Slot 28
- **Upload this PNG:** `Osteogenesis imperfecta/figures/expanded_5fold_accuracy.png`
- **Placeholder:** `![PLACEHOLDER SLOT 28 - expanded_5fold_accuracy.png](PLACEHOLDER_SLOT_28.png)`
### Visual Slot 29
- **Upload this PNG:** `Osteogenesis imperfecta/figures/human_grouped5_accuracy.png`
- **Placeholder:** `![PLACEHOLDER SLOT 29 - human_grouped5_accuracy.png](PLACEHOLDER_SLOT_29.png)`
### Visual Slot 30
- **Upload this PNG:** `Osteogenesis imperfecta/figures/human_grouped5_optimized_lr_roc.png`
- **Placeholder:** `![PLACEHOLDER SLOT 30 - human_grouped5_optimized_lr_roc.png](PLACEHOLDER_SLOT_30.png)`
### Visual Slot 31
- **Upload this PNG:** `ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_poster.png`
- **Placeholder:** `![PLACEHOLDER SLOT 31 - general_sepsis_v11_poster.png](PLACEHOLDER_SLOT_31.png)`
### Visual Slot 32
- **Upload this PNG:** `ACSEF_Final_Submission/figures/fig_acsef_poster_layout_draft.png`
- **Placeholder:** `![PLACEHOLDER SLOT 32 - fig_acsef_poster_layout_draft.png](PLACEHOLDER_SLOT_32.png)`
### Visual Slot 33
- **Upload this PNG:** `ACSEF_Final_Submission/figures/fig_architecture_flowchart.png`
- **Placeholder:** `![PLACEHOLDER SLOT 33 - fig_architecture_flowchart.png](PLACEHOLDER_SLOT_33.png)`
### Visual Slot 34
- **Upload this PNG:** `ACSEF_Final_Submission/figures/fig_general_sepsis_v11_impact_infographic.png`
- **Placeholder:** `![PLACEHOLDER SLOT 34 - fig_general_sepsis_v11_impact_infographic.png](PLACEHOLDER_SLOT_34.png)`
### Visual Slot 35
- **Upload this PNG:** `ACSEF_Final_Submission/figures/fig_general_sepsis_v11_literature_comparison.png`
- **Placeholder:** `![PLACEHOLDER SLOT 35 - fig_general_sepsis_v11_literature_comparison.png](PLACEHOLDER_SLOT_35.png)`
### Visual Slot 36
- **Upload this PNG:** `ACSEF_Final_Submission/figures/fig_model_metric_radar.png`
- **Placeholder:** `![PLACEHOLDER SLOT 36 - fig_model_metric_radar.png](PLACEHOLDER_SLOT_36.png)`

## File-Type Distribution
- `.png`: 95
- `.md`: 80
- `.py`: 80
- `.pt`: 68
- `.csv`: 58
- `.json`: 54
- `.txt`: 38
- `.gz`: 34
- `.html`: 6
- `.log`: 6
- `.pdf`: 6
- `.pkl`: 2
- `.svg`: 2
- `.xlsx`: 2
- `.tex`: 1
- `.zip`: 1
- `[no_ext]`: 1

## Active Top-Level Folder Summary
- `.gitignore`: 1 files
- `01_id_mapping.py`: 1 files
- `02_merge_combat.py`: 1 files
- `03_graph_construction.py`: 1 files
- `04_create_graphs.py`: 1 files
- `04_create_graphs_enhanced.py`: 1 files
- `04_create_graphs_expanded.py`: 1 files
- `04_create_graphs_variance_filtered.py`: 1 files
- `05_baseline_models.py`: 1 files
- `05_train_baselines.py`: 1 files
- `05_train_baselines_varying_features.py`: 1 files
- `06_train_gnn.py`: 1 files
- `06_train_gnn_optimized.py`: 1 files
- `06a_train_gcn.py`: 1 files
- `06b_train_gat.py`: 1 files
- `06c_train_gat_expanded.py`: 1 files
- `07_verification_sanity_check.py`: 1 files
- `09_data_rescue_expansion.py`: 1 files
- `ACSEF_Final_Submission`: 71 files
- `ALZHEIMERS_STRATEGIC_PATHWAY`: 71 files
- `CH_DANN_Plan`: 101 files
- `COMPREHENSIVE_PROJECT_SUMMARY.md`: 1 files
- `ENGINEERING_NOTEBOOK_MASTER.md`: 1 files
- `General_Sepsis_V11`: 62 files
- `Master_Project_Plan.md`: 1 files
- `Osteogenesis imperfecta`: 93 files
- `PROJECT_SUMMARY.md`: 1 files
- `Project-Material-Guidelines-acsef.txt`: 1 files
- `Sepsis_GNN_V2`: 12 files
- `data`: 7 files
- `docs`: 32 files
- `download_data.py`: 1 files
- `download_data_from_urls.py`: 1 files
- `download_data_with_gseapy.py`: 1 files
- `download_geoparse.py`: 1 files
- `download_remaining.py`: 1 files
- `draw_general_sepsis_topology.py`: 1 files
- `figures`: 2 files
- `gnn_optimized`: 3 files
- `investigate_gse26440_age.py`: 1 files
- `logs`: 12 files
- `models`: 1 files
- `requirements.txt`: 1 files
- `verify_env.py`: 1 files

## Core File Walkthrough
- `General_Sepsis_V11/scripts/01_download_and_preprocess.py`: This is the gatekeeper for data legitimacy in the current sepsis rebuild. It downloads or reuses cached GEO SOFT files, applies dataset-specific sample parsing rules, enforces the healthy-vs-sepsis-only policy, triggers the deterministic GSE134347 fallback when GSE95233 admission parsing fails QC, performs train-only ComBat harmonization, selects the top MAD genes, and exports the exact expression and metadata tables consumed by every later step. If this file is wrong, every downstream metric becomes untrustworthy because cohort composition, gene set membership, and normalization parameters all originate here.
- `General_Sepsis_V11/scripts/02_build_graphs.py`: This file converts the selected gene list into graph priors without leaking holdout information. It queries KEGG for pathway hyperedges, maps gene symbols to STRING protein identifiers, streams the STRING edge file with a score threshold, computes relation coverage diagnostics, and explicitly documents that co-expression is not persisted globally because it must be rebuilt from fold-train samples only. In practice, this script defines the biological topology contract that the V11 architecture is allowed to use.
- `General_Sepsis_V11/scripts/03_train_v11_general_sepsis.py`: This is the core training script for the current architecture. It builds either LODO or SGKF folds, performs fold-internal MAD feature selection and normalization, rebuilds co-expression on the training partition only, instantiates the multiplex hypergraph plus DANN model, tracks relation attention over epochs, saves per-fold checkpoints, and emits the raw cross-validation payload used for later audit. The important engineering point is that the file keeps feature selection, graph construction, and normalization inside each fold so the resulting OOF predictions are defensible.
- `General_Sepsis_V11/scripts/04_evaluate.py`: This file is the formal comparison and validation layer. It reads the saved OOF predictions, scores the hybrid model, rebuilds tabular baselines on the identical folds, performs paired permutation testing, runs external holdout inference with the saved checkpoint configuration, checks leakage conditions, and writes the machine-readable result bundles that the poster and publication assets depend on. It is where “does the architecture really beat its baselines under the locked protocol?” is answered in audit-ready form.
- `General_Sepsis_V11/scripts/05_build_master_notebook.py`: This script generates the ACSEF TeX/PDF engineering notebook artifact. It crawls the active code folders, parses imports and function definitions with `ast`, folds in recent git/log history, injects mathematical sections, and appends the overhaul execution trace. Conceptually, this file is not part of model training; it is the reproducibility compiler that turns a working repository into a submission-grade notebook package.
- `General_Sepsis_V11/scripts/06_metrics_and_plots.py`: This is the visualization and reviewer-facing reporting stage for the robust sepsis branch. It converts JSON/CSV outputs into ROC and PR overlays, dataset and platform heatmaps, relation-attention summaries, a 3D topology render, and the SHAP-based reference analysis. It also writes `metrics_report.md`, so many high-level claims in the docs and poster are downstream of this one file.
- `General_Sepsis_V11/results/cohort_manifest.json`: This JSON is the provenance ledger for the rebuilt sepsis experiment. It records which datasets were active, whether fallback logic triggered, how many samples and genes survived policy filtering, and which artifact paths were exported. Judges or collaborators can use it to verify that the reported cohort policy matches the actual run.
- `General_Sepsis_V11/results/cv_metrics_raw.json`: This is the most important machine-readable training artifact in the branch. It stores fold definitions, train/validation sample IDs, selected genes, normalization statistics, epoch traces, OOF predictions, and checkpoint paths. Any attempt to reproduce the plots or challenge leakage claims should start here.
- `General_Sepsis_V11/results/baseline_comparison.json`: This file stores the baseline model predictions and pooled metrics under the same split protocol used by the hybrid model. It matters because the visible charts are only summaries; this JSON contains the exact evidence used to say whether logistic, MLP, or linear ablations were competitive or not.
- `General_Sepsis_V11/results/general_sepsis_v11_results.json`: This is the top-level results contract for the current sepsis architecture. It aggregates CV metrics, bootstrap confidence intervals, external holdout results, baseline summaries, permutation-test outputs, and hard-pass gates. Most downstream documents read from this file, directly or indirectly, rather than recomputing results.
- `CH_DANN_Plan/scripts/19_train_v11_multiplex_dann.py`: This is the lineage script that precedes the current generalized rebuild. It captures the earlier V11 multiplex DANN design decisions and serves as the architectural bridge between the neonatal sepsis-only branch and the more disciplined General_Sepsis_V11 workflow. When explaining why the current design exists, this file is part of the answer.
- `CH_DANN_Plan/scripts/22_train_v11_alzheimers_transfer.py`: This file demonstrates that the graph-guided and domain-adversarial ideas were not confined to one disease. It adapts the V11-style machinery to Alzheimer’s transfer experiments, which is why the project can make a broader platform claim rather than a single-dataset sepsis claim.
- `Osteogenesis imperfecta/scripts/04_baselines.py`: This is the rare-disease tabular benchmark stage. It provides the strongest non-graph comparison point in the osteogenesis branch and helps frame whether graph-based modeling remains useful when sample counts are small and cohort heterogeneity is high.
- `Osteogenesis imperfecta/scripts/05_train_gnn.py`: This is the disease-transfer counterpart to the sepsis training code. It tests whether graph-guided classification survives domain shift in a rare-disease setting, which is why the notebook treats it as supporting evidence for generality rather than as an unrelated side project.
- `ACSEF_Final_Submission/scripts/02_compile_metrics_and_make_figures.py`: This is the packaging bridge from experiment outputs to submission visuals. It reads archived and current result JSONs, standardizes naming, assembles compiled metrics, and generates the figures and justification documents used in the ACSEF folder. If a number appears on a poster or summary sheet, this script is one of the first places to audit.
- `ACSEF_Final_Submission/scripts/05_generate_general_sepsis_v11_publication.py`: This file takes the robust sepsis outputs and turns them into synced figures, poster exports, figure manifests, and claim-traceability tables. It is the final publication assembly step, not a modeling step, and it exists so the submission package can be regenerated from source rather than manually edited.
- `ENGINEERING_NOTEBOOK_MASTER.md`: The root notebook is the human-readable control plane for the repository. Its role is to unify scientific rationale, code inventory, artifact mapping, and project-governance notes so the repository can be reviewed as a coherent engineering system instead of a pile of disconnected experiment folders.

## Complete File-by-File Inventory and Rationale
### Group: `.gitignore`
- `.gitignore`: Project evidence artifact retained to preserve the full engineering history and reproducibility chain.
### Group: `01_id_mapping.py`
- `01_id_mapping.py`: Early preprocessing utility that aligns GEO identifiers across downloaded tables so later merge steps can join expression and metadata without manual spreadsheet cleanup.
### Group: `02_merge_combat.py`
- `02_merge_combat.py`: Legacy harmonization step that merges cohort-level matrices and applies ComBat correction, representing the older baseline lineage before the stricter General_Sepsis_V11 rebuild.
### Group: `03_graph_construction.py`
- `03_graph_construction.py`: Root-level graph builder from the earlier sepsis pipeline; it established the first pass at topology generation before the fold-safe graph logic was moved into the newer branch.
### Group: `04_create_graphs.py`
- `04_create_graphs.py`: Baseline graph-export script retained as provenance for how patient-level graphs were originally assembled from expression and biological priors.
### Group: `04_create_graphs_enhanced.py`
- `04_create_graphs_enhanced.py`: Variant graph builder that experiments with a richer topology construction strategy and therefore documents the intermediate design space before the current multiplex formulation stabilized.
### Group: `04_create_graphs_expanded.py`
- `04_create_graphs_expanded.py`: Expansion-oriented graph construction path used when the project explored larger feature sets and broader cohort inclusion.
### Group: `04_create_graphs_variance_filtered.py`
- `04_create_graphs_variance_filtered.py`: Graph builder that couples topology generation to variance filtering, providing a direct historical record of how feature pruning affected graph size and sparsity.
### Group: `05_baseline_models.py`
- `05_baseline_models.py`: Historical flat-expression benchmark runner for tabular classifiers, useful for tracing how early RF/LR baselines were established before the robust V11 comparison stack.
### Group: `05_train_baselines.py`
- `05_train_baselines.py`: Legacy baseline-training script that records the project’s earlier 5-fold comparison protocol and serves as a contrast against the later fold-safe evaluation design.
### Group: `05_train_baselines_varying_features.py`
- `05_train_baselines_varying_features.py`: Baseline sweep utility used to test how tabular performance changed as the feature budget expanded or contracted.
### Group: `06_train_gnn.py`
- `06_train_gnn.py`: Early graph neural network training entrypoint retained to show the first generation of model training before the project adopted multiplex hypergraphs and adversarial domain alignment.
### Group: `06_train_gnn_optimized.py`
- `06_train_gnn_optimized.py`: Optimization-oriented successor to the original GNN trainer, capturing the first round of hyperparameter and training-loop refinement.
### Group: `06a_train_gcn.py`
- `06a_train_gcn.py`: Dedicated GCN baseline trainer preserved so architecture evolution can be compared against a plain graph-convolution control.
### Group: `06b_train_gat.py`
- `06b_train_gat.py`: Dedicated GAT baseline trainer that captures the attention-based graph baseline prior to the final multiplex design.
### Group: `06c_train_gat_expanded.py`
- `06c_train_gat_expanded.py`: Expanded GAT training variant used during broader architecture sweeps and retained for model-lineage evidence.
### Group: `07_verification_sanity_check.py`
- `07_verification_sanity_check.py`: Sanity-check script that verifies basic pipeline assumptions and protects against silent corruption of intermediate artifacts.
### Group: `09_data_rescue_expansion.py`
- `09_data_rescue_expansion.py`: Recovery utility used when additional cohorts or missing artifacts had to be reintroduced without rebuilding the entire repository from scratch.
### Group: `ACSEF_Final_Submission`
- `ACSEF_Final_Submission/acsef_documents/acsef_official_abstract.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ACSEF_Final_Submission/acsef_documents/acsef_project_presentation.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `ACSEF_Final_Submission/acsef_documents/acsef_quad_chart_content.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `ACSEF_Final_Submission/acsef_documents/engineering_notebook_master.pdf`: Submission or publication artifact used for formal dissemination and print-ready packaging.
- `ACSEF_Final_Submission/acsef_documents/engineering_notebook_master.tex`: Submission or publication artifact used for formal dissemination and print-ready packaging.
- `ACSEF_Final_Submission/acsef_documents/final_validation_checklist.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `ACSEF_Final_Submission/acsef_documents/model_justification_and_architecture.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `ACSEF_Final_Submission/acsef_documents/publication_package/acsef_poster_layout_draft.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `ACSEF_Final_Submission/acsef_documents/publication_package/acsef_poster_layout_draft.pdf`: Submission or publication artifact used for formal dissemination and print-ready packaging.
- `ACSEF_Final_Submission/acsef_documents/publication_package/acsef_poster_layout_draft.svg`: Submission or publication artifact used for formal dissemination and print-ready packaging.
- `ACSEF_Final_Submission/acsef_documents/publication_package/figure_manifest.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_claim_traceability.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_figure_manifest.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_poster.pdf`: Submission or publication artifact used for formal dissemination and print-ready packaging.
- `ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_poster.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/acsef_documents/publication_package/general_sepsis_v11_poster.svg`: Submission or publication artifact used for formal dissemination and print-ready packaging.
- `ACSEF_Final_Submission/acsef_documents/publication_package/README.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `ACSEF_Final_Submission/acsef_documents/publication_package/table_external_validation.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `ACSEF_Final_Submission/acsef_documents/publication_package/table_main_metrics.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `ACSEF_Final_Submission/acsef_documents/rare_disease_scaling_osteogenesis.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `ACSEF_Final_Submission/data/expression_top2000.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `ACSEF_Final_Submission/data/metadata_aligned.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `ACSEF_Final_Submission/data/top2000_gene_list.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ACSEF_Final_Submission/figures/fig_acsef_poster_layout_draft.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/figures/fig_architecture_flowchart.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/figures/fig_biomarker_attributions.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/figures/fig_biomarker_correlation_heatmap.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/figures/fig_external_validation_gse26440.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/figures/fig_general_sepsis_v11_architecture.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/figures/fig_general_sepsis_v11_data_biomarkers.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/figures/fig_general_sepsis_v11_impact_infographic.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/figures/fig_general_sepsis_v11_literature_comparison.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/figures/fig_general_sepsis_v11_performance_panels.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/figures/fig_interactive_model_metrics.html`: Interactive visualization export for reviewer-side exploration of results.
- `ACSEF_Final_Submission/figures/fig_interactive_top20_biomarkers.html`: Interactive visualization export for reviewer-side exploration of results.
- `ACSEF_Final_Submission/figures/fig_model_metric_radar.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/figures/fig_normalization_distributions.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/figures/fig_osteogenesis_scaling_summary.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/figures/fig_relation_attention_heatmap.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/figures/fig_roc_comparisons.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/figures/fig_top_biomarker_pca.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/images/acsef_poster_layout_draft.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/images/fig_acsef_poster_layout_draft.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/images/fig_architecture_flowchart.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/images/fig_biomarker_attributions.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/images/fig_biomarker_correlation_heatmap.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/images/fig_external_validation_gse26440.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/images/fig_general_sepsis_v11_architecture.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/images/fig_general_sepsis_v11_data_biomarkers.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/images/fig_general_sepsis_v11_performance_panels.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/images/fig_model_metric_radar.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/images/fig_normalization_distributions.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/images/fig_osteogenesis_scaling_summary.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/images/fig_relation_attention_heatmap.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/images/fig_roc_comparisons.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/images/fig_top_biomarker_pca.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `ACSEF_Final_Submission/images/README.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `ACSEF_Final_Submission/logs/execution_log_2026-02-24.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `ACSEF_Final_Submission/logs/failure_analysis_log.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `ACSEF_Final_Submission/logs/xai_pipeline_2026-02-24.log`: Project evidence artifact retained to preserve the full engineering history and reproducibility chain.
- `ACSEF_Final_Submission/models/multiplex_hyper_dann_mlp_acsef.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ACSEF_Final_Submission/results/all_gene_attributions.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `ACSEF_Final_Submission/results/compiled_model_metrics.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `ACSEF_Final_Submission/results/relation_attention_distribution.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `ACSEF_Final_Submission/results/top_100_biomarkers.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `ACSEF_Final_Submission/results/xai_training_metrics.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `ACSEF_Final_Submission/scripts/01_run_robust_xai_biomarkers.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `ACSEF_Final_Submission/scripts/02_compile_metrics_and_make_figures.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `ACSEF_Final_Submission/scripts/03_build_publication_assets.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `ACSEF_Final_Submission/scripts/04_generate_print_poster_draft.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `ACSEF_Final_Submission/scripts/05_generate_general_sepsis_v11_publication.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
### Group: `ALZHEIMERS_STRATEGIC_PATHWAY`
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_expanded_ad_1000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_expanded_ad_2000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_expanded_ad_3000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_expanded_ad_5000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_1000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_1500.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_2000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_2500.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_3000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_500.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_5000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_breakthrough.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_enriched.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_augmented_1000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_expanded_1000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_expanded_2000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_expanded_3000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_fixed_1000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_10000_1000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_10000_2000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_10000_3000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_3000_1000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_3000_2000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_3000_3000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_5000_1000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_5000_2000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_5000_3000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_8000_1000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_8000_2000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_synthetic_8000_3000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_1000.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_2000.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_3000.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_expanded_1000.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_expanded_2000.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_expanded_3000.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_expanded_5000.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_expanded_all.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_real_expanded_1000.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_real_expanded_2000.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_real_expanded_3000.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_synthetic_10000.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_synthetic_3000.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_synthetic_5000.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/gene_list_synthetic_8000.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw/9606.protein.links.v12.0.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw/9606.protein.links.v12.0.txt.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw/GPL10558.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw/GSE63060_series_matrix.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw/GSE63060_series_matrix.txt.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GPL10558.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GPL1211.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GPL16699.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GPL570.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GPL6947.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GPL96.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE122063_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE1297_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE28146_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE4226_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE48350_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE5281_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE63060_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE63061_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/raw_expanded/GSE97760_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `ALZHEIMERS_STRATEGIC_PATHWAY/src/data/adni/processed/dataset_mci_conversion_1000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/src/data/adni/processed/dataset_mci_conversion_500.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `ALZHEIMERS_STRATEGIC_PATHWAY/src/data/adni/processed/gene_list_mci_conversion_1000.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/src/data/adni/raw_mci_conversion/GPL21263.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `ALZHEIMERS_STRATEGIC_PATHWAY/src/data/adni/raw_mci_conversion/GSE150693_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
### Group: `CH_DANN_Plan`
- `CH_DANN_Plan/CH_DANN_PROJECT_PLAN.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `CH_DANN_Plan/data/alz/alz_blood_true_domains_2000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/data/alz/alz_blood_true_domains_expression_top2000.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `CH_DANN_Plan/data/alz/alz_blood_true_domains_metadata_top2000.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `CH_DANN_Plan/data/alz/alz_brain_true_domains_2000.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/data/alz/alz_brain_true_domains_expression_top2000.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `CH_DANN_Plan/data/alz/alz_brain_true_domains_metadata_top2000.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `CH_DANN_Plan/data/alz/gene_list_2000.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `CH_DANN_Plan/model_architecture.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `CH_DANN_Plan/models/hgcn_v2_best.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/hgcn_v2_fold1.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/hgcn_v2_fold2.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/hgcn_v2_fold3.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/hgcn_v2_fold4.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/hgcn_v2_fold5.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_brain_loco_best.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_brain_loco_GSE1297.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_brain_loco_GSE28146.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_brain_loco_GSE5281.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_brain_seed_123_best.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_brain_seed_21_best.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_brain_seed_42_best.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_brain_seed_77_best.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_brain_seed_7_best.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_brain_true_domains_best.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_brain_true_domains_nodann_best.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_transfer_best.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_transfer_fold_1.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_transfer_fold_2.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_transfer_fold_3.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_transfer_fold_4.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/models/v11_alz_transfer_fold_5.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `CH_DANN_Plan/results/a1_summary.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/a1_v2_results.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `CH_DANN_Plan/results/a1_v2_summary.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/expression_combat_v2.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `CH_DANN_Plan/results/gene_list.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/gene_list_v2.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/metadata_v2.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `CH_DANN_Plan/results/pathway_info.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/pathway_info_v2.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/publication_ready/figure_loco_cohort_performance.pdf`: Submission or publication artifact used for formal dissemination and print-ready packaging.
- `CH_DANN_Plan/results/publication_ready/figure_loco_cohort_performance.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `CH_DANN_Plan/results/publication_ready/figure_model_lineage_neonatal.pdf`: Submission or publication artifact used for formal dissemination and print-ready packaging.
- `CH_DANN_Plan/results/publication_ready/figure_model_lineage_neonatal.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `CH_DANN_Plan/results/publication_ready/figure_seed_stability.pdf`: Submission or publication artifact used for formal dissemination and print-ready packaging.
- `CH_DANN_Plan/results/publication_ready/figure_seed_stability.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `CH_DANN_Plan/results/publication_ready/manuscript_draft.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `CH_DANN_Plan/results/publication_ready/publication_manifest.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/publication_ready/table_loco_per_cohort.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `CH_DANN_Plan/results/publication_ready/table_seed_stability_per_seed.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `CH_DANN_Plan/results/publication_ready/table_summary_main.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `CH_DANN_Plan/results/publication_ready/table_summary_main.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `CH_DANN_Plan/results/seed_stability/dann/v11_alz_brain_seed_123_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/seed_stability/dann/v11_alz_brain_seed_21_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/seed_stability/dann/v11_alz_brain_seed_42_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/seed_stability/dann/v11_alz_brain_seed_77_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/seed_stability/dann/v11_alz_brain_seed_7_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/seed_stability/nodann/v11_alz_brain_seed_123_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/seed_stability/nodann/v11_alz_brain_seed_21_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/seed_stability/nodann/v11_alz_brain_seed_42_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/seed_stability/nodann/v11_alz_brain_seed_77_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/seed_stability/nodann/v11_alz_brain_seed_7_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/v10_multiplex_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/v11_alz_blood_true_domains_static_dann_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/v11_alz_blood_true_domains_static_nodann_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/v11_alz_brain_loco_dann_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/v11_alz_brain_loco_nodann_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/v11_alz_brain_seed_stability_dann_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/v11_alz_brain_seed_stability_nodann_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/v11_alz_brain_true_domains_nodann_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/v11_alz_brain_true_domains_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/v11_alz_brain_true_domains_static_nodann_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/v11_alz_transfer_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/v11_biomarkers_barplot.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `CH_DANN_Plan/results/v11_gnn_topology_visual.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `CH_DANN_Plan/results/v11_gse26440_external_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/v11_multiplex_dann_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/v7_sgkf_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/v8_guided_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/results/v9_residual_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `CH_DANN_Plan/scripts/10_rebuild_and_train_a1.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/10_train_hgcn_a1.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/11_train_hgcn_v3.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/12_train_hybrid_v4.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/13_train_v5_lobo_dann.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/14_train_v6_simple_split.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/15_train_v7_fixed_cv.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/16_train_v8_gnn_guided.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/17_train_v9_residual_fusion.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/18_train_v10_multiplex.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/19_train_v11_multiplex_dann.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/20_evaluate_v11_gse26440.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/21_train_v12_pure_hgcn.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/22_explain_v11.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/22_train_v11_alzheimers_transfer.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/23_prepare_alz_true_domains.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/24_evaluate_alz_brain_loco.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/25_evaluate_alz_brain_seed_stability.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/scripts/26_build_publication_summary.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `CH_DANN_Plan/V11_Multiplex_DANN_Final_Report.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
### Group: `COMPREHENSIVE_PROJECT_SUMMARY.md`
- `COMPREHENSIVE_PROJECT_SUMMARY.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
### Group: `ENGINEERING_NOTEBOOK_MASTER.md`
- `ENGINEERING_NOTEBOOK_MASTER.md`: Canonical root notebook that consolidates rationale, artifact mapping, file inventory, and project-governance notes for the full workspace.
### Group: `General_Sepsis_V11`
- `General_Sepsis_V11/data/raw/GSE134347_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `General_Sepsis_V11/data/raw/GSE26378_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `General_Sepsis_V11/data/raw/GSE54514_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `General_Sepsis_V11/data/raw/GSE57065_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `General_Sepsis_V11/data/raw/GSE95233_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `General_Sepsis_V11/logs/2026-03-02_04_evaluate.log`: Project evidence artifact retained to preserve the full engineering history and reproducibility chain.
- `General_Sepsis_V11/logs/2026-03-03_01_download_and_preprocess.log`: Project evidence artifact retained to preserve the full engineering history and reproducibility chain.
- `General_Sepsis_V11/logs/2026-03-03_02_build_graphs.log`: Project evidence artifact retained to preserve the full engineering history and reproducibility chain.
- `General_Sepsis_V11/logs/2026-03-03_03_train_v11_general_sepsis.log`: Project evidence artifact retained to preserve the full engineering history and reproducibility chain.
- `General_Sepsis_V11/logs/2026-03-03_04_evaluate.log`: Project evidence artifact retained to preserve the full engineering history and reproducibility chain.
- `General_Sepsis_V11/models/general_sepsis_v11_best.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `General_Sepsis_V11/models/general_sepsis_v11_fold1.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `General_Sepsis_V11/models/general_sepsis_v11_fold2.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `General_Sepsis_V11/models/general_sepsis_v11_fold3.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `General_Sepsis_V11/models/general_sepsis_v11_fold4.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `General_Sepsis_V11/models/general_sepsis_v11_fold5.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `General_Sepsis_V11/results/baseline_comparison.json`: Fold-aligned baseline prediction store used to compare the hybrid architecture against logistic, MLP, and linear ablations under the exact same validation splits.
- `General_Sepsis_V11/results/cohort_manifest.json`: Cohort provenance ledger listing selected datasets, fallback-policy outcome, sample counts, and the exported artifact paths for the active sepsis rebuild.
- `General_Sepsis_V11/results/cv_metrics_raw.json`: Raw training/evaluation payload containing fold definitions, selected genes, normalization stats, epoch logs, checkpoints, and OOF predictions.
- `General_Sepsis_V11/results/expression_combat.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `General_Sepsis_V11/results/expression_raw_selected.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `General_Sepsis_V11/results/final_package_checklist.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `General_Sepsis_V11/results/gene_list.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `General_Sepsis_V11/results/general_sepsis_v11_results.json`: Master results bundle for the V11 rebuild, combining CV metrics, external holdout performance, confidence intervals, baseline summary, permutation testing, and pass/fail gates.
- `General_Sepsis_V11/results/metadata.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `General_Sepsis_V11/results/metrics_by_dataset.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `General_Sepsis_V11/results/metrics_by_platform.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `General_Sepsis_V11/results/metrics_external.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `General_Sepsis_V11/results/metrics_overall.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `General_Sepsis_V11/results/metrics_report.md`: Human-readable summary that translates the raw JSON outputs into reviewer-friendly tables and plot references.
- `General_Sepsis_V11/results/overhaul_execution_log.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `General_Sepsis_V11/results/pathway_info.json`: Graph-prior manifest containing the retained genes, KEGG pathway memberships, STRING edges, co-expression preview, and relation-coverage QC.
- `General_Sepsis_V11/results/pca_by_condition.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/pca_by_dataset.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/auroc_by_dataset_cv_oof.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/auroc_by_platform_cv_oof.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/auroc_heatmap_by_dataset_cv.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/gnn_topology_3d.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/metrics_heatmap_cv.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/metrics_heatmap_external.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/pr_cv_model_comparison.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/pr_cv_oof.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/pr_external_holdout.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/relation_attention_by_fold.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/relation_attention_external.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/relation_attention_heatmap.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/roc_cv_model_comparison.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/roc_cv_oof.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/roc_external_holdout.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/roc_external_model_comparison.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/shap_heatmap_top20.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/shap_summary_top20.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `General_Sepsis_V11/results/plots/shap_top20_features.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `General_Sepsis_V11/results/shap_summary.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `General_Sepsis_V11/results/validation_audit_report.md`: Audit narrative for leakage checks, statistical gates, and residual risks in the robust sepsis evaluation.
- `General_Sepsis_V11/results/verification_report.md`: Separate verification narrative used to defend the integrity of metrics and validation claims for the sepsis branch.
- `General_Sepsis_V11/scripts/01_download_and_preprocess.py`: Active Step 01 pipeline entrypoint that enforces cohort policy, performs train-only ComBat, selects the gene set, and exports the canonical matrices.
- `General_Sepsis_V11/scripts/02_build_graphs.py`: Active Step 02 graph-construction stage that assembles KEGG and STRING priors while explicitly keeping co-expression fold-local.
- `General_Sepsis_V11/scripts/03_train_v11_general_sepsis.py`: Active Step 03 training stage for the multiplex hypergraph plus DANN architecture, including fold-safe feature selection and saved checkpoints.
- `General_Sepsis_V11/scripts/04_evaluate.py`: Active Step 04 evaluation stage that scores the hybrid model, rebuilds baselines on matched folds, runs permutation tests, and audits leakage.
- `General_Sepsis_V11/scripts/05_build_master_notebook.py`: Active Step 05 reproducibility/documentation stage that compiles the ACSEF TeX and PDF engineering notebook artifacts.
- `General_Sepsis_V11/scripts/06_metrics_and_plots.py`: Active Step 06 reporting stage that turns result JSONs into figures, tables, SHAP outputs, and the robust metrics report.
### Group: `Master_Project_Plan.md`
- `Master_Project_Plan.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
### Group: `Osteogenesis imperfecta`
- `Osteogenesis imperfecta/data/processed/combined_expression_log2.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/combined_metadata.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/datasets/GSE160207_expr.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/datasets/GSE160207_meta.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/datasets/GSE163812_expr.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/datasets/GSE163812_meta.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/datasets/GSE180838_expr.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/datasets/GSE180838_meta.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/datasets/GSE186141_expr.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/datasets/GSE186141_meta.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/ensp_to_gene_cache.pkl`: Project evidence artifact retained to preserve the full engineering history and reproducibility chain.
- `Osteogenesis imperfecta/data/processed/expanded_datasets/GSE154748_expr.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/expanded_datasets/GSE154748_meta.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/expanded_datasets/GSE160207_expr.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/expanded_datasets/GSE160207_meta.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/expanded_datasets/GSE163812_expr.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/expanded_datasets/GSE163812_meta.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/expanded_datasets/GSE180838_expr.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/expanded_datasets/GSE180838_meta.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/expanded_datasets/GSE186141_expr.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/expanded_datasets/GSE186141_meta.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/expanded_expression_common.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/expanded_metadata.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/expression_combat.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/final_genes.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `Osteogenesis imperfecta/data/processed/graph_metadata.pkl`: Project evidence artifact retained to preserve the full engineering history and reproducibility chain.
- `Osteogenesis imperfecta/data/processed/graphs.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `Osteogenesis imperfecta/data/processed/GSE160207_expr.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/GSE160207_meta.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/GSE163812_expr.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/GSE163812_meta.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/metadata_combat.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/multicohort_expression_common.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/multicohort_metadata.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `Osteogenesis imperfecta/data/processed/top_genes.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `Osteogenesis imperfecta/data/raw/9606.protein.links.v12.0.txt.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `Osteogenesis imperfecta/data/raw/GSE154748_ALL_FPKM.txt.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `Osteogenesis imperfecta/data/raw/GSE154748_series_matrix.txt.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `Osteogenesis imperfecta/data/raw/GSE160207_EE_OI_RNAseq_counts.txt.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `Osteogenesis imperfecta/data/raw/GSE160207_series_matrix.txt.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `Osteogenesis imperfecta/data/raw/GSE163812_ESAT_counts.txt.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `Osteogenesis imperfecta/data/raw/GSE163812_series_matrix.txt.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `Osteogenesis imperfecta/data/raw/GSE180838_FKBP10.fkpm.xlsx`: Project evidence artifact retained to preserve the full engineering history and reproducibility chain.
- `Osteogenesis imperfecta/data/raw/GSE180838_series_matrix.txt.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `Osteogenesis imperfecta/data/raw/GSE186141_FPKM9.6Col1.vs.2Ctrl.xlsx`: Project evidence artifact retained to preserve the full engineering history and reproducibility chain.
- `Osteogenesis imperfecta/data/raw/GSE186141_series_matrix.txt.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `Osteogenesis imperfecta/figures/expanded_5fold_accuracy.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Osteogenesis imperfecta/figures/human_grouped5_accuracy.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Osteogenesis imperfecta/figures/human_grouped5_l2_lr_tuning_best.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Osteogenesis imperfecta/figures/human_grouped5_optimized_lr_roc.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Osteogenesis imperfecta/figures/pca_after_combat.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Osteogenesis imperfecta/figures/pca_before_combat.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Osteogenesis imperfecta/figures/real_external_accuracy_by_holdout.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Osteogenesis imperfecta/figures/real_roc_GSE160207.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Osteogenesis imperfecta/figures/real_roc_GSE163812.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Osteogenesis imperfecta/figures/real_roc_GSE180838.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Osteogenesis imperfecta/figures/real_roc_GSE186141.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Osteogenesis imperfecta/figures/roc_gat_v2.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Osteogenesis imperfecta/figures/roc_gcn.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Osteogenesis imperfecta/figures/roc_logisticregression.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Osteogenesis imperfecta/figures/roc_randomforest.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Osteogenesis imperfecta/models/gat_v2_best.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `Osteogenesis imperfecta/models/gcn_best.pt`: Serialized model checkpoint retained for reproducibility and comparative validation.
- `Osteogenesis imperfecta/results/baseline_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `Osteogenesis imperfecta/results/expanded_5fold_summary.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `Osteogenesis imperfecta/results/expanded_5fold_tuning_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `Osteogenesis imperfecta/results/expanded_data_inventory.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `Osteogenesis imperfecta/results/gnn_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `Osteogenesis imperfecta/results/human_grouped5_l2_lr_tuning.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `Osteogenesis imperfecta/results/human_grouped5_l2_lr_tuning_summary.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `Osteogenesis imperfecta/results/human_grouped5_optimized_lr.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `Osteogenesis imperfecta/results/human_grouped5_optimized_lr_summary.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `Osteogenesis imperfecta/results/human_grouped5_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `Osteogenesis imperfecta/results/human_grouped5_summary.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `Osteogenesis imperfecta/results/real_data_inventory.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `Osteogenesis imperfecta/results/real_world_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `Osteogenesis imperfecta/results/real_world_summary.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `Osteogenesis imperfecta/results/summary.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `Osteogenesis imperfecta/scripts/00_download_data.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Osteogenesis imperfecta/scripts/01_prepare_expression.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Osteogenesis imperfecta/scripts/02_combat_correction.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Osteogenesis imperfecta/scripts/03_build_graphs.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Osteogenesis imperfecta/scripts/04_baselines.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Osteogenesis imperfecta/scripts/05_train_gnn.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Osteogenesis imperfecta/scripts/06_summarize_results.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Osteogenesis imperfecta/scripts/07_prepare_multicohort_real.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Osteogenesis imperfecta/scripts/08_run_real_external_eval.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Osteogenesis imperfecta/scripts/09_prepare_expanded_real.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Osteogenesis imperfecta/scripts/10_tune_5fold_combined.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Osteogenesis imperfecta/scripts/11_human_grouped5_eval.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Osteogenesis imperfecta/scripts/12_human_grouped5_tune_fast.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Osteogenesis imperfecta/scripts/13_human_grouped5_lr_only_tuning.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Osteogenesis imperfecta/scripts/14_human_grouped5_l2_lr_tuning.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
### Group: `PROJECT_SUMMARY.md`
- `PROJECT_SUMMARY.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
### Group: `Project-Material-Guidelines-acsef.txt`
- `Project-Material-Guidelines-acsef.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
### Group: `Sepsis_GNN_V2`
- `Sepsis_GNN_V2/data/processed/final_genes.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `Sepsis_GNN_V2/data/processed/top_genes.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
- `Sepsis_GNN_V2/figures/pca_after_combat.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Sepsis_GNN_V2/figures/pca_before_combat.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `Sepsis_GNN_V2/results/baseline_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `Sepsis_GNN_V2/results/gnn_results.json`: Stores machine-readable outputs (metrics, manifests, model results, or metadata) consumed by downstream analysis and reporting scripts.
- `Sepsis_GNN_V2/scripts/01_combat_correction.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Sepsis_GNN_V2/scripts/02_build_graphs.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Sepsis_GNN_V2/scripts/03_baselines.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Sepsis_GNN_V2/scripts/04_train_gnn.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Sepsis_GNN_V2/scripts/05_external_validation.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
- `Sepsis_GNN_V2/scripts/06_explainability.py`: Implements a concrete engineering pipeline stage so experiments can be re-run and audited end-to-end.
### Group: `data`
- `data/raw/9606.protein.links.v12.0.txt.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `data/raw/BIOGRID-ORGANISM-Homo_sapiens-4.4.229.tab3.zip`: Project evidence artifact retained to preserve the full engineering history and reproducibility chain.
- `data/raw/GSE25504_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `data/raw/GSE26440_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `data/raw/GSE26440_series_matrix.txt.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `data/raw/GSE69686_family.soft.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
- `data/raw/GSE69686_series_matrix.txt.gz`: Compressed data payload retained to support offline reproducible processing during judging constraints.
### Group: `docs`
- `docs/publication/architecture/oigatv2_architecture.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/figure_captions.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `docs/publication/generate_publication_assets.py`: Python execution entrypoint or utility used for data handling, training, diagnostics, or environment validation.
- `docs/publication/interactive/dataset_composition.html`: Interactive visualization export for reviewer-side exploration of results.
- `docs/publication/interactive/external_accuracy_by_holdout.html`: Interactive visualization export for reviewer-side exploration of results.
- `docs/publication/interactive/l2_lr_tuning_top10.html`: Interactive visualization export for reviewer-side exploration of results.
- `docs/publication/interactive/metrics_summary_accuracy.html`: Interactive visualization export for reviewer-side exploration of results.
- `docs/publication/methods_summary.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `docs/publication/README.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `docs/publication/results_summary.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `docs/publication/tables/dataset_counts.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `docs/publication/tables/dataset_counts.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `docs/publication/tables/metrics_summary.csv`: Tabular result or dataset artifact for quantitative analysis, cross-checking, and figure generation.
- `docs/publication/tables/metrics_summary.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `docs/publication/visuals/dataset_composition.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/human_grouped5_accuracy.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/human_grouped5_l2_lr_tuning_best.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/human_grouped5_optimized_lr_roc.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/metrics_summary_accuracy.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/oigatv2_architecture.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/pca_after_combat.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/pca_before_combat.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/pipeline_overview.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/real_external_accuracy_by_holdout.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/real_roc_GSE160207.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/real_roc_GSE163812.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/real_roc_GSE180838.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/real_roc_GSE186141.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/roc_gat_v2.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/roc_gcn.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/roc_logisticregression.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `docs/publication/visuals/roc_randomforest.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
### Group: `download_data.py`
- `download_data.py`: Python execution entrypoint or utility used for data handling, training, diagnostics, or environment validation.
### Group: `download_data_from_urls.py`
- `download_data_from_urls.py`: Python execution entrypoint or utility used for data handling, training, diagnostics, or environment validation.
### Group: `download_data_with_gseapy.py`
- `download_data_with_gseapy.py`: Python execution entrypoint or utility used for data handling, training, diagnostics, or environment validation.
### Group: `download_geoparse.py`
- `download_geoparse.py`: Python execution entrypoint or utility used for data handling, training, diagnostics, or environment validation.
### Group: `download_remaining.py`
- `download_remaining.py`: Python execution entrypoint or utility used for data handling, training, diagnostics, or environment validation.
### Group: `draw_general_sepsis_topology.py`
- `draw_general_sepsis_topology.py`: Python execution entrypoint or utility used for data handling, training, diagnostics, or environment validation.
### Group: `figures`
- `figures/pca_after_combat.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
- `figures/pca_before_combat.png`: Generated figure used as visual evidence in notebook, poster, and publication assets.
### Group: `gnn_optimized`
- `gnn_optimized/01_build_graphs.py`: Python execution entrypoint or utility used for data handling, training, diagnostics, or environment validation.
- `gnn_optimized/02_train_gcn.py`: Python execution entrypoint or utility used for data handling, training, diagnostics, or environment validation.
- `gnn_optimized/03_train_graphsage.py`: Python execution entrypoint or utility used for data handling, training, diagnostics, or environment validation.
### Group: `investigate_gse26440_age.py`
- `investigate_gse26440_age.py`: Python execution entrypoint or utility used for data handling, training, diagnostics, or environment validation.
### Group: `logs`
- `logs/baseline_optimization_results.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `logs/GNN_Diagnostic_Report.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `logs/gnn_optimization_final_report.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `logs/gnn_optimization_results.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `logs/graphsage_results.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `logs/Module_A_Execution_Log.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `logs/Module_B_Execution_Log.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `logs/Module_C_Execution_Log.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `logs/Module_D_Execution_Log.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `logs/Optimization_Phase_Log.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `logs/results.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
- `logs/results_merged.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
### Group: `models`
- `models/gcn_merged.md`: Human-readable engineering artifact (design note, report, execution log, publication text, or reproducibility narrative).
### Group: `requirements.txt`
- `requirements.txt`: Reference or raw text artifact preserved for constraints, configuration, or evidence traceability.
### Group: `useless_for_now`
- `useless_for_now/agent_and_temp_artifacts/ACSEF_Master_Agent_Prompt.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/agent_and_temp_artifacts/ACSEF_swarm_state.json`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/agent_and_temp_artifacts/explain_log.txt`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/agent_and_temp_artifacts/projectcontext.txt`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/agent_and_temp_artifacts/projectimplementation-1.txt`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/agent_and_temp_artifacts/swarm_state/general_sepsis_v11_state.json`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/agent_and_temp_artifacts/swarm_state/overhaul_state.json`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/agent_and_temp_artifacts/tmp_geo/GSE154748_family.soft.gz`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/agent_and_temp_artifacts/tmp_geo/GSE270443_family.soft.gz`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/00_project_origin_and_scope.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/01_data_acquisition_and_qc.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/02_preprocessing_harmonization_batch.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/03_architecture_and_math.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/04_training_and_validation.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/05_xai_and_biomarkers.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/06_baselines_and_justification.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/07_osteogenesis_transfer.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/08_reproducibility_and_file_map.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/acsef_documents_engineering_notebook/README.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/acsef_submission_notebooks/engineering_notebook_2026-02-24.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/acsef_submission_notebooks/engineering_notebook_2026-03-03.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/docs_engineering_notebook/00_project_origin.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/docs_engineering_notebook/01_data_sources.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/docs_engineering_notebook/02_data_collection.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/docs_engineering_notebook/03_data_cleaning_qc.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/docs_engineering_notebook/04_batch_correction.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/docs_engineering_notebook/05_feature_engineering.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/docs_engineering_notebook/06_graph_construction.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/docs_engineering_notebook/07_model_architecture.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/docs_engineering_notebook/08_training_validation.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/docs_engineering_notebook/09_hyperparameter_tuning.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/docs_engineering_notebook/10_results_and_interpretation.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/docs_engineering_notebook/11_code_inventory.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/docs_engineering_notebook/12_limitations_next_steps.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
- `useless_for_now/legacy_notebooks/docs_engineering_notebook/README.md`: Archived during cleanup to reduce active-workspace noise while retaining provenance and recovery ability.
### Group: `verify_env.py`
- `verify_env.py`: Python execution entrypoint or utility used for data handling, training, diagnostics, or environment validation.

## Extended Engineering Narrative Appendix

This appendix records the broader engineering logic behind repository evolution. A high-complexity project almost always accumulates branch artifacts and alternate experiment lines; the key is not to erase that history but to make it legible.

The strongest reproducibility pattern in this workspace is artifact chaining: raw or corrected datasets flow into graph builders, graph outputs flow into training and evaluation, and final metrics flow into publication-generation scripts. Because outputs are file-based, claims can be traced and regenerated.

Another practical engineering insight is that naming quality directly affects maintenance cost. Version-only naming like v2, v3, or V11 is useful during rapid iteration, but eventually must be translated into semantic names when documentation stabilization is required.

The notebook therefore balances two principles that can conflict: preserve exact historical artifacts for auditability, and improve active readability so collaborators can operate the project safely without reverse engineering every path.

Modeling branches across sepsis, osteogenesis, and Alzheimer-related experiments indicate deliberate stress testing of representation choices and generalization behavior. Even when some branches are exploratory, they contribute to design confidence by clarifying what did and did not transfer.

Display-and-safety constraints from ACSEF/ISEF materially shape engineering communication architecture. Requirements such as no active links, per-graphic credits, and offline-ready materials force a packaging approach where critical evidence is embedded, local, and explicit.

In a judging context, a clean engineering notebook is not only documentation; it is operational risk control. It prevents accidental omission of key methods, highlights known limitations, and gives reviewers a direct path from high-level claim to low-level artifact.

Future maintainers should consider introducing a small metadata registry that tags each output with producer script, run date, parameter hash, and dependency version snapshot. That would reduce ambiguity when multiple similarly named results coexist.

A second future improvement is to formalize a results contract for each stage: expected files, schemas, and validation checks. Such a contract can be tested automatically and prevent silent drift between training outputs and publication summaries.

This master notebook intentionally keeps wording direct and technical, because the main objective is traceability and engineering clarity rather than marketing narrative. The resulting structure should be robust to future scaling and branch additions.

This appendix records the broader engineering logic behind repository evolution. A high-complexity project almost always accumulates branch artifacts and alternate experiment lines; the key is not to erase that history but to make it legible.

The strongest reproducibility pattern in this workspace is artifact chaining: raw or corrected datasets flow into graph builders, graph outputs flow into training and evaluation, and final metrics flow into publication-generation scripts. Because outputs are file-based, claims can be traced and regenerated.

Another practical engineering insight is that naming quality directly affects maintenance cost. Version-only naming like v2, v3, or V11 is useful during rapid iteration, but eventually must be translated into semantic names when documentation stabilization is required.

The notebook therefore balances two principles that can conflict: preserve exact historical artifacts for auditability, and improve active readability so collaborators can operate the project safely without reverse engineering every path.

Modeling branches across sepsis, osteogenesis, and Alzheimer-related experiments indicate deliberate stress testing of representation choices and generalization behavior. Even when some branches are exploratory, they contribute to design confidence by clarifying what did and did not transfer.

Display-and-safety constraints from ACSEF/ISEF materially shape engineering communication architecture. Requirements such as no active links, per-graphic credits, and offline-ready materials force a packaging approach where critical evidence is embedded, local, and explicit.

In a judging context, a clean engineering notebook is not only documentation; it is operational risk control. It prevents accidental omission of key methods, highlights known limitations, and gives reviewers a direct path from high-level claim to low-level artifact.

Future maintainers should consider introducing a small metadata registry that tags each output with producer script, run date, parameter hash, and dependency version snapshot. That would reduce ambiguity when multiple similarly named results coexist.

A second future improvement is to formalize a results contract for each stage: expected files, schemas, and validation checks. Such a contract can be tested automatically and prevent silent drift between training outputs and publication summaries.

This master notebook intentionally keeps wording direct and technical, because the main objective is traceability and engineering clarity rather than marketing narrative. The resulting structure should be robust to future scaling and branch additions.

This appendix records the broader engineering logic behind repository evolution. A high-complexity project almost always accumulates branch artifacts and alternate experiment lines; the key is not to erase that history but to make it legible.

The strongest reproducibility pattern in this workspace is artifact chaining: raw or corrected datasets flow into graph builders, graph outputs flow into training and evaluation, and final metrics flow into publication-generation scripts. Because outputs are file-based, claims can be traced and regenerated.

Another practical engineering insight is that naming quality directly affects maintenance cost. Version-only naming like v2, v3, or V11 is useful during rapid iteration, but eventually must be translated into semantic names when documentation stabilization is required.

The notebook therefore balances two principles that can conflict: preserve exact historical artifacts for auditability, and improve active readability so collaborators can operate the project safely without reverse engineering every path.

Modeling branches across sepsis, osteogenesis, and Alzheimer-related experiments indicate deliberate stress testing of representation choices and generalization behavior. Even when some branches are exploratory, they contribute to design confidence by clarifying what did and did not transfer.

Display-and-safety constraints from ACSEF/ISEF materially shape engineering communication architecture. Requirements such as no active links, per-graphic credits, and offline-ready materials force a packaging approach where critical evidence is embedded, local, and explicit.

In a judging context, a clean engineering notebook is not only documentation; it is operational risk control. It prevents accidental omission of key methods, highlights known limitations, and gives reviewers a direct path from high-level claim to low-level artifact.

Future maintainers should consider introducing a small metadata registry that tags each output with producer script, run date, parameter hash, and dependency version snapshot. That would reduce ambiguity when multiple similarly named results coexist.

A second future improvement is to formalize a results contract for each stage: expected files, schemas, and validation checks. Such a contract can be tested automatically and prevent silent drift between training outputs and publication summaries.

This master notebook intentionally keeps wording direct and technical, because the main objective is traceability and engineering clarity rather than marketing narrative. The resulting structure should be robust to future scaling and branch additions.

This appendix records the broader engineering logic behind repository evolution. A high-complexity project almost always accumulates branch artifacts and alternate experiment lines; the key is not to erase that history but to make it legible.

The strongest reproducibility pattern in this workspace is artifact chaining: raw or corrected datasets flow into graph builders, graph outputs flow into training and evaluation, and final metrics flow into publication-generation scripts. Because outputs are file-based, claims can be traced and regenerated.

Another practical engineering insight is that naming quality directly affects maintenance cost. Version-only naming like v2, v3, or V11 is useful during rapid iteration, but eventually must be translated into semantic names when documentation stabilization is required.

The notebook therefore balances two principles that can conflict: preserve exact historical artifacts for auditability, and improve active readability so collaborators can operate the project safely without reverse engineering every path.

Modeling branches across sepsis, osteogenesis, and Alzheimer-related experiments indicate deliberate stress testing of representation choices and generalization behavior. Even when some branches are exploratory, they contribute to design confidence by clarifying what did and did not transfer.

Display-and-safety constraints from ACSEF/ISEF materially shape engineering communication architecture. Requirements such as no active links, per-graphic credits, and offline-ready materials force a packaging approach where critical evidence is embedded, local, and explicit.

In a judging context, a clean engineering notebook is not only documentation; it is operational risk control. It prevents accidental omission of key methods, highlights known limitations, and gives reviewers a direct path from high-level claim to low-level artifact.

Future maintainers should consider introducing a small metadata registry that tags each output with producer script, run date, parameter hash, and dependency version snapshot. That would reduce ambiguity when multiple similarly named results coexist.

A second future improvement is to formalize a results contract for each stage: expected files, schemas, and validation checks. Such a contract can be tested automatically and prevent silent drift between training outputs and publication summaries.

This master notebook intentionally keeps wording direct and technical, because the main objective is traceability and engineering clarity rather than marketing narrative. The resulting structure should be robust to future scaling and branch additions.

This appendix records the broader engineering logic behind repository evolution. A high-complexity project almost always accumulates branch artifacts and alternate experiment lines; the key is not to erase that history but to make it legible.

The strongest reproducibility pattern in this workspace is artifact chaining: raw or corrected datasets flow into graph builders, graph outputs flow into training and evaluation, and final metrics flow into publication-generation scripts. Because outputs are file-based, claims can be traced and regenerated.

Another practical engineering insight is that naming quality directly affects maintenance cost. Version-only naming like v2, v3, or V11 is useful during rapid iteration, but eventually must be translated into semantic names when documentation stabilization is required.

The notebook therefore balances two principles that can conflict: preserve exact historical artifacts for auditability, and improve active readability so collaborators can operate the project safely without reverse engineering every path.

Modeling branches across sepsis, osteogenesis, and Alzheimer-related experiments indicate deliberate stress testing of representation choices and generalization behavior. Even when some branches are exploratory, they contribute to design confidence by clarifying what did and did not transfer.

Display-and-safety constraints from ACSEF/ISEF materially shape engineering communication architecture. Requirements such as no active links, per-graphic credits, and offline-ready materials force a packaging approach where critical evidence is embedded, local, and explicit.

In a judging context, a clean engineering notebook is not only documentation; it is operational risk control. It prevents accidental omission of key methods, highlights known limitations, and gives reviewers a direct path from high-level claim to low-level artifact.

Future maintainers should consider introducing a small metadata registry that tags each output with producer script, run date, parameter hash, and dependency version snapshot. That would reduce ambiguity when multiple similarly named results coexist.

A second future improvement is to formalize a results contract for each stage: expected files, schemas, and validation checks. Such a contract can be tested automatically and prevent silent drift between training outputs and publication summaries.

This master notebook intentionally keeps wording direct and technical, because the main objective is traceability and engineering clarity rather than marketing narrative. The resulting structure should be robust to future scaling and branch additions.

This appendix records the broader engineering logic behind repository evolution. A high-complexity project almost always accumulates branch artifacts and alternate experiment lines; the key is not to erase that history but to make it legible.

The strongest reproducibility pattern in this workspace is artifact chaining: raw or corrected datasets flow into graph builders, graph outputs flow into training and evaluation, and final metrics flow into publication-generation scripts. Because outputs are file-based, claims can be traced and regenerated.

Another practical engineering insight is that naming quality directly affects maintenance cost. Version-only naming like v2, v3, or V11 is useful during rapid iteration, but eventually must be translated into semantic names when documentation stabilization is required.

The notebook therefore balances two principles that can conflict: preserve exact historical artifacts for auditability, and improve active readability so collaborators can operate the project safely without reverse engineering every path.

Modeling branches across sepsis, osteogenesis, and Alzheimer-related experiments indicate deliberate stress testing of representation choices and generalization behavior. Even when some branches are exploratory, they contribute to design confidence by clarifying what did and did not transfer.

Display-and-safety constraints from ACSEF/ISEF materially shape engineering communication architecture. Requirements such as no active links, per-graphic credits, and offline-ready materials force a packaging approach where critical evidence is embedded, local, and explicit.

In a judging context, a clean engineering notebook is not only documentation; it is operational risk control. It prevents accidental omission of key methods, highlights known limitations, and gives reviewers a direct path from high-level claim to low-level artifact.
### Comparative Engineering Narrative by Workstream (Old Snapshot vs Current Workspace)

#### Workstream 1: Repository Intent, Scope Boundary, and Audience
In the old snapshot, the repository read as a technically strong but primarily builder-facing environment: someone deeply familiar with prior iterations could navigate it, but a reviewer arriving cold would need to infer context from multiple partially overlapping documents. In the current workspace, the intent has shifted toward dual-audience readability. It remains capable of serving a technical builder, but it now also serves an evaluator who needs traceability across the entire engineering process. The most meaningful shift is not simply that more files exist; it is that more files now carry explicit packaging roles. Old-state friction came from context switching between legacy notebook chapters, branch plans, publication assets, and run logs. New-state structure intentionally lowers that friction by pulling core narrative into one master notebook and reclassifying non-primary materials into clearly named archive zones. This is important for engineering design-process reporting because it shows a transition from execution-first organization to verification-first organization. For project defense, this means claims can be traced in fewer hops, and this lower path length directly improves audit reliability.

#### Workstream 2: Data Lifecycle Governance and Asset Proliferation
The old snapshot already had nontrivial data handling, but the new workspace demonstrates a broader data lifecycle surface, especially in Alzheimer-related and transfer-oriented paths. A large amount of new data-oriented artifacts appears not as random cache noise but as reproducibility-grade deliverables: processed datasets, expanded cohort variants, cohort metadata alignments, and specialized gene lists. Compared with the old snapshot, the current repository behaves more like a data product with explicit derivation branches than a single pipeline with a handful of outputs. The engineering implication is that data provenance now must be discussed as a branching graph rather than a linear sequence. This pushes the notebook to document not only what transformed data exists, but why parallel variants were retained. In design-process terms, this reflects a conscious tradeoff: storage and complexity increase were accepted in exchange for stronger falsifiability and ability to test robustness claims. For judges or reviewers, this provides a better basis for questions like "what changed between cohort variant A and B" or "which gene-set cardinality was used for this specific result."

#### Workstream 3: Model Artifact Strategy and Reproducibility Granularity
The model layer changed from having representative checkpoints and summaries to holding a significantly broader set of run-specific and branch-specific `.pt` artifacts. This is not merely volume inflation. The added checkpoints represent expanded model lineage documentation across architecture families, disease contexts, and evaluation conditions. In the old snapshot, one could reconstruct intent from script naming and selected outputs. In the current workspace, one can often reconstruct intent from script naming plus concrete model artifacts, which is stronger evidence. Engineering-wise, this reduces reliance on memory and narrative-only claims. A checkpoint plus associated metrics artifact provides an evidentiary pair that can be validated independently of notebook prose. The tradeoff is maintenance burden: more artifacts require clearer manifests and stronger naming discipline. The overhaul addresses this partially through curated visual and result summaries, but future work could tighten linkage further by assigning immutable run identifiers. Still, compared with old state, the new state clearly favors reproducibility granularity over minimalist repository size, which is a defensible choice for scientific engineering submissions.

#### Workstream 4: Script Surface and Real Functional Delta
A key analytical finding in this comparison is that the modified-file count alone overstates functional algorithm change. Most modified text files, when normalized for line endings, are semantically unchanged. This matters because it prevents false narrative inflation: a large modified count does not automatically indicate large logic drift. The real functional delta is concentrated, and that concentration itself is a quality signal. The two CH-DANN training scripts show an important correction: preprocessing scaling is now performed inside each fold rather than globally before splitting. This closes a leakage pathway and improves validity of baseline comparison metrics. In the old snapshot, that leakage risk existed for those paths; in the current workspace, it is remediated. Additional substantive edits in documentation and image guidance are packaging-consistency improvements rather than model-logic changes. This pattern indicates deliberate maturity: avoid unnecessary code churn, patch high-impact validity issues, and focus broad repository growth on results and curation layers. In engineering-process language, this is a targeted corrective iteration, not a wholesale algorithm rewrite.

#### Workstream 5: Documentation Topology, Notebook Consolidation, and Reviewer Navigation
The old snapshot contained multiple notebook trees that were individually useful but collectively redundant for a final-facing narrative. The current workspace reframes that structure by establishing a canonical root notebook and demoting split notebooks into archival locations. This is not equivalent to deletion; it is controlled demotion. The old structure optimized for modular drafting, while the new structure optimizes for cohesive review. The engineering notebook now has to carry both scientific content and repository governance metadata, and this comparison section is part of that governance role. One practical benefit of consolidation is reduced contradictory phrasing across duplicated chapters. Another is that figure placeholder logic can be centrally maintained. A remaining risk is that canonicalization can hide historical context if archival links are unclear, but the current layout mitigates that by preserving legacy directories under explicit names. Compared with old state, the new state gives a stronger single-source-of-truth contract, which is critical when submission pressure increases and multiple derivative documents exist (poster drafts, manuscripts, figure manifests, and checklists).

#### Workstream 6: Visual Storytelling Architecture and Submission Readiness
In the old snapshot, figure assets were present, but their operational status for final display was not as clearly segregated from intermediate/raw galleries. The current workspace introduces a curated twelve-image sequence under `ACSEF_Final_Submission/final_visuals`, supported by machine-readable manifests. This shift is strategically important because it turns visualization from an ad hoc browse task into a deterministic pipeline output. For review settings where time is limited, deterministic figure ordering lowers ambiguity and improves consistency between oral explanation, poster layout, and notebook references. The new placeholder map added in this notebook aligns directly with those curated images, creating a one-to-one mapping from narrative slot to file path. In comparison terms, old state had richer exploratory figure context but weaker final-display curation boundaries; new state has both, with clearer intended usage. From engineering design-process perspective, this is a communication-stage maturation: after model and validation cycles, the team formalized artifact presentation constraints without discarding underlying exploratory assets.

#### Workstream 7: Cross-Disease Generalization Narrative and Branch Coupling
The old snapshot included multi-branch experimentation, but the relationship between branches was less explicitly packaged as one coherent engineering story. In the current workspace, branch outputs across sepsis, Alzheimer-related transfer, and osteogenesis are more visibly co-present and better summarized. This does not mean the biological problems are identical; it means the engineering framework for representation learning and validation is being exercised across heterogeneous contexts. The practical value of this expansion is in stress-testing assumptions: if a modeling pattern only works in one narrow setting, that limitation should be visible. The current layout enables that visibility by preserving branch-specific results while introducing cross-branch comparison figures and summaries. Compared with old state, this broadens the evidence base for discussing robustness and portability. It also raises the burden on notebook clarity, because readers must distinguish between primary objective claims and transfer exploratory claims. The updated notebook addresses that by separating manifest facts from interpretation and by preserving branch-local artifact naming.

#### Workstream 8: Archival Policy, Hygiene, and Non-Destructive Cleanup
A central engineering decision in the overhaul is non-destructive cleanup. Instead of hard-deleting historical notebooks and temporary orchestration artifacts, the project moved these into `useless_for_now` archival areas. In old state, those materials lived closer to active paths, which improved immediate access for developers but increased noise for evaluators. In new state, the archive policy creates a compromise: provenance is retained, active workspace readability improves. This is a textbook engineering governance tradeoff between minimalism and audit completeness. For competition or publication contexts, the chosen policy is stronger because it supports retrospective scrutiny. If a reviewer asks where earlier notebooks went, the answer is explicit and verifiable. If a collaborator needs to recover earlier context, paths still exist. Compared with old state, the project now has a clearer lifecycle concept for artifacts: active, curated, archived. This lifecycle framing reduces accidental misuse of deprecated files and lowers the chance that reviewers interpret obsolete drafts as current canonical methodology.

#### Workstream 9: Evidence Density, Claim Traceability, and Risk Posture
The current workspace increases evidence density substantially: more result JSONs, more metrics summaries, more visual outputs, and clearer manifests. Evidence density, however, only helps when traceability is maintained. The comparison indicates traceability improved through curated maps and centralized narrative, though it remains dependent on naming consistency and path clarity. A strong sign is that the overhaul exposes where true functional changes occurred and where only formatting normalization happened. That transparency improves trust because it avoids overstating novelty. Compared with old snapshot, risk posture improved in several dimensions: leakage risk mitigation in specific scripts, reviewer navigation risk reduction through notebook consolidation, and communication risk reduction through curated final visuals. Residual risks remain: high artifact volume can still overwhelm readers, and mixed-era naming conventions can still confuse if not continually curated. Nonetheless, the engineering direction is clearly toward defensible traceability rather than opaque complexity, which aligns well with formal engineering design-process expectations.

#### Workstream 10: Practical Reproducibility Operations
From an operations perspective, the old snapshot offered reproducibility through scripts and outputs, but execution pathways required more implicit knowledge. The current workspace improves operational reproducibility by making output landscapes richer and packaging intent clearer. In practical terms, someone reproducing a claim now has better chances of finding needed artifacts without reconstructing undocumented assumptions. The presence of curated results directories, explicit figure manifests, and stable top-level planning documents reduces execution ambiguity. Another operational improvement is semantic separation of presentation-ready outputs from exploratory materials. During high-pressure deadlines, this reduces accidental use of stale or non-final assets. The comparison also shows that while many files changed, the core pipeline identity did not fracture. This continuity matters: large repository changes can create reproducibility regressions if the execution contract shifts unpredictably. Here, the contract appears strengthened rather than replaced, with targeted corrections and broad evidence expansion. That is a positive signal for both immediate submission quality and longer-term maintainability.

#### Workstream 11: What Actually Improved Relative to Reviewer Questions
A useful comparison lens is to map repository changes to likely reviewer questions. Old state could answer many technical questions but often required opening multiple disconnected files. New state improves response quality for questions like: "Which figures are final?", "Which notebook is canonical?", "What exactly changed since prior snapshot?", and "Where is evidence for branch-specific claims?". The exhaustive path manifest in this notebook directly addresses the delta question, and the curated final visuals address the final-figure question. The centralized notebook structure addresses canonical narrative ambiguity. The remaining reviewer challenge is volume: there is now enough evidence that curation discipline must be actively maintained to keep the narrative legible. Compared with old state, however, the new repository is better prepared for formal scrutiny because it can provide deterministic answers to provenance and packaging questions that previously required more interpretation.

#### Workstream 12: Forward Engineering Recommendations Anchored to This Delta
Based on this comparison, the next maturity step is not more raw artifact growth but stronger artifact contracts. A lightweight run registry could map each key output to producer script, parameter profile, and timestamp window. This would make future old-vs-new diffs even more actionable by distinguishing semantic changes from formatting or re-exports automatically. Another recommendation is to standardize path aliases for canonical artifacts (for example, a single `latest` manifest per branch) so downstream documents never point to stale filenames. Finally, unit-style validation for notebook references would catch renamed file drift before packaging, extending the consistency improvements already visible in the `.tex` updates. Relative to old snapshot, the project has already moved significantly toward robust engineering documentation. These recommendations would consolidate that gain and reduce maintenance cost in subsequent overhaul cycles.
