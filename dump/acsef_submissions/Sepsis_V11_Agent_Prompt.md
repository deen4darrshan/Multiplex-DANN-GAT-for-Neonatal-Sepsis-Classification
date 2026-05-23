# MISSION: Scale the V11 Multiplex GNN–DANN Architecture to General Sepsis

You are an expert AI Bioinformatics Engineer. Your mission is to **replicate and adapt** a proven Multiplex Hypergraph GNN + Domain-Adversarial architecture (internally called "V11") — originally built for **neonatal sepsis** — and apply it to **general (adult/pediatric) sepsis** classification using publicly available GEO expression datasets. You are working inside the `ppi_gnn_combined_dataset` repository.

---

## STEP 0: Analyze the Existing Codebase

Before writing a single line of new code, you **MUST** thoroughly read and understand the following files and directories. Do not skip any of them.

### Core Architecture (Read First)
| File | What to Learn |
|---|---|
| `CH_DANN_Plan/scripts/19_train_v11_multiplex_dann.py` | **THE reference implementation.** Contains the full `MultiplexGNNGuidedDANN` model class, the 3-relation hypergraph construction (KEGG, STRING, Co-Expression), the `GeneScorer` attention mask, the Domain-Adversarial training loop with `GradientReversalLayer`, and the `collate_multiplex` batching function. Study every function. |
| `CH_DANN_Plan/model_architecture.md` | High-level documentation of the architecture design decisions. |
| `CH_DANN_Plan/CH_DANN_PROJECT_PLAN.md` | The full evolution from V1→V12, including what failed and why. |

### Data Pipeline (Read Second)
| File | What to Learn |
|---|---|
| `CH_DANN_Plan/results/expression_combat_v2.csv` | The final batch-corrected expression matrix used for neonatal sepsis (~101 MB, 319 samples × 2000 genes). Understand the format. |
| `CH_DANN_Plan/results/metadata_v2.csv` | Sample metadata with `condition` (Sepsis/Control), `batch` (GSE accession), and `platform` columns. |
| `CH_DANN_Plan/results/gene_list_v2.json` | The 2000 most-variable genes selected for graph construction. |
| `CH_DANN_Plan/results/pathway_info_v2.json` | Pre-computed KEGG pathway and STRING PPI edge information. |
| `01_id_mapping.py`, `02_merge_combat.py` | How raw GEO data was downloaded, probe-to-gene mapped, and batch-corrected with ComBat. |

### Prior Model Evolution (Skim for Context)
| File | What to Learn |
|---|---|
| `CH_DANN_Plan/scripts/16_train_v8_gnn_guided.py` | The GNN-Guided feature selection approach. |
| `CH_DANN_Plan/scripts/18_train_v10_multiplex.py` | The Multiplex (3-relation) architecture before DANN was added. |
| `CH_DANN_Plan/scripts/21_train_v12_pure_hgcn.py` | The pure HGCN experiment (no MLP) — it **failed** (collapsed to random chance). This justifies the MLP component. |
| `CH_DANN_Plan/results/v11_multiplex_dann_results.json` | V11's final cross-validation metrics. |
| `CH_DANN_Plan/results/v11_gse26440_external_results.json` | V11's external validation performance (0.985 AUROC). |

### Scaling Precedent
| File | What to Learn |
|---|---|
| `Osteogenesis imperfecta/` (entire folder) | This folder contains a **prior successful scaling** of the same architecture to a completely different rare disease (Osteogenesis Imperfecta). Study how the data pipeline and training scripts were adapted. |

### Other Relevant Context
| File | What to Learn |
|---|---|
| `Sepsis_GNN_V2/` | An earlier, simpler GNN attempt on sepsis data. May contain useful data download scripts. |
| `gnn_optimized/03_train_graphsage.py` | GraphSAGE baseline — useful for comparison. |
| `06c_train_gat_expanded.py` | GAT baseline. |
| `requirements.txt` | Current Python dependencies. |
| `logs/` | Historical execution logs and results summaries. |

---

## STEP 1: Data Acquisition for General Sepsis

You must source **general sepsis** (not neonatal-specific) gene expression datasets from NCBI GEO. Target adult/pediatric sepsis cohorts.

**Key requirements:**
- Find at least **2–3 independent GEO datasets** with sepsis vs. healthy-control labels (e.g., GSE65682, GSE95233, GSE69528, GSE134347 — verify availability).
- Download expression matrices using GEOparse or direct download.
- Perform probe-to-gene-symbol ID mapping (use the approach in `01_id_mapping.py` as a reference).
- Apply ComBat batch correction across all datasets (use `02_merge_combat.py` as a reference).
- Select the top 2000 most-variable genes (same strategy as `gene_list_v2.json`).
- Save the final cleaned data into a new folder: `General_Sepsis_V11/results/`.

---

## STEP 2: Graph Construction

Build the 3-relation multiplex hypergraph for the general sepsis gene set, exactly as the V11 architecture expects:

1. **KEGG Pathway Hyperedges**: Map the 2000 selected genes to KEGG pathways. Each pathway becomes one hyperedge connecting all member genes.
2. **STRING PPI Edges**: Download STRING v12.0 interactions for Homo sapiens (taxid 9606). Filter for combined_score ≥ 700. Match to gene set.
3. **Co-Expression Edges**: Compute Spearman rank correlation on the **training fold only** (to prevent data leakage). Threshold at |ρ| > 0.7.

> ⚠️ **CRITICAL WARNING — Windows SciPy Deadlock**: On Python 3.13 + Windows, importing `scipy.stats.rankdata` or `median_abs_deviation` at the module level can cause the interpreter to silently deadlock. If this happens, implement Spearman correlation using pure Pandas `.rank()` + NumPy matrix operations instead of SciPy. See the workaround in `CH_DANN_Plan/scripts/22_explain_v11.py` for reference.

> ⚠️ **CRITICAL WARNING — STRING Memory**: The STRING database is ~11M rows. Load it in chunks (`pd.read_csv(..., chunksize=500000)`) to prevent swap-thrashing OOM crashes.

---

## STEP 3: Train the V11 Architecture

Directly adapt `19_train_v11_multiplex_dann.py`. The architecture consists of:

```
Input Gene Expression (N × 2000)
        ↓
┌─── Multiplex HypergraphConv ───┐
│  Relation 1: KEGG Pathways     │
│  Relation 2: STRING PPI        │
│  Relation 3: Co-Expression     │
└────────────────────────────────┘
        ↓
   Relation Attention (learned α₁, α₂, α₃)
        ↓
   Gene Scorer (attention mask ∈ [0,1] per gene)
        ↓
   Weighted Expression → MLP Classifier → Sepsis/Control
                           ↓ (GRL)
                    Domain Adversarial Head → Batch Prediction
```

**Training requirements:**
- Use **Stratified Group K-Fold** (stratify by condition, group by patient/batch) with K=5.
- Use the **Domain-Adversarial Neural Network (DANN)** head to suppress batch effects between datasets. The `batch` column from metadata drives this.
- Early stopping on validation accuracy (patience=20).
- Hyperparameters: Start with the same ones from V11 (lr=1e-3, hidden_dim=128, dropout=0.5, λ_dann=0.1). Tune if needed.
- Save all fold models to `General_Sepsis_V11/models/`.
- Save cross-validation results to `General_Sepsis_V11/results/`.

---

## STEP 4: Evaluate & Compare

1. **Cross-validation metrics**: Report AUROC, Accuracy, F1, Precision, Recall for each fold and their means.
2. **Hold-out external validation**: If you found ≥3 GEO datasets, hold one out entirely for external validation.
3. **Save all results** as JSON in `General_Sepsis_V11/results/general_sepsis_v11_results.json`.

---

## STEP 5: Folder Structure

All work must be done in a new subfolder. Final layout:

```
General_Sepsis_V11/
├── scripts/
│   ├── 01_download_and_preprocess.py
│   ├── 02_build_graphs.py
│   ├── 03_train_v11_general_sepsis.py
│   └── 04_evaluate.py
├── results/
│   ├── expression_combat.csv
│   ├── metadata.csv
│   ├── gene_list.json
│   ├── pathway_info.json
│   └── general_sepsis_v11_results.json
├── models/
│   └── (fold .pt files)
└── logs/
    └── (dated execution logs)
```

---

## CRITICAL REMINDERS
- **This is general sepsis, NOT neonatal sepsis.** The datasets, patient populations, and clinical context are different. Make sure you source adult/pediatric sepsis cohorts from GEO.
- **Do not modify any files under `CH_DANN_Plan/`.** That is the reference neonatal implementation. Copy and adapt into `General_Sepsis_V11/`.
- **Date all logs.** If anything fails, document what went wrong and how you fixed it.
- **The MLP component is ESSENTIAL.** The pure HGCN (V12) collapsed. Do not remove the MLP.
- **Compute Co-Expression edges per fold on training data only** to prevent data leakage.

**Execute all steps to completion. The final `General_Sepsis_V11/` folder should be a fully self-contained, trained, and evaluated replication of the V11 architecture on general sepsis data.**
