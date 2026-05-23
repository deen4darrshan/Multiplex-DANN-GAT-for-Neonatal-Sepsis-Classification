# Pipeline Overview

This repository contains machine learning pipelines for applying Multiplex Domain-Adversarial Graph Attention Networks (Multiplex DANN-GAT) to biomedical classification tasks across multiple disease domains.

## Project Structure

```
pipelines/
├── sepsis_multiplex_dann/    # General sepsis classification pipeline
├── alzheimers_transfer/       # Alzheimer's transfer learning pipeline
└── osteogenesis_transfer/     # Osteogenesis Imperfecta transfer learning pipeline

models/
├── architectures/             # Model architecture definitions
│   ├── multiplex_dann_gat.py  # Main Multiplex DANN-GAT architecture
│   ├── hgcn.py
│   ├── gcn.py
│   ├── gat.py
│   └── baselines.py
└── checkpoints/              # Pre-trained model weights
    ├── sepsis/
    ├── alzheimers/
    └── osteogenesis/

results/
├── metrics/                   # JSON performance metrics
├── logs/                      # Training/evaluation logs
└── figures/                   # Evaluation visualizations

docs/
├── pipeline_overview.md       # This file
├── model_architectures.md     # Architecture documentation
└── results_summary.md         # Results summary

dump/                          # Legacy files preserved for reference
```

## Pipelines

### Sepsis Multiplex DANN Pipeline

The sepsis pipeline applies the Multiplex DANN-GAT architecture to gene expression data for sepsis classification.

**Scripts:**
- `01_preprocess_sepsis.py` - Download GEO datasets, ID mapping, ComBat harmonization
- `02_construct_sepsis_graphs.py` - Build PPI graphs using STRING database (KEGG, STRING, Co-Expression)
- `03_train_multiplex_dann.py` - Train the primary Multiplex DANN-GAT model
- `04_train_ablation_gat.py` - Train baseline GAT models for comparison
- `05_evaluate_sepsis.py` - Evaluate on pediatric holdout GSE26440
- `06_generate_sepsis_plots.py` - Generate ROC, PR, SHAP, and attribution plots

### Alzheimer's Transfer Pipeline

Applies transfer learning from the sepsis model to Alzheimer's disease classification using ADNI and GEO datasets.

**Scripts:**
- `01_preprocess_alzheimers.py` - Prepare ADNI and GEO series
- `02_train_alzheimers_transfer.py` - Weight transfer and fine-tuning
- `03_evaluate_alzheimers.py` - Brain LOCO (Leave-One-Cohort-Out) evaluation
- `04_evaluate_seed_stability.py` - Measure seed stability
- `05_summarize_alzheimers.py` - Compile cross-cohort matrices

### Osteogenesis Transfer Pipeline

Applies the architecture to Osteogenesis Imperfecta classification using multi-cohort gene expression data.

**Scripts:**
- `01_preprocess_oi.py` - Harmonize osteogenesis datasets
- `02_construct_oi_graphs.py` - Build graphs from PPI network
- `03_train_oi_gnn.py` - Train baselines and GNNs
- `04_evaluate_oi.py` - Multi-cohort evaluation
- `05_summarize_oi.py` - Write final metrics

## Data Sources

- **Gene Expression Omnibus (GEO)** - Public gene expression datasets
- **STRING Database** - Protein-protein interaction networks
- **KEGG Pathway Database** - Pathway annotations
- **ADNI** - Alzheimer's Disease Neuroimaging Initiative

## Requirements

See `requirements.txt` for Python dependencies. Key packages:
- PyTorch + PyTorch Geometric
- scikit-learn
- GEOparse
- pandas, numpy, scipy