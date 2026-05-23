# Multiplex DANN-GAT for Biomedical Classification

A PyTorch implementation of **Multiplex Domain-Adversarial Graph Attention Networks** for biomedical classification using gene expression data. This framework applies multiplex hypergraph neural networks with domain adaptation to suppress batch effects across heterogeneous datasets.

## Overview

This project implements a neural network architecture that:
- Models biological relationships (KEGG pathways, PPI networks, co-expression) as a multiplex hypergraph
- Uses relation-aware attention to aggregate multi-relational gene features
- Employs Domain-Adversarial Neural Network (DANN) training to suppress batch effects
- Enables transfer learning across related disease domains

## Project Structure

```
.
├── pipelines/                    # Executable ML pipelines
│   ├── sepsis_multiplex_dann/    # Sepsis classification pipeline
│   ├── alzheimers_transfer/       # Alzheimer's transfer learning pipeline
│   └── osteogenesis_transfer/    # Osteogenesis Imperfecta pipeline
├── models/
│   ├── architectures/            # Model architecture definitions
│   │   └── multiplex_dann_gat.py # Main Multiplex DANN-GAT architecture
│   └── checkpoints/               # Pre-trained model weights
│       ├── sepsis/
│       ├── alzheimers/
│       └── osteogenesis/
├── results/
│   ├── metrics/                   # JSON performance metrics
│   ├── logs/                      # Training/evaluation logs
│   └── figures/                   # Evaluation visualizations
├── docs/                          # Technical documentation
│   ├── pipeline_overview.md
│   ├── model_architectures.md
│   └── results_summary.md
└── data/                          # Raw and processed data
    ├── raw/                        # Original GEO datasets
    ├── processed/                 # Batch-corrected expression matrices
    └── graphs/                    # Constructed PPI graphs
```

## Pipelines

### Sepsis Classification (`pipelines/sepsis_multiplex_dann/`)

1. **01_preprocess_sepsis.py** - Download GEO datasets, ID mapping, ComBat harmonization
2. **02_construct_sepsis_graphs.py** - Build PPI graphs (KEGG, STRING, Co-Expression)
3. **03_train_multiplex_dann.py** - Train the Multiplex DANN-GAT model
4. **04_train_ablation_gat.py** - Train baseline GAT models
5. **05_evaluate_sepsis.py** - Evaluate on pediatric holdout GSE26440
6. **06_generate_sepsis_plots.py** - Generate ROC, PR, and SHAP plots

### Alzheimer's Transfer (`pipelines/alzheimers_transfer/`)

1. **01_preprocess_alzheimers.py** - Prepare ADNI and GEO series
2. **02_train_alzheimers_transfer.py** - Weight transfer and fine-tuning
3. **03_evaluate_alzheimers.py** - Brain LOCO evaluation
4. **04_evaluate_seed_stability.py** - Measure seed stability
5. **05_summarize_alzheimers.py** - Compile cross-cohort matrices

### Osteogenesis Transfer (`pipelines/osteogenesis_transfer/`)

1. **01_preprocess_oi.py** - Harmonize osteogenesis datasets
2. **02_construct_oi_graphs.py** - Build graphs from PPI network
3. **03_train_oi_gnn.py** - Train baselines and GNNs
4. **04_evaluate_oi.py** - Multi-cohort evaluation
5. **05_summarize_oi.py** - Write final metrics

## Architecture

The core **Multiplex DANN-GAT** architecture:

```
Input Gene Expression (N × 2000 genes)
        ↓
┌──────────────────────────────────────────┐
│  Multiplex HypergraphConv (3 Relations)   │
│  ┌────────────────────────────────────┐  │
│  │  Relation 1: KEGG Pathways        │  │
│  │  Relation 2: STRING PPI           │  │
│  │  Relation 3: Co-Expression        │  │
│  └────────────────────────────────────┘  │
└──────────────────────────────────────────┘
        ↓
   Relation Attention (learned α₁, α₂, α₃)
        ↓
   Gene Scorer (attention mask ∈ [0,1])
        ↓
   MLP Classifier → Disease Prediction
           ↓ (Gradient Reversal Layer)
    Domain Adversarial Head → Batch Prediction
```

### Key Features

- **Multiplex HypergraphConv**: Three parallel relation layers capturing different biological relationships
- **Relation Attention**: Learned weights (α₁, α₂, α₃) aggregating per-gene representations
- **Gene Scorer**: Per-gene importance weights for feature selection
- **DANN Training**: Gradient reversal layer for batch-invariant representations

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric
- scikit-learn
- GEOparse
- pandas, numpy, scipy

## Usage

### Data Preprocessing

```bash
cd pipelines/sepsis_multiplex_dann
python 01_preprocess_sepsis.py
python 02_construct_sepsis_graphs.py
```

### Training

```bash
python 03_train_multiplex_dann.py
```

### Evaluation

```bash
python 05_evaluate_sepsis.py
python 06_generate_sepsis_plots.py
```

## Results

| Task | Dataset | AUROC | Accuracy | F1 | Model |
|------|---------|-------|----------|----|----|
| Neonatal Sepsis | GEO (5-fold CV) | 0.980 | 0.978 | 0.973 | Multiplex DANN-GAT |
| Alzheimer's | ADNI + GEO (LOCO) | 0.944 | 0.905 | 0.900 | Multiplex DANN-GAT (transfer) |
| Osteogenesis Imperfecta | GEO (cross-cohort) | 0.774 | 0.542 | 0.565 | GAT |

**Comparison against baselines (LOBO protocol):**
- Logistic Regression: sepsis AUROC 0.913, AD AUROC 0.828
- Random Forest: sepsis AUROC 0.900, AD AUROC 0.802

Detailed results and per-fold metrics available in `results/metrics/` and `results/figures/`.

## Documentation

- [Pipeline Overview](docs/pipeline_overview.md) - Detailed pipeline documentation
- [Model Architectures](docs/model_architectures.md) - Deep-dive into underlying architecture
- [Results Summary](docs/results_summary.md) - Performance metrics and findings

## Data Sources

- **Gene Expression Omnibus (GEO)** - Public gene expression datasets
- **STRING Database** - Protein-protein interaction networks  
- **KEGG Pathway Database** - Pathway annotations
- **ADNI** - Alzheimer's Disease Neuroimaging Initiative

## Citation

If this work is useful for your research, please cite:

```bibtex
@software{multiplex_dann_gat,
  title={Multiplex DANN-GAT for Biomedical Classification},
  author={Deenadarrshan Sathiyamoorthi and Terry Ding},
  year={2026},
  url={https://github.com/deen4darrshan/PPI-Guided-GNN-for-Neonatal-Sepsis}
}
```

## License

This project is available for academic use. Please contact the author for commercial licensing inquiries.
