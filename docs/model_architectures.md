# Model Architectures

## Multiplex DANN-GAT (Primary Architecture)

The core architecture is a **Multiplex Domain-Adversarial Graph Attention Network** designed for biomedical classification with batch effect suppression.

### Architecture Design

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

### Key Components

1. **Multiplex HypergraphConv**: Three parallel relation layers operating on the same gene nodes, each capturing different biological relationships (pathway, PPI, co-expression).

2. **Relation Attention**: Learned attention weights (α₁, α₂, α₃) that aggregate per-gene representations from all three relations.

3. **Gene Scorer**: A scoring head that produces per-gene importance weights (sigmoid output ∈ [0,1]) for feature selection.

4. **MLP Classifier**: Processes attention-weighted gene expression for final classification.

5. **Domain-Adversarial Training (DANN)**: Gradient reversal layer that trains the model to be invariant to batch effects (dataset of origin).

### Class: `MultiplexGNNGuidedDANN`

```python
class MultiplexGNNGuidedDANN(nn.Module):
    def __init__(self, n_genes, h_dim=64, dropout=0.3, n_relations=3):
        # Shared gene embedding
        self.gene_embed = nn.Sequential(nn.Linear(1, h_dim), nn.LayerNorm(h_dim), nn.GELU())
        
        # Per-relation HypergraphConv branches
        self.convs1 = nn.ModuleList([HypergraphConv(h_dim, h_dim) for _ in range(n_relations)])
        self.convs2 = nn.ModuleList([HypergraphConv(h_dim, h_dim) for _ in range(n_relations)])
        
        # Relation-aware attention
        self.relation_attn = nn.Sequential(...)
        
        # Per-gene scoring head
        self.gene_scorer = nn.Sequential(...)
        
        # MLP and classifier
        self.mlp = nn.Sequential(...)
        self.classifier = nn.Sequential(...)
        
        # DANN domain discriminator
        self.domain_discriminator = nn.Sequential(...)
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 64 |
| Dropout | 0.3 |
| Learning rate | 3e-4 |
| Weight decay | 5e-4 |
| Batch size | 16 |
| Epochs | 150 |
| Patience | 30 |
| STRING threshold | 700 |
| Co-expression threshold | 0.7 (Spearman) |

## Baseline Models

### GCN (Graph Convolutional Network)
Standard 2-layer GCN for node classification on PPI graph.

### GAT (Graph Attention Network)
Multi-head attention mechanism for aggregating neighbor information.

### HGCN (Hypergraph Convolutional Network)
Hypergraph convolutions for modeling higher-order relationships.

### Traditional Baselines
- **Logistic Regression**: L2-regularized logistic regression on gene expression
- **Random Forest**: Ensemble tree-based classifier

## Training Strategy

1. **Stratified Group K-Fold**: Stratify by condition (case/control), group by patient/batch
2. **Co-expression edges**: Computed per-fold on training data only to prevent leakage
3. **Early stopping**: Patience=30 on validation accuracy
4. **Domain adversarial**: λ_dann=0.1 for domain loss weight