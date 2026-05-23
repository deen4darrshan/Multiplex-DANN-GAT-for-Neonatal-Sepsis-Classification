"""
A1 Experiment: Pathway-Hypergraph Convolutional Network (HGCN) for Neonatal Sepsis
===================================================================================
Architecture: HypergraphConv (KEGG pathway hyperedges) + ComBat normalization, NO DANN.
This is the PRIMARY experiment from the CH-DANN plan.

Pipeline:
  1. Load ComBat-corrected expression + metadata
  2. Separate training (GSE25504 + GSE69686) from external holdout (GSE26440)
  3. MAD-filter top 2000 genes
  4. Fetch KEGG pathway gene sets (via gseapy or bundled fallback)
  5. Build hyperedge incidence matrix
  6. Construct per-patient HypergraphData objects
  7. Train HGCN with 5-fold stratified CV
  8. Save best model weights + results

CoVe checkpoints are printed at each stage.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Resolve project root (two levels up from CH_DANN_Plan/scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from scipy.stats import median_abs_deviation
import numpy as np
import pandas as pd
import pickle
import json
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_DIR = os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "results")
MODEL_DIR = os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters (from plan Section 6.1)
TOP_K_GENES = 2000
HIDDEN_CHANNELS = 64
NUM_LAYERS = 2
DROPOUT = 0.5
BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 20
N_SPLITS = 5
EDGE_DROP_RATE = 0.1
NOISE_STD = 0.05
STRING_THRESHOLD = 700

# Known sepsis biomarkers (Tier 1) for CoVe validation
TIER1_BIOMARKERS = [
    'FCGR1A', 'MMP9', 'S100A8', 'S100A9', 'TLR4',
    'MYD88', 'IL6', 'CXCL8', 'MPO', 'CEACAM8'
]

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
def load_data():
    """Load ComBat-corrected expression and metadata."""
    print("=" * 60)
    print("STEP 1: Loading ComBat-corrected data")
    print("=" * 60)

    expr = pd.read_csv(os.path.join(DATA_DIR, "combined_expression.csv"), index_col=0)
    meta = pd.read_csv(os.path.join(DATA_DIR, "combined_metadata.csv"))

    print(f"  Expression shape: {expr.shape} (genes x samples)")
    print(f"  Metadata shape:   {meta.shape}")
    print(f"  Batches:          {meta['Batch'].value_counts().to_dict()}")
    print(f"  Conditions:       {meta['Condition'].value_counts().to_dict()}")

    # CoVe: Basic sanity
    assert expr.shape[1] == len(meta), f"Column mismatch: {expr.shape[1]} vs {len(meta)}"
    assert set(meta['Condition'].unique()) <= {'Sepsis', 'Control', 'Unknown'}, \
        f"Unexpected conditions: {meta['Condition'].unique()}"
    print("  ✓ CoVe PASS: Data loaded and shapes match.")
    return expr, meta


# ============================================================================
# STEP 2: SPLIT TRAINING vs EXTERNAL HOLDOUT
# ============================================================================
def split_train_external(expr, meta):
    """Separate training samples from GSE26440 external holdout."""
    print("\n" + "=" * 60)
    print("STEP 2: Splitting Training vs External Holdout")
    print("=" * 60)

    external_mask = meta['Batch'] == 'GSE26440_Neo'
    train_mask = ~external_mask

    # Also remove 'Unknown' condition samples
    known_mask = meta['Condition'].isin(['Sepsis', 'Control'])
    train_mask = train_mask & known_mask

    train_samples = meta.loc[train_mask, 'SampleID'].tolist()
    ext_samples = meta.loc[external_mask & known_mask, 'SampleID'].tolist()

    # Ensure sample IDs are in expression columns
    train_samples = [s for s in train_samples if s in expr.columns]
    ext_samples = [s for s in ext_samples if s in expr.columns]

    expr_train = expr[train_samples]
    expr_ext = expr[ext_samples] if ext_samples else None

    meta_train = meta[meta['SampleID'].isin(train_samples)].reset_index(drop=True)
    meta_ext = meta[meta['SampleID'].isin(ext_samples)].reset_index(drop=True) if ext_samples else None

    print(f"  Training samples:  {len(train_samples)}")
    print(f"  External samples:  {len(ext_samples)}")
    if len(train_samples) > 0:
        print(f"  Train conditions:  {meta_train['Condition'].value_counts().to_dict()}")
        print(f"  Train batches:     {meta_train['Batch'].value_counts().to_dict()}")

    # CoVe
    assert len(train_samples) >= 250, f"Too few training samples: {len(train_samples)}"
    print("  ✓ CoVe PASS: Train/External split complete.")
    return expr_train, meta_train, expr_ext, meta_ext


# ============================================================================
# STEP 3: MAD VARIANCE FILTERING
# ============================================================================
def variance_filter(expr, top_k=TOP_K_GENES):
    """Select top-K genes by Median Absolute Deviation."""
    print("\n" + "=" * 60)
    print(f"STEP 3: MAD Variance Filtering (Top {top_k})")
    print("=" * 60)

    mad_scores = expr.apply(median_abs_deviation, axis=1)
    mad_scores = mad_scores.sort_values(ascending=False)

    top_genes = mad_scores.head(top_k).index.tolist()
    expr_filtered = expr.loc[top_genes]

    # CoVe: Check biomarker coverage
    biomarkers_found = [g for g in TIER1_BIOMARKERS if g in top_genes]
    print(f"  Selected {len(top_genes)} genes by MAD")
    print(f"  Tier 1 biomarkers found: {len(biomarkers_found)}/10 — {biomarkers_found}")
    missing = [g for g in TIER1_BIOMARKERS if g not in top_genes]
    if missing:
        print(f"  Missing biomarkers: {missing}")

    assert len(top_genes) == top_k, f"Expected {top_k} genes, got {len(top_genes)}"
    print(f"  ✓ CoVe PASS: {top_k} genes selected. {len(biomarkers_found)}/10 biomarkers present.")
    return expr_filtered, top_genes


# ============================================================================
# STEP 4: BUILD KEGG PATHWAY HYPEREDGES
# ============================================================================
def build_kegg_hyperedges(gene_list):
    """Build hyperedge incidence from KEGG pathways.

    Strategy: Use gseapy to fetch KEGG gene sets. If gseapy is unavailable,
    fall back to a curated set of key immune/sepsis pathways.
    """
    print("\n" + "=" * 60)
    print("STEP 4: Building KEGG Pathway Hyperedges")
    print("=" * 60)

    gene_set = set(gene_list)
    pathway_dict = {}

    # Try gseapy first
    try:
        import gseapy as gp
        print("  Using gseapy to fetch KEGG gene sets...")
        kegg = gp.get_library("KEGG_2021_Human")
        for pathway_name, genes in kegg.items():
            overlap = list(set(genes) & gene_set)
            if len(overlap) >= 3:  # minimum 3 genes per hyperedge
                pathway_dict[pathway_name] = overlap
        print(f"  Fetched {len(kegg)} KEGG pathways from gseapy")
    except Exception as e:
        print(f"  gseapy failed ({e}), trying Reactome...")
        try:
            import gseapy as gp
            reactome = gp.get_library("Reactome_2022")
            for pathway_name, genes in reactome.items():
                overlap = list(set(genes) & gene_set)
                if len(overlap) >= 3:
                    pathway_dict[pathway_name] = overlap
            print(f"  Fetched {len(reactome)} Reactome pathways from gseapy")
        except Exception as e2:
            print(f"  Reactome also failed ({e2}), using curated fallback...")
            pathway_dict = _get_curated_pathways(gene_set)

    # Also add STRING PPI-based pairwise edges as fallback for isolated genes
    hyperedge_genes_flat = set()
    for genes in pathway_dict.values():
        hyperedge_genes_flat.update(genes)

    isolated = gene_set - hyperedge_genes_flat
    print(f"\n  Pathway hyperedges: {len(pathway_dict)}")
    print(f"  Genes covered by pathways: {len(hyperedge_genes_flat)}/{len(gene_set)}")
    print(f"  Isolated genes (no pathway): {len(isolated)}")

    # Build STRING fallback edges for isolated genes
    string_edges = _build_string_fallback(gene_list, isolated)

    # Print some pathway examples
    for name in list(pathway_dict.keys())[:5]:
        print(f"    {name}: {len(pathway_dict[name])} genes")

    # CoVe
    immune_pathways = [p for p in pathway_dict.keys()
                       if any(kw in p.lower() for kw in
                              ['toll', 'nfkb', 'neutrophil', 'innate', 'inflammat',
                               'complement', 'cytokine', 'chemokine', 'nod-like',
                               'sepsis', 'immune'])]
    print(f"\n  Immune-related pathways found: {len(immune_pathways)}")
    for p in immune_pathways[:10]:
        print(f"    • {p}")

    assert len(pathway_dict) >= 10, f"Too few pathways: {len(pathway_dict)}"
    print(f"  ✓ CoVe PASS: {len(pathway_dict)} pathway hyperedges built.")
    return pathway_dict, string_edges


def _get_curated_pathways(gene_set):
    """Fallback curated sepsis-relevant pathways."""
    curated = {
        "Toll-like receptor signaling": ['TLR4', 'TLR2', 'TLR1', 'MYD88', 'TIRAP',
            'IRAK1', 'IRAK4', 'TRAF6', 'NFKB1', 'NFKB2', 'RELA', 'MAP3K7',
            'MAPK14', 'JUN', 'FOS', 'IRF3', 'IRF7', 'IFNB1'],
        "NF-kB signaling": ['NFKB1', 'NFKB2', 'RELA', 'RELB', 'REL', 'IKBKB',
            'IKBKG', 'CHUK', 'TNFAIP3', 'BCL3', 'TRAF2', 'TRAF6'],
        "Neutrophil degranulation": ['MPO', 'MMP9', 'S100A8', 'S100A9', 'CEACAM8',
            'FCGR1A', 'ITGAM', 'ITGB2', 'FPR1', 'FPR2', 'CD14', 'LTF',
            'ELANE', 'CTSG', 'AZU1', 'DEFA1', 'DEFA3', 'CAMP'],
        "Cytokine signaling": ['IL6', 'IL1B', 'IL10', 'TNF', 'CXCL8', 'CCL2',
            'CCL3', 'CXCL10', 'IL1A', 'IL18', 'IFNG', 'CSF2', 'CSF3'],
        "Complement cascade": ['C3', 'C5', 'C1QA', 'C1QB', 'C1QC', 'CFB', 'CFD',
            'CFH', 'MASP1', 'MASP2', 'C4A', 'C4B', 'C2'],
        "JAK-STAT signaling": ['JAK1', 'JAK2', 'JAK3', 'TYK2', 'STAT1', 'STAT3',
            'STAT5A', 'STAT5B', 'STAT6', 'SOCS1', 'SOCS3'],
        "Apoptosis": ['CASP3', 'CASP8', 'CASP9', 'BCL2', 'BAX', 'BID', 'CYCS',
            'APAF1', 'FADD', 'FAS', 'TNFRSF1A', 'XIAP'],
        "MAPK signaling": ['MAPK1', 'MAPK3', 'MAPK14', 'MAP2K1', 'MAP2K2',
            'MAP3K1', 'MAP3K7', 'RAF1', 'BRAF', 'KRAS', 'HRAS'],
        "Oxidative phosphorylation": ['NDUFA1', 'NDUFS1', 'SDHA', 'SDHB',
            'UQCRC1', 'COX5A', 'ATP5F1A', 'ATP5F1B', 'ATP5MC1'],
        "Coagulation cascade": ['F2', 'F5', 'F7', 'F10', 'F12', 'SERPINC1',
            'PROC', 'PROS1', 'THBD', 'VWF', 'FGA', 'FGB', 'FGG'],
    }
    result = {}
    for name, genes in curated.items():
        overlap = [g for g in genes if g in gene_set]
        if len(overlap) >= 3:
            result[name] = overlap
    return result


def _build_string_fallback(gene_list, isolated_genes):
    """Load STRING edges for isolated genes (fallback connectivity)."""
    ppi_path = os.path.join(DATA_DIR, "ppi_network.csv")
    if not os.path.exists(ppi_path):
        print("  WARNING: No PPI network file found for fallback edges.")
        return []

    ppi = pd.read_csv(ppi_path)
    gene_set = set(gene_list)
    # Filter to edges within our gene set AND at the right threshold
    ppi_filtered = ppi[(ppi['source'].isin(gene_set)) &
                       (ppi['target'].isin(gene_set)) &
                       (ppi['score'] >= STRING_THRESHOLD)]

    edges = list(zip(ppi_filtered['source'].tolist(), ppi_filtered['target'].tolist()))
    print(f"  STRING fallback edges (score≥{STRING_THRESHOLD}): {len(edges)}")
    return edges


# ============================================================================
# STEP 5: BUILD PATIENT HYPERGRAPH DATA OBJECTS
# ============================================================================
def build_patient_graphs(expr, meta, gene_list, pathway_dict, string_edges):
    """Build PyG Data objects with hyperedge_index for each patient."""
    print("\n" + "=" * 60)
    print("STEP 5: Building Patient Hypergraph Data Objects")
    print("=" * 60)

    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    num_nodes = len(gene_list)

    # Build hyperedge_index: [2, num_connections]
    # Format: hyperedge_index[0] = node indices, hyperedge_index[1] = hyperedge indices
    node_indices = []
    hedge_indices = []
    hedge_names = []

    for hedge_id, (pathway_name, genes) in enumerate(pathway_dict.items()):
        for gene in genes:
            if gene in gene_to_idx:
                node_indices.append(gene_to_idx[gene])
                hedge_indices.append(hedge_id)
        hedge_names.append(pathway_name)

    num_hedges = len(pathway_dict)

    # Add STRING pairwise edges as additional tiny hyperedges (size=2)
    for src, tgt in string_edges:
        if src in gene_to_idx and tgt in gene_to_idx:
            hedge_id = num_hedges
            node_indices.append(gene_to_idx[src])
            hedge_indices.append(hedge_id)
            node_indices.append(gene_to_idx[tgt])
            hedge_indices.append(hedge_id)
            num_hedges += 1

    hyperedge_index = torch.tensor([node_indices, hedge_indices], dtype=torch.long)

    print(f"  Hyperedge index shape: {hyperedge_index.shape}")
    print(f"  Pathway hyperedges: {len(pathway_dict)}")
    print(f"  STRING pair hyperedges: {num_hedges - len(pathway_dict)}")
    print(f"  Total hyperedges: {num_hedges}")

    # Build per-patient Data objects
    data_list = []
    label_map = {'Control': 0, 'Sepsis': 1}

    for _, row in meta.iterrows():
        sample_id = row['SampleID']
        condition = row['Condition']
        if condition not in label_map:
            continue
        if sample_id not in expr.columns:
            continue

        # Node features: expression values for this patient
        x = torch.tensor(expr[sample_id].values, dtype=torch.float32).unsqueeze(1)  # [N, 1]
        y = torch.tensor(label_map[condition], dtype=torch.long)

        data = Data(x=x, y=y)
        data.hyperedge_index = hyperedge_index.clone()
        data.num_nodes = num_nodes
        data.sample_id = sample_id
        data.batch_label = row['Batch']
        data_list.append(data)

    print(f"  Built {len(data_list)} patient graphs")
    labels = [d.y.item() for d in data_list]
    print(f"  Class distribution: Control={labels.count(0)}, Sepsis={labels.count(1)}")

    # CoVe
    assert len(data_list) >= 250, f"Too few graphs: {len(data_list)}"
    assert data_list[0].x.shape == (num_nodes, 1), \
        f"Wrong feature shape: {data_list[0].x.shape}"

    # Verify features vary across patients
    x0 = data_list[0].x.squeeze()
    x1 = data_list[1].x.squeeze()
    diff = (x0 - x1).abs().mean().item()
    print(f"  Mean abs feature diff between patient 0 & 1: {diff:.4f}")
    assert diff > 0.01, "Features are nearly identical across patients!"

    print(f"  ✓ CoVe PASS: {len(data_list)} patient hypergraphs built.")
    return data_list, hedge_names


# ============================================================================
# STEP 6: MODEL DEFINITION
# ============================================================================
class HypergraphSepsisNet(nn.Module):
    """2-layer Hypergraph Convolutional Network for Sepsis Classification.

    Architecture (from plan Section 5.1):
        HypergraphConv(1 → 64) + BN + LeakyReLU + Dropout
        HypergraphConv(64 → 64) + BN + LeakyReLU + Dropout
        GlobalMeanPool ⊕ GlobalMaxPool → 128
        Linear(128 → 64) + LeakyReLU + Dropout
        Linear(64 → 2)
    """
    def __init__(self, in_channels=1, hidden_channels=64, num_classes=2, dropout=0.5):
        super().__init__()
        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),  # mean+max pool
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
        self.dropout = dropout

    def forward(self, x, hyperedge_index, batch):
        # Layer 1
        x = self.conv1(x, hyperedge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, hyperedge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Pooling: mean + max
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Classifier
        x = self.classifier(x)
        return x


# ============================================================================
# STEP 7: TRAINING LOOP
# ============================================================================
def augment_data(data, hedge_drop_rate=EDGE_DROP_RATE, noise_std=NOISE_STD):
    """On-the-fly data augmentation: hyperedge dropout + feature noise."""
    data = data.clone()

    # Hyperedge dropout: randomly remove entire hyperedges
    if hedge_drop_rate > 0 and data.hyperedge_index.size(1) > 0:
        unique_hedges = data.hyperedge_index[1].unique()
        keep_mask = torch.rand(unique_hedges.max().item() + 1) > hedge_drop_rate
        # Keep connections whose hyperedge_id passes the mask
        col_mask = keep_mask[data.hyperedge_index[1]]
        data.hyperedge_index = data.hyperedge_index[:, col_mask]

    # Feature noise
    if noise_std > 0:
        data.x = data.x + torch.randn_like(data.x) * noise_std

    return data


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = augment_data(data)
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.hyperedge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * data.y.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    total_loss = 0

    criterion = nn.CrossEntropyLoss()

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.hyperedge_index, data.batch)
        loss = criterion(out, data.y)
        total_loss += loss.item() * data.y.size(0)

        probs = F.softmax(out, dim=1)[:, 1]
        preds = out.argmax(dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    # Handle edge case: all same class
    if len(set(all_labels)) < 2:
        auroc = 0.5
    else:
        auroc = roc_auc_score(all_labels, all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return auroc, acc, f1, avg_loss


def train_fold(model, train_loader, val_loader, fold, device):
    """Train a single fold with early stopping."""
    # Class weights for imbalanced data
    train_labels = []
    for d in train_loader.dataset:
        train_labels.append(d.y.item())
    n_control = train_labels.count(0)
    n_sepsis = train_labels.count(1)
    total = n_control + n_sepsis
    weight = torch.tensor([total / (2 * n_control), total / (2 * n_sepsis)],
                          dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=LEARNING_RATE,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=1)

    best_auroc = 0.0
    best_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        train_losses.append(train_loss)

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            val_auroc, val_acc, val_f1, val_loss = evaluate(model, val_loader, device)
            val_losses.append(val_loss)

            if epoch % 25 == 0 or epoch == 1:
                print(f"    Epoch {epoch:3d}: TrainLoss={train_loss:.4f} "
                      f"ValLoss={val_loss:.4f} ValAUROC={val_auroc:.4f} "
                      f"ValAcc={val_acc:.3f}")

            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE // 5:
                print(f"    Early stopping at epoch {epoch}")
                break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval on best model
    final_auroc, final_acc, final_f1, _ = evaluate(model, val_loader, device)

    return max(best_auroc, final_auroc), final_acc, final_f1, train_losses, val_losses


# ============================================================================
# STEP 8: MAIN - 5-FOLD CV
# ============================================================================
def main():
    start_time = time.time()

    print("=" * 60)
    print("A1 EXPERIMENT: Pathway-HGCN + ComBat (No DANN)")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Step 1-2: Load & split
    expr, meta = load_data()
    expr_train, meta_train, expr_ext, meta_ext = split_train_external(expr, meta)

    # Step 3: Variance filter
    expr_filtered, gene_list = variance_filter(expr_train, TOP_K_GENES)

    # Step 4: Build hyperedges
    pathway_dict, string_edges = build_kegg_hyperedges(gene_list)

    # Step 5: Build patient graphs
    data_list, hedge_names = build_patient_graphs(
        expr_filtered, meta_train, gene_list, pathway_dict, string_edges)

    # Save gene list and pathway info
    with open(os.path.join(OUT_DIR, "gene_list.json"), 'w') as f:
        json.dump(gene_list, f)
    with open(os.path.join(OUT_DIR, "pathway_info.json"), 'w') as f:
        json.dump({k: v for k, v in pathway_dict.items()}, f, indent=2)

    # Save patient graphs for reuse
    with open(os.path.join(OUT_DIR, "patient_graphs_hyper.pkl"), 'wb') as f:
        pickle.dump({'data_list': data_list, 'gene_list': gene_list,
                     'hedge_names': hedge_names}, f)
    print(f"\n  Saved patient graphs to {OUT_DIR}/patient_graphs_hyper.pkl")

    # ---- 5-FOLD CV ----
    print("\n" + "=" * 60)
    print("STEP 7: 5-Fold Stratified Cross-Validation")
    print("=" * 60)

    labels = np.array([d.y.item() for d in data_list])
    batches = np.array([d.batch_label for d in data_list])
    strat_key = np.array([f"{l}_{b}" for l, b in zip(labels, batches)])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    fold_results = []
    best_overall_auroc = 0
    best_overall_state = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(data_list)), strat_key)):
        print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
        print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")

        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]

        # Verify class balance in fold
        train_labels = [d.y.item() for d in train_data]
        val_labels = [d.y.item() for d in val_data]
        print(f"  Train classes: C={train_labels.count(0)} S={train_labels.count(1)}")
        print(f"  Val classes:   C={val_labels.count(0)} S={val_labels.count(1)}")

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

        model = HypergraphSepsisNet(
            in_channels=1,
            hidden_channels=HIDDEN_CHANNELS,
            num_classes=2,
            dropout=DROPOUT
        ).to(DEVICE)

        auroc, acc, f1, train_losses, val_losses = train_fold(
            model, train_loader, val_loader, fold, DEVICE)

        fold_results.append({
            'fold': fold + 1,
            'auroc': auroc,
            'accuracy': acc,
            'f1': f1
        })

        # Save fold model weights
        fold_model_path = os.path.join(MODEL_DIR, f"hgcn_fold{fold+1}.pt")
        torch.save(model.state_dict(), fold_model_path)
        print(f"  Fold {fold+1} → AUROC={auroc:.4f} Acc={acc:.3f} F1={f1:.3f}")
        print(f"  Saved weights: {fold_model_path}")

        if auroc > best_overall_auroc:
            best_overall_auroc = auroc
            best_overall_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

    # Save best overall model
    best_model_path = os.path.join(MODEL_DIR, "hgcn_best.pt")
    if best_overall_state:
        torch.save(best_overall_state, best_model_path)
        print(f"\nSaved best model to {best_model_path} (AUROC={best_overall_auroc:.4f})")

    # ---- RESULTS SUMMARY ----
    print("\n" + "=" * 60)
    print("STEP 8: A1 EXPERIMENT RESULTS")
    print("=" * 60)

    aurocs = [r['auroc'] for r in fold_results]
    accs = [r['accuracy'] for r in fold_results]
    f1s = [r['f1'] for r in fold_results]

    print(f"\n  {'Fold':<6} {'AUROC':<10} {'Accuracy':<10} {'F1':<10}")
    print(f"  {'-'*36}")
    for r in fold_results:
        star = " ★" if r['auroc'] == max(aurocs) else ""
        print(f"  {r['fold']:<6} {r['auroc']:<10.4f} {r['accuracy']:<10.3f} {r['f1']:<10.3f}{star}")
    print(f"  {'-'*36}")
    print(f"  {'Mean':<6} {np.mean(aurocs):<10.4f} {np.mean(accs):<10.3f} {np.mean(f1s):<10.3f}")
    print(f"  {'Std':<6} {np.std(aurocs):<10.4f} {np.std(accs):<10.3f} {np.std(f1s):<10.3f}")

    print(f"\n  Comparison benchmarks:")
    print(f"    Phase 2 GCN baseline:     AUROC = 0.681 ± 0.048")
    print(f"    Phase 2 GCN (optimized):  AUROC = 0.685 ± 0.091")
    print(f"    LR baseline:              AUROC = 0.816")

    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed/60:.1f} minutes")

    # Save results
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(OUT_DIR, "a1_hgcn_results.csv"), index=False)

    summary = {
        'experiment': 'A1_HGCN_ComBat',
        'mean_auroc': float(np.mean(aurocs)),
        'std_auroc': float(np.std(aurocs)),
        'mean_accuracy': float(np.mean(accs)),
        'mean_f1': float(np.mean(f1s)),
        'best_fold_auroc': float(max(aurocs)),
        'worst_fold_auroc': float(min(aurocs)),
        'num_genes': len(gene_list),
        'num_pathway_hedges': len(pathway_dict),
        'num_samples': len(data_list),
        'epochs': EPOCHS,
        'hidden_channels': HIDDEN_CHANNELS,
        'dropout': DROPOUT,
        'elapsed_minutes': elapsed / 60
    }
    with open(os.path.join(OUT_DIR, "a1_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {OUT_DIR}/")
    print(f"  Models saved to {MODEL_DIR}/")

    # CoVe final gate
    print("\n" + "=" * 60)
    print("CoVe FINAL GATE")
    print("=" * 60)
    mean_auroc = np.mean(aurocs)
    if mean_auroc >= 0.78:
        print(f"  ✓ PASS: Mean AUROC {mean_auroc:.4f} >= 0.78 target")
    elif mean_auroc >= 0.68:
        print(f"  ⚠ PARTIAL: Mean AUROC {mean_auroc:.4f} >= 0.68 (matches Phase 2 GCN)")
        print(f"    Hypergraph is competitive but not yet beating baselines.")
    else:
        print(f"  ✗ BELOW BASELINE: Mean AUROC {mean_auroc:.4f} < 0.68")
        print(f"    Consider: reduce complexity, increase regularization, or pivot strategy.")

    if np.std(aurocs) < 0.05:
        print(f"  ✓ Stability: Std {np.std(aurocs):.4f} < 0.05 (stable)")
    elif np.std(aurocs) < 0.10:
        print(f"  ⚠ Moderate variance: Std {np.std(aurocs):.4f}")
    else:
        print(f"  ✗ High variance: Std {np.std(aurocs):.4f} — model is unstable")


if __name__ == "__main__":
    main()
