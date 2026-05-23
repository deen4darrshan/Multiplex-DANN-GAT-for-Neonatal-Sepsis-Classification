"""
GAT Training with Expanded Graphs

Uses expanded patient graphs (3,498 nodes, 125K edges) for better attention-based learning.
Includes edge dropout augmentation and optimized hyperparameters for larger graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm

# Paths
GRAPH_FILE = "data/graphs/patient_graphs_expanded.pkl"  # Use expanded graphs
OUT_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# OPTIMIZED Hyperparameters for Expanded Graphs
HIDDEN_CHANNELS = 128      # Increased for larger graph
NUM_HEADS = 4              # More heads for attention diversity
NUM_LAYERS = 2
DROPOUT = 0.6
EDGE_DROPOUT = 0.1         # Data augmentation
FEATURE_NOISE = 0.05       # Slight noise for regularization
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
N_SPLITS = 5

print(f"Using device: {device}")
print(f"\n{'='*60}")
print("=== GAT TRAINING WITH EXPANDED GRAPHS ===")
print(f"{'='*60}")
print(f"Hidden Channels: {HIDDEN_CHANNELS}")
print(f"Attention Heads: {NUM_HEADS}")
print(f"Layers: {NUM_LAYERS}")
print(f"Dropout: {DROPOUT}")
print(f"Edge Dropout: {EDGE_DROPOUT}")
print(f"Epochs: {EPOCHS}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"{'='*60}\n")


class ExpandedGAT(nn.Module):
    """GAT architecture optimized for expanded graph size."""
    
    def __init__(self, in_channels, hidden_channels, num_classes=2, heads=4, dropout=0.6):
        super(ExpandedGAT, self).__init__()
        
        # Layer 1: Multi-head attention
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, 
                             dropout=0.0, add_self_loops=True)
        
        # Layer 2: Reduce heads for efficiency
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, 
                             heads=2, concat=False, dropout=0.0, add_self_loops=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch):
        # Input dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT Layer 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x


def load_graphs():
    """Load expanded patient graphs."""
    print(f"Loading graphs from {GRAPH_FILE}...")
    with open(GRAPH_FILE, 'rb') as f:
        data = pickle.load(f)
    data_list = data['data_list']
    config = data.get('config', {})
    
    print(f"Loaded {len(data_list)} graphs")
    print(f"  - Nodes: {data_list[0].x.shape[0]}")
    print(f"  - Features: {data_list[0].x.shape[1]}")
    print(f"  - Edges: {data_list[0].edge_index.shape[1] // 2}")
    if config:
        print(f"  - Config: {config}")
    
    return data_list


def train_epoch(model, loader, optimizer, criterion):
    """Train one epoch with edge dropout augmentation."""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Data Augmentation: Edge Dropout
        edge_index, _ = dropout_edge(data.edge_index, p=EDGE_DROPOUT, 
                                     force_undirected=True, training=True)
        
        # Data Augmentation: Feature Noise
        x = data.x
        if FEATURE_NOISE > 0:
            noise = torch.randn_like(x) * FEATURE_NOISE
            x = x + noise
        
        out = model(x, edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    """Evaluate model on loader."""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        probs = F.softmax(out, dim=1)[:, 1]
        preds = out.argmax(dim=1)
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
    
    auroc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    return auroc, acc


def train_cv(data_list):
    """5-fold cross-validation training."""
    labels = np.array([d.y.item() for d in data_list])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    fold_aurocs, fold_accs = [], []
    best_overall_auroc = 0
    best_overall_model_state = None
    
    print(f"\n{'='*60}")
    print("Training GAT with 5-Fold Cross-Validation")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(data_list)), labels)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
        
        in_channels = data_list[0].x.shape[1]
        model = ExpandedGAT(in_channels, HIDDEN_CHANNELS, num_classes=2, 
                            heads=NUM_HEADS, dropout=DROPOUT).to(device)
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if fold == 0:
            print(f"Model parameters: {num_params:,}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, 
                                       weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.CrossEntropyLoss()
        
        best_auroc = 0
        best_model_state = None
        
        pbar = tqdm(range(1, EPOCHS + 1), desc=f"Fold {fold+1}", ncols=100)
        for epoch in pbar:
            loss = train_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()
            
            if epoch % 10 == 0 or epoch == EPOCHS:
                auroc, acc = evaluate(model, val_loader)
                pbar.set_postfix({'loss': f'{loss:.4f}', 'auroc': f'{auroc:.3f}', 
                                  'acc': f'{acc:.3f}'})
                
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_model_state = model.state_dict().copy()
        
        # Save fold model
        fold_model_path = os.path.join(MODEL_DIR, f"gat_expanded_fold{fold+1}.pt")
        torch.save(best_model_state, fold_model_path)
        print(f"Saved fold {fold+1} best model (AUROC={best_auroc:.4f})")
        
        # Track overall best
        if best_auroc > best_overall_auroc:
            best_overall_auroc = best_auroc
            best_overall_model_state = best_model_state.copy()
        
        fold_aurocs.append(best_auroc)
        _, final_acc = evaluate(model, val_loader)
        fold_accs.append(final_acc)
        print(f"Fold {fold+1} Best AUROC: {best_auroc:.4f}")
    
    # Save overall best model
    best_model_path = os.path.join(MODEL_DIR, "gat_expanded_best.pt")
    torch.save({
        'model_state_dict': best_overall_model_state,
        'auroc': best_overall_auroc,
        'config': {
            'in_channels': data_list[0].x.shape[1],
            'hidden_channels': HIDDEN_CHANNELS,
            'heads': NUM_HEADS,
            'num_classes': 2,
            'dropout': DROPOUT
        }
    }, best_model_path)
    print(f"\n✓ Saved overall best model (AUROC={best_overall_auroc:.4f}) to {best_model_path}")
    
    mean_auroc = np.mean(fold_aurocs)
    std_auroc = np.std(fold_aurocs)
    mean_acc = np.mean(fold_accs)
    
    return mean_auroc, std_auroc, mean_acc, fold_aurocs, best_overall_auroc


def main():
    data_list = load_graphs()
    gat_auroc, gat_std, gat_acc, fold_aurocs, best_auroc = train_cv(data_list)
    
    # Save results
    results = pd.DataFrame({
        'Model': ['GAT_Expanded'],
        'AUROC': [gat_auroc],
        'Std': [gat_std],
        'Accuracy': [gat_acc],
        'Best_Fold_AUROC': [best_auroc],
        'Nodes': [3498],
        'Edges': [125288]
    })
    results.to_csv(os.path.join(OUT_DIR, "gat_expanded_results.csv"), index=False)
    
    print(f"\n{'='*60}")
    print("=== GAT TRAINING COMPLETE ===")
    print(f"{'='*60}")
    print(f"Mean AUROC: {gat_auroc:.4f} (+/- {gat_std:.4f})")
    print(f"Mean Accuracy: {gat_acc*100:.1f}%")
    print(f"Best Fold AUROC: {best_auroc:.4f}")
    print(f"Per-fold AUROCs: {[f'{x:.4f}' for x in fold_aurocs]}")
    print(f"\nModel weights saved to: {MODEL_DIR}/")
    print(f"  - gat_expanded_fold1.pt through gat_expanded_fold5.pt")
    print(f"  - gat_expanded_best.pt (overall best)")
    
    # Comparison
    print(f"\n{'='*60}")
    print("=== COMPARISON ===")
    print(f"{'='*60}")
    print(f"Previous GCN (1,491 nodes): 0.6851 ± 0.0914")
    print(f"GAT Expanded (3,498 nodes): {gat_auroc:.4f} ± {gat_std:.4f}")
    print(f"LR Baseline:                0.8164 ± 0.0744")


if __name__ == "__main__":
    main()
