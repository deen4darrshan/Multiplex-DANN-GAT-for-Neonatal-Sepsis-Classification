"""
GNN Optimized - Step 2: Train GCN

Train GCN with optimized hyperparameters and detailed per-epoch reporting.
Reports accuracy, loss, and AUROC every epoch with progress bar for every fold.

Configuration:
- Hidden channels: 64
- Layers: 3
- Dropout: 0.5
- Edge dropout: 10%
- Feature noise: 0.1 std
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
HIDDEN_CHANNELS = 64
NUM_LAYERS = 3
DROPOUT = 0.5
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
N_SPLITS = 5
EDGE_DROPOUT = 0.10
FEATURE_NOISE_STD = 0.1

# Paths
DATA_DIR = "data"
MODEL_DIR = "models"
LOG_DIR = "../logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("GNN OPTIMIZED - GCN TRAINING")
print("=" * 70)
print(f"Device: {device}")
print(f"\nHyperparameters:")
print(f"  Hidden Channels: {HIDDEN_CHANNELS}")
print(f"  Num Layers: {NUM_LAYERS}")
print(f"  Dropout: {DROPOUT}")
print(f"  Edge Dropout: {EDGE_DROPOUT}")
print(f"  Feature Noise: {FEATURE_NOISE_STD}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Weight Decay: {WEIGHT_DECAY}")
print(f"  Epochs: {EPOCHS}")
print(f"  CV Folds: {N_SPLITS}")
print("=" * 70)


class OptimizedGCN(nn.Module):
    """3-Layer GCN with optimized architecture."""
    
    def __init__(self, in_channels, hidden_channels, num_classes, dropout):
        super().__init__()
        self.dropout = dropout
        
        # 3 GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels // 2)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels // 2)
        
        # Classifier
        self.fc = nn.Linear(hidden_channels // 2, num_classes)
    
    def forward(self, x, edge_index, batch):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classifier
        x = self.fc(x)
        return x


def load_graphs():
    """Load patient graphs."""
    print("\nLoading patient graphs...")
    with open(os.path.join(DATA_DIR, "patient_graphs_optimized.pkl"), 'rb') as f:
        data = pickle.load(f)
    
    data_list = data['data_list']
    config = data['config']
    
    print(f"  Loaded {len(data_list)} graphs")
    print(f"  Nodes: {config['final_nodes']}, Edges: {config['final_edges']}")
    
    return data_list, config


def add_feature_noise(data, std):
    """Add Gaussian noise to node features during training."""
    if std > 0:
        noise = torch.randn_like(data.x) * std
        data.x = data.x + noise
    return data


def train_epoch(model, loader, optimizer, criterion):
    """Train for one epoch with augmentation."""
    model.train()
    total_loss = 0
    all_probs, all_preds, all_labels = [], [], []
    
    for data in loader:
        data = data.to(device)
        
        # Data augmentation: edge dropout
        edge_index, _ = dropout_edge(data.edge_index, p=EDGE_DROPOUT, 
                                     force_undirected=True, training=True)
        
        # Data augmentation: feature noise
        data = add_feature_noise(data, FEATURE_NOISE_STD)
        
        optimizer.zero_grad()
        out = model(data.x, edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions for metrics
        probs = F.softmax(out.detach(), dim=1)[:, 1]
        preds = out.detach().argmax(dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
    
    # Calculate training metrics
    train_auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    train_acc = accuracy_score(all_labels, all_preds)
    
    return total_loss / len(loader), train_auroc, train_acc


@torch.no_grad()
def evaluate(model, loader):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_probs, all_preds, all_labels = [], [], []
    criterion = nn.CrossEntropyLoss()
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        total_loss += loss.item()
        
        probs = F.softmax(out, dim=1)[:, 1]
        preds = out.argmax(dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
    
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    acc = accuracy_score(all_labels, all_preds)
    
    return total_loss / len(loader), auroc, acc


def train_fold(fold, train_data, val_data, in_channels):
    """Train a single fold with detailed per-epoch reporting."""
    print(f"\n{'='*70}")
    print(f"FOLD {fold + 1}/{N_SPLITS}")
    print(f"{'='*70}")
    print(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples")
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    model = OptimizedGCN(in_channels, HIDDEN_CHANNELS, num_classes=2, dropout=DROPOUT).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, factor=0.5)
    
    best_auroc = 0
    best_model_state = None
    history = []
    
    pbar = tqdm(range(1, EPOCHS + 1), desc=f"Fold {fold+1}", ncols=100)
    for epoch in pbar:
        # Train
        train_loss, train_auroc, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        
        # Validate
        val_loss, val_auroc, val_acc = evaluate(model, val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_auroc)
        
        # Track best
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Log history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_auroc': train_auroc,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_auroc': val_auroc,
            'val_acc': val_acc
        })
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{val_loss:.3f}',
            'auroc': f'{val_auroc:.3f}',
            'acc': f'{val_acc:.3f}',
            'best': f'{best_auroc:.3f}'
        })
    
    print(f"\nFold {fold+1} Complete - Best Val AUROC: {best_auroc:.4f}")
    
    return best_auroc, val_acc, best_model_state, history


def main():
    # Load data
    data_list, config = load_graphs()
    
    labels = np.array([d.y.item() for d in data_list])
    in_channels = data_list[0].x.shape[1]
    
    print(f"\nClass distribution: Control={np.sum(labels==0)}, Sepsis={np.sum(labels==1)}")
    print(f"Input channels: {in_channels}")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    fold_aurocs = []
    fold_accs = []
    all_histories = []
    best_overall_auroc = 0
    best_overall_state = None
    best_fold = 0
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(data_list)), labels)):
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        
        auroc, acc, model_state, history = train_fold(fold, train_data, val_data, in_channels)
        
        fold_aurocs.append(auroc)
        fold_accs.append(acc)
        all_histories.append(history)
        
        # Save fold model
        torch.save(model_state, os.path.join(MODEL_DIR, f"gcn_fold{fold+1}.pt"))
        
        # Track overall best
        if auroc > best_overall_auroc:
            best_overall_auroc = auroc
            best_overall_state = model_state
            best_fold = fold + 1
    
    # Save best model
    torch.save({
        'model_state_dict': best_overall_state,
        'auroc': best_overall_auroc,
        'config': {
            'in_channels': in_channels,
            'hidden_channels': HIDDEN_CHANNELS,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT
        }
    }, os.path.join(MODEL_DIR, "gcn_best.pt"))
    
    # Results summary
    mean_auroc = np.mean(fold_aurocs)
    std_auroc = np.std(fold_aurocs)
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\n| Fold | AUROC | Accuracy |")
    print(f"|------|-------|----------|")
    for i, (auc, acc) in enumerate(zip(fold_aurocs, fold_accs)):
        marker = " ★" if i + 1 == best_fold else ""
        print(f"| {i+1}    | {auc:.4f} | {acc:.4f} |{marker}")
    
    print(f"\n| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| Mean AUROC | {mean_auroc:.4f} ± {std_auroc:.4f} |")
    print(f"| Mean Accuracy | {mean_acc:.4f} ± {std_acc:.4f} |")
    print(f"| Best Fold AUROC | {best_overall_auroc:.4f} (Fold {best_fold}) |")
    
    # Save results to markdown
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    results_md = f"""# GCN Optimized Results

**Date:** {timestamp}

## Configuration

| Parameter | Value |
|-----------|-------|
| Hidden Channels | {HIDDEN_CHANNELS} |
| Num Layers | {NUM_LAYERS} |
| Dropout | {DROPOUT} |
| Edge Dropout | {EDGE_DROPOUT} |
| Feature Noise | {FEATURE_NOISE_STD} |
| Learning Rate | {LEARNING_RATE} |
| Weight Decay | {WEIGHT_DECAY} |
| Batch Size | {BATCH_SIZE} |
| Epochs | {EPOCHS} |
| Nodes | {config['final_nodes']} |
| Edges | {config['final_edges']} |

## Results

| Fold | AUROC | Accuracy |
|------|-------|----------|
"""
    for i, (auc, acc) in enumerate(zip(fold_aurocs, fold_accs)):
        results_md += f"| {i+1} | {auc:.4f} | {acc:.4f} |\n"
    
    results_md += f"""
## Summary

| Metric | Value |
|--------|-------|
| **Mean AUROC** | **{mean_auroc:.4f} ± {std_auroc:.4f}** |
| Mean Accuracy | {mean_acc:.4f} ± {std_acc:.4f} |
| Best Fold | {best_overall_auroc:.4f} (Fold {best_fold}) |

## Comparison with Baselines

| Model | AUROC |
|-------|-------|
| GCN Optimized | {mean_auroc:.4f} |
| Previous GCN | 0.6812 |
| LR Baseline | 0.8164 |
"""
    
    with open(os.path.join(LOG_DIR, "gnn_optimization_results.md"), 'w') as f:
        f.write(results_md)
    
    print(f"\nResults saved to: {LOG_DIR}/gnn_optimization_results.md")
    print(f"Models saved to: {MODEL_DIR}/")


if __name__ == "__main__":
    main()
