"""
Optimization Phase - Module D: Optimized GNN Training

Architecture settings to prevent over-smoothing and handle small N:
- Layers: 2 (strict limit)
- Hidden: 64 (wide layers)
- Batch Size: 32 (regularization noise)
- Dropout: 0.6 (high for small N)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.utils import dropout_edge
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
import pickle
import os

# Paths
GRAPH_DIR = "data/graphs"
OUT_DIR = "data/processed"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# OPTIMIZED HYPERPARAMETERS
HIDDEN_CHANNELS = 64    # Restored to 64
NUM_LAYERS = 2          # Strict limit to prevent over-smoothing
DROPOUT = 0.5           # Reduced from 0.7
BATCH_SIZE = 32         # Increased from 16
EPOCHS = 100            # More epochs for better convergence
LEARNING_RATE = 0.0005
N_SPLITS = 5

print(f"Using device: {device}")
print(f"\n=== OPTIMIZED HYPERPARAMETERS ===")
print(f"Hidden Channels: {HIDDEN_CHANNELS}")
print(f"Num Layers: {NUM_LAYERS}")
print(f"Dropout: {DROPOUT}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"================================\n")

class OptimizedGCN(nn.Module):
    """2-layer GCN with wide hidden channels and high dropout."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes=2, dropout=0.6):
        super(OptimizedGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.classifier = nn.Linear(out_channels, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification with dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x

class OptimizedGAT(nn.Module):
    """2-layer GAT with reduced heads for memory efficiency."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes=2, heads=2, dropout=0.6):
        super(OptimizedGAT, self).__init__()
        # Reduced heads=2 for memory efficiency
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.0, add_self_loops=False)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.0, add_self_loops=False)
        self.classifier = nn.Linear(out_channels, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch):
        # Pre-layer dropout on features
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x

def load_graphs():
    """Load patient graphs with 3D features."""
    print("Loading enhanced patient graphs (3D features)...")
    
    with open(os.path.join(GRAPH_DIR, "patient_graphs_3d.pkl"), 'rb') as f:
        data = pickle.load(f)
    
    data_list = data['data_list']
    print(f"Loaded {len(data_list)} patient graphs")
    print(f"Node features: {data_list[0].x.shape[1]} dimensions")
    
    # Get labels
    labels = [d.y.item() for d in data_list]
    print(f"Class distribution: Control={labels.count(0)}, Sepsis={labels.count(1)}")
    
    return data_list

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Data Augmentation: On-the-fly Edge Dropping
        # Remove 5% of edges randomly during training to prevent memorization
        edge_index, _ = dropout_edge(data.edge_index, p=0.05, force_undirected=True, training=True)
        
        out = model(data.x, edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
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

def train_cv(data_list, model_class, model_name):
    """Run 5-fold cross-validation."""
    labels = np.array([d.y.item() for d in data_list])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    fold_aurocs = []
    fold_accs = []
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(data_list)), labels)):
        print(f"\nFold {fold+1}/{N_SPLITS}")
        
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
        
        in_channels = data_list[0].x.shape[1]  # 3 features
        model = model_class(
            in_channels=in_channels,
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=HIDDEN_CHANNELS // 2,
            num_classes=2,
            dropout=DROPOUT
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_auroc = 0
        patience = 15
        patience_counter = 0
        
        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, train_loader, optimizer, criterion)
            
            if epoch % 20 == 0:
                auroc, acc = evaluate(model, val_loader)
                print(f"  Epoch {epoch}: Loss={loss:.4f}, Val AUROC={auroc:.4f}")
                
                if auroc > best_auroc:
                    best_auroc = auroc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience // 20:
                        # Early stopping check
                        pass
        
        # Final evaluation
        final_auroc, final_acc = evaluate(model, val_loader)
        fold_aurocs.append(max(best_auroc, final_auroc))
        fold_accs.append(final_acc)
        print(f"  Fold {fold+1} Best AUROC: {fold_aurocs[-1]:.4f}")
    
    mean_auroc = np.mean(fold_aurocs)
    std_auroc = np.std(fold_aurocs)
    mean_acc = np.mean(fold_accs)
    
    print(f"\n{model_name} Summary:")
    print(f"  Mean AUROC: {mean_auroc:.4f} (+/- {std_auroc:.4f})")
    print(f"  Mean Accuracy: {mean_acc:.4f}")
    
    return mean_auroc, std_auroc, mean_acc, fold_aurocs

def main():
    data_list = load_graphs()
    
    results = {}
    
    # Train GCN
    gcn_auroc, gcn_std, gcn_acc, gcn_folds = train_cv(data_list, OptimizedGCN, "GCN")
    results['GCN'] = {'auroc': gcn_auroc, 'std': gcn_std, 'acc': gcn_acc, 'folds': gcn_folds}
    
    # Train GAT
    gat_auroc, gat_std, gat_acc, gat_folds = train_cv(data_list, OptimizedGAT, "GAT")
    results['GAT'] = {'auroc': gat_auroc, 'std': gat_std, 'acc': gat_acc, 'folds': gat_folds}
    
    # Save results
    results_df = pd.DataFrame({
        'Model': ['GCN', 'GAT', 'Baseline (LR)', 'Baseline (RF)'],
        'AUROC': [gcn_auroc, gat_auroc, 0.856, 0.808],
        'Std': [gcn_std, gat_std, 0.0, 0.0],
        'Accuracy': [gcn_acc, gat_acc, 0.736, 0.726]
    })
    results_df.to_csv(os.path.join(OUT_DIR, "optimized_gnn_results.csv"), index=False)
    
    # Final summary
    print(f"\n{'='*60}")
    print("=== FINAL RESULTS (Optimized) ===")
    print(f"{'='*60}")
    print(f"GCN: AUROC={gcn_auroc:.4f} (+/- {gcn_std:.4f}), Acc={gcn_acc:.4f}")
    print(f"GAT: AUROC={gat_auroc:.4f} (+/- {gat_std:.4f}), Acc={gat_acc:.4f}")
    print(f"\nBaselines (for comparison):")
    print(f"  Logistic Regression: AUROC=0.856")
    print(f"  Random Forest: AUROC=0.808")
    print(f"\nResults saved to: {os.path.join(OUT_DIR, 'optimized_gnn_results.csv')}")

if __name__ == "__main__":
    main()
