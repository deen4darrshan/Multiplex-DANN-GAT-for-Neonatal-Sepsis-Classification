"""
Module D - Task D.2: GNN Implementation (GCN & GAT)

Implements Graph Convolutional Network and Graph Attention Network
for graph-level classification of patient sepsis status.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import numpy as np
import pickle
import os

# Paths
DATA_DIR = "data/graphs"
OUT_DIR = "data/processed"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class GCN(nn.Module):
    """Graph Convolutional Network for graph classification."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes=2, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.classifier = nn.Linear(out_channels, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch):
        # Convolution layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling (graph-level representation)
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x

class GAT(nn.Module):
    """Graph Attention Network for graph classification."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes=2, heads=4, dropout=0.5):
        super(GAT, self).__init__()
        # Remove dropout from GATConv to avoid dimension mismatch issues
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.0)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.0)
        self.classifier = nn.Linear(out_channels, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch):
        # Apply dropout to node features instead
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Attention layers
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = self.classifier(x)
        return x

def load_graphs():
    """Load patient graphs."""
    print("Loading patient graphs...")
    with open(os.path.join(DATA_DIR, "patient_graphs.pkl"), 'rb') as f:
        data = pickle.load(f)
    
    data_list = data['data_list']
    
    # Filter to only Sepsis (1) and Control (0) - exclude Unknown (2)
    filtered = [d for d in data_list if d.y.item() != 2]
    print(f"Filtered data: {len(filtered)} graphs (excluding Unknown)")
    
    return filtered

def train_epoch(model, loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    """Evaluate model."""
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            prob = F.softmax(out, dim=1)[:, 1]
            
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
    
    auroc = roc_auc_score(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)
    
    return auroc, accuracy

def train_cv(data_list, model_class, model_name, hidden_channels=64, epochs=100, lr=0.001, n_splits=5):
    """Train with cross-validation."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Get labels for stratification
    labels = [d.y.item() for d in data_list]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_aurocs = []
    fold_accs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(data_list)), labels)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16, shuffle=False)
        
        # Get input dimension
        in_channels = data_list[0].x.shape[1]
        
        # Initialize model
        if model_class == GAT:
            model = GAT(in_channels, hidden_channels, hidden_channels // 2, num_classes=2).to(device)
        else:
            model = GCN(in_channels, hidden_channels, hidden_channels // 2, num_classes=2).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_auroc = 0
        for epoch in range(epochs):
            loss = train_epoch(model, train_loader, optimizer, criterion)
            auroc, acc = evaluate(model, val_loader)
            
            if auroc > best_auroc:
                best_auroc = auroc
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Val AUROC={auroc:.4f}")
        
        fold_aurocs.append(best_auroc)
        fold_accs.append(acc)
        print(f"  Fold {fold+1} Best AUROC: {best_auroc:.4f}")
    
    print(f"\n{model_name} Summary:")
    print(f"  Mean AUROC: {np.mean(fold_aurocs):.4f} (+/- {np.std(fold_aurocs):.4f})")
    print(f"  Mean Accuracy: {np.mean(fold_accs):.4f}")
    
    return fold_aurocs, fold_accs

def main():
    # Load data
    data_list = load_graphs()
    
    results = {}
    
    # 1. Train GCN
    gcn_aurocs, gcn_accs = train_cv(data_list, GCN, "GCN", hidden_channels=32, epochs=50)
    results['GCN'] = {'auroc': np.mean(gcn_aurocs), 'acc': np.mean(gcn_accs)}
    
    # 2. Train GAT
    gat_aurocs, gat_accs = train_cv(data_list, GAT, "GAT", hidden_channels=32, epochs=50)
    results['GAT'] = {'auroc': np.mean(gat_aurocs), 'acc': np.mean(gat_accs)}
    
    # Summary
    print(f"\n{'='*60}")
    print("=== GNN RESULTS SUMMARY ===")
    print(f"{'='*60}")
    
    for model_name, metrics in results.items():
        print(f"{model_name}: AUROC={metrics['auroc']:.4f}, Accuracy={metrics['acc']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(OUT_DIR, "gnn_results.csv"))
    
    # Verification
    print(f"\n{'='*60}")
    print("=== VERIFICATION (CoVe) ===")
    print(f"{'='*60}")
    
    target_auroc = 0.78
    best_auroc = max([m['auroc'] for m in results.values()])
    
    if best_auroc >= target_auroc:
        print(f"✓ PASS: Best GNN AUROC >= {target_auroc} ({best_auroc:.4f})")
    else:
        print(f"⚠ WARNING: Best GNN AUROC < {target_auroc} ({best_auroc:.4f})")
    
    # Compare to baseline
    try:
        baseline = pd.read_csv(os.path.join(OUT_DIR, "baseline_results.csv"), index_col=0)
        best_baseline = baseline['auroc'].max()
        improvement = best_auroc - best_baseline
        print(f"Baseline best AUROC: {best_baseline:.4f}")
        print(f"GNN improvement: {improvement:+.4f}")
        
        if improvement > 0:
            print("✓ GNN outperforms baseline!")
        else:
            print("⚠ GNN does not outperform baseline")
    except:
        pass

if __name__ == "__main__":
    main()
