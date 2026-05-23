"""
Module D - GAT Training with Model Weight Saving (Optimized)

Saves the best model weights from each fold and the overall best model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm

# Paths
GRAPH_DIR = "data/graphs"
OUT_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# OPTIMIZED Hyperparameters
HIDDEN_CHANNELS = 64
NUM_HEADS = 2
NUM_LAYERS = 2
DROPOUT = 0.6
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0005
EARLY_STOPPING_PATIENCE = 150 # Effectively disabled (Epochs=100)
N_SPLITS = 5

print(f"Using device: {device}")
print(f"\n=== GAT OPTIMIZED CONFIGURATION ===")
print(f"Hidden Channels: {HIDDEN_CHANNELS}")
print(f"Attention Heads: {NUM_HEADS}")
print(f"Layers: {NUM_LAYERS}")
print(f"Dropout: {DROPOUT}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
print(f"Model weights will be saved to: {MODEL_DIR}/")
print(f"====================================\n")

class OptimizedGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2, heads=2, dropout=0.6):
        super(OptimizedGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, 
                             dropout=0.0, add_self_loops=False)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels // 2, 
                             heads=1, concat=False, dropout=0.0, add_self_loops=False)
        self.classifier = nn.Linear(hidden_channels // 2, num_classes)
        self.dropout = dropout
    
    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x

def load_graphs():
    print("Loading patient graphs...")
    with open(os.path.join(GRAPH_DIR, "patient_graphs_3d.pkl"), 'rb') as f:
        data = pickle.load(f)
    data_list = data['data_list']
    print(f"Loaded {len(data_list)} graphs, {data_list[0].x.shape[0]} nodes, {data_list[0].x.shape[1]} features")
    return data_list

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
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
        model = OptimizedGAT(in_channels, HIDDEN_CHANNELS, num_classes=2, 
                             heads=NUM_HEADS, dropout=DROPOUT).to(device)
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if fold == 0:
            print(f"Model parameters: {num_params:,}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        best_auroc = 0
        best_model_state = None
        patience_counter = 0
        
        pbar = tqdm(range(1, EPOCHS + 1), desc=f"Fold {fold+1}", ncols=80)
        for epoch in pbar:
            loss = train_epoch(model, train_loader, optimizer, criterion)
            
            if epoch % 10 == 0 or epoch == EPOCHS:
                auroc, acc = evaluate(model, val_loader)
                pbar.set_postfix({'loss': f'{loss:.4f}', 'auroc': f'{auroc:.3f}'})
                
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= EARLY_STOPPING_PATIENCE // 10:
                    pbar.set_description(f"Fold {fold+1} (early stop)")
                    break
        
        # Save best model for this fold
        fold_model_path = os.path.join(MODEL_DIR, f"gat_fold{fold+1}.pt")
        torch.save(best_model_state, fold_model_path)
        print(f"Saved fold {fold+1} best model (AUROC={best_auroc:.4f}) to {fold_model_path}")
        
        # Track overall best
        if best_auroc > best_overall_auroc:
            best_overall_auroc = best_auroc
            best_overall_model_state = best_model_state.copy()
        
        fold_aurocs.append(best_auroc)
        _, final_acc = evaluate(model, val_loader)
        fold_accs.append(final_acc)
        print(f"Fold {fold+1} Best AUROC: {best_auroc:.4f}")
    
    # Save overall best model
    best_model_path = os.path.join(MODEL_DIR, "gat_best.pt")
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
    
    results = pd.DataFrame({
        'Model': ['GAT'],
        'AUROC': [gat_auroc],
        'Std': [gat_std],
        'Accuracy': [gat_acc],
        'Best_Fold_AUROC': [best_auroc]
    })
    results.to_csv(os.path.join(OUT_DIR, "gat_results_optimized.csv"), index=False)
    
    print(f"\n{'='*60}")
    print("=== GAT TRAINING COMPLETE ===")
    print(f"{'='*60}")
    print(f"Mean AUROC: {gat_auroc:.4f} (+/- {gat_std:.4f})")
    print(f"Mean Accuracy: {gat_acc*100:.1f}%")
    print(f"Best Fold AUROC: {best_auroc:.4f}")
    print(f"\nModel weights saved to: {MODEL_DIR}/")
    print(f"  - gat_fold1.pt through gat_fold5.pt")
    print(f"  - gat_best.pt (overall best)")

if __name__ == "__main__":
    main()
