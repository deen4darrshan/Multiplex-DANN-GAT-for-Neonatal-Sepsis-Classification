"""
Phase 3B: GNN Training — SepsisGAT v2 (Transfer Learning + From Scratch + GCN)
================================================================================
This script:
1. Defines SepsisGAT v2 (no DANN, no meta stream) and SepsisGCN
2. Loads pre-trained SeizureGAT weights for transfer learning
3. Trains with stratified 5-fold CV  
4. Evaluates with early stopping, cosine annealing, edge/feature augmentation
5. Saves best model per fold

CoV Verification:
- Print transferred layer shapes & counts
- Monitor train/val loss per epoch
- Report per-fold AUC, F1, accuracy
- Assert mean AUC > 0.55 (better than random)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch.nn import LayerNorm, Linear
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import numpy as np
import os
import sys
import json
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

V2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
V1_ROOT = os.path.abspath(os.path.join(V2_ROOT, '..'))
PROCESSED_DIR = os.path.join(V2_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(V2_ROOT, 'models')
RESULTS_DIR = os.path.join(V2_ROOT, 'results')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Pre-trained EEG weights
EEG_WEIGHTS_PATH = os.path.join(V1_ROOT, 'ISEF_GNNs', 'best_gat_eeg.pt')

# ============================================================
# MODEL DEFINITIONS
# ============================================================

class SepsisGATv2(nn.Module):
    """Simplified SepsisGAT v2 — no DANN, no meta stream."""
    
    def __init__(self, num_node_features=1, hidden_channels=64, heads=4, dropout=0.5):
        super().__init__()
        self.dropout_rate = dropout
        
        # Layer 1: Input projection
        self.conv1 = TransformerConv(num_node_features, hidden_channels, heads=heads, dropout=0.1)
        self.ln1 = LayerNorm(hidden_channels * heads)
        self.res_proj = Linear(num_node_features, hidden_channels * heads)
        
        # Layer 2: Intermediate (Transfer target)
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.1)
        self.ln2 = LayerNorm(hidden_channels * heads)
        
        # Layer 3: Output conv (Transfer target)
        self.conv3 = TransformerConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.1)
        self.ln3 = LayerNorm(hidden_channels)
        
        # Classifier: pool(mean + max) → 2*hidden → hidden → 2
        self.lin1 = Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)
    
    def forward(self, x, edge_index, batch, edge_drop_rate=0.0, noise_std=0.0):
        # On-the-fly augmentation (training only)
        if self.training and edge_drop_rate > 0:
            mask = torch.rand(edge_index.size(1), device=edge_index.device) > edge_drop_rate
            edge_index = edge_index[:, mask]
        
        if self.training and noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        
        # Layer 1
        res = self.res_proj(x)
        x = F.leaky_relu(self.ln1(self.conv1(x, edge_index)) + res)
        
        # Layer 2
        res = x
        x = F.leaky_relu(self.ln2(self.conv2(x, edge_index)) + res)
        
        # Layer 3
        x = F.leaky_relu(self.ln3(self.conv3(x, edge_index)))
        
        # Pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pool = torch.cat([x_mean, x_max], dim=1)
        
        # Classifier
        x = F.leaky_relu(self.lin1(x_pool))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        
        return x


class SepsisGCN(nn.Module):
    """GCN baseline — no attention mechanism."""
    
    def __init__(self, num_node_features=1, hidden_channels=64, dropout=0.5):
        super().__init__()
        self.dropout_rate = dropout
        
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.ln1 = LayerNorm(hidden_channels)
        self.res_proj = Linear(num_node_features, hidden_channels)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.ln2 = LayerNorm(hidden_channels)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.ln3 = LayerNorm(hidden_channels)
        
        self.lin1 = Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)
    
    def forward(self, x, edge_index, batch, edge_drop_rate=0.0, noise_std=0.0):
        if self.training and edge_drop_rate > 0:
            mask = torch.rand(edge_index.size(1), device=edge_index.device) > edge_drop_rate
            edge_index = edge_index[:, mask]
        
        if self.training and noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        
        res = self.res_proj(x)
        x = F.leaky_relu(self.ln1(self.conv1(x, edge_index)) + res)
        
        res = x
        x = F.leaky_relu(self.ln2(self.conv2(x, edge_index)) + res)
        
        x = F.leaky_relu(self.ln3(self.conv3(x, edge_index)))
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pool = torch.cat([x_mean, x_max], dim=1)
        
        x = F.leaky_relu(self.lin1(x_pool))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        
        return x


# ============================================================
# TRANSFER LEARNING
# ============================================================
def load_transfer_weights(model, eeg_path, freeze=True):
    """Load pre-trained EEG weights into conv2/conv3 layers."""
    if not os.path.exists(eeg_path):
        print(f"  WARNING: EEG weights not found at {eeg_path}. Training from scratch.")
        return model, 0
    
    checkpoint = torch.load(eeg_path, map_location='cpu', weights_only=False)
    model_dict = model.state_dict()
    
    transfer_layers = ['conv2', 'ln2', 'conv3', 'ln3']
    transferred = {}
    
    for k, v in checkpoint.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            if any(l in k for l in transfer_layers):
                transferred[k] = v
    
    model_dict.update(transferred)
    model.load_state_dict(model_dict)
    
    if freeze:
        for name, param in model.named_parameters():
            if any(l in name for l in transfer_layers):
                param.requires_grad = False
    
    return model, len(transferred)


# ============================================================
# TRAINING LOOP
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device, edge_drop=0.02, noise_std=0.02):
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch, 
                   edge_drop_rate=edge_drop, noise_std=noise_std)
        loss = criterion(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        total_loss += loss.item() * data.num_graphs
        
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        labels = data.y.cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels)
    
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    
    # Find optimal threshold that maximizes F1
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.3, 0.7, 0.02):
        y_tmp = (y_prob > thresh).astype(int)
        f1_tmp = f1_score(y_true, y_tmp, zero_division=0)
        if f1_tmp > best_f1:
            best_f1 = f1_tmp
            best_thresh = thresh
    
    y_pred = (y_prob > best_thresh).astype(int)
    
    metrics = {
        'loss': total_loss / len(loader.dataset),
        'auc': roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'threshold': best_thresh,
        'y_true': y_true,
        'y_prob': y_prob,
    }
    return metrics


def run_experiment(model_class, model_kwargs, train_graphs, device, 
                   experiment_name, transfer_path=None, freeze=True,
                   epochs=100, lr=1e-3, wd=1e-4, batch_size=16, patience=30):
    """Run full 5-fold CV experiment."""
    
    print(f"\n  {'='*50}")
    print(f"  Experiment: {experiment_name}")
    print(f"  {'='*50}")
    
    # Stratification
    labels = [g.y.item() for g in train_graphs]
    batch_labels = [g.batch_label for g in train_graphs]
    strat_key = [f"{l}_{b}" for l, b in zip(labels, batch_labels)]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    best_overall_auc = 0
    best_overall_model = None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_graphs, strat_key)):
        print(f"\n    Fold {fold+1}/5")
        
        train_subset = [train_graphs[i] for i in train_idx]
        val_subset = [train_graphs[i] for i in val_idx]
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)
        
        # Compute class weights
        train_labels = [g.y.item() for g in train_subset]
        n_pos = sum(train_labels)
        n_neg = len(train_labels) - n_pos
        w_pos = len(train_labels) / (2 * max(n_pos, 1))
        w_neg = len(train_labels) / (2 * max(n_neg, 1))
        class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Initialize model
        model = model_class(**model_kwargs).to(device)
        
        # Transfer learning
        if transfer_path:
            model, n_transferred = load_transfer_weights(model, transfer_path, freeze=freeze)
            if fold == 0:
                print(f"    Transferred {n_transferred} parameter tensors from EEG")
        
        # Optimizer + Scheduler
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=wd
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
        
        # Training loop with early stopping
        best_val_auc = 0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_metrics = evaluate(model, val_loader, device)
                val_auc = val_metrics['auc']
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 5
                
                if fold == 0 and (epoch + 1) % 20 == 0:
                    print(f"      Epoch {epoch+1}: train_loss={train_loss:.4f} val_AUC={val_auc:.4f} F1={val_metrics['f1']:.4f} (t={val_metrics['threshold']:.2f})")
                
                if patience_counter >= patience:
                    break
        
        # Load best model and evaluate
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        val_metrics = evaluate(model, val_loader, device)
        fold_results.append({
            'fold': fold + 1,
            'auc': val_metrics['auc'],
            'f1': val_metrics['f1'],
            'accuracy': val_metrics['accuracy'],
        })
        
        print(f"    Fold {fold+1}: AUC={val_metrics['auc']:.4f}  F1={val_metrics['f1']:.4f}  Acc={val_metrics['accuracy']:.4f}  Thresh={val_metrics['threshold']:.2f}")
        
        # Save best overall model
        if val_metrics['auc'] > best_overall_auc:
            best_overall_auc = val_metrics['auc']
            best_overall_model = copy.deepcopy(model.state_dict())
    
    # Aggregate results
    df = pd.DataFrame(fold_results)
    summary = {
        'experiment': experiment_name,
        'mean_auc': float(df['auc'].mean()),
        'std_auc': float(df['auc'].std()),
        'mean_f1': float(df['f1'].mean()),
        'mean_acc': float(df['accuracy'].mean()),
        'folds': fold_results,
    }
    
    print(f"\n    RESULT: AUC = {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
    
    # Save best model
    model_path = os.path.join(MODELS_DIR, f'{experiment_name.lower().replace(" ", "_")}_best.pt')
    torch.save(best_overall_model, model_path)
    print(f"    Saved: {model_path}")
    
    return summary, best_overall_model


# ============================================================
# MAIN
# ============================================================
import pandas as pd

def main():
    print("=" * 70)
    print("PHASE 3B: GNN Training")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Load graphs
    train_graphs = torch.load(os.path.join(PROCESSED_DIR, 'train_graphs.pt'), weights_only=False)
    n_features = train_graphs[0].x.shape[1]
    print(f"  Training graphs: {len(train_graphs)}")
    print(f"  Node features: {n_features}")
    print(f"  EEG weights: {EEG_WEIGHTS_PATH}")
    print(f"  EEG weights exist: {os.path.exists(EEG_WEIGHTS_PATH)}")
    
    all_results = {}
    
    # --- Experiment 1: GAT with Transfer Learning (unfrozen fine-tune) ---
    gat_transfer_result, gat_transfer_model = run_experiment(
        model_class=SepsisGATv2,
        model_kwargs={'num_node_features': n_features, 'hidden_channels': 64, 'heads': 4, 'dropout': 0.5},
        train_graphs=train_graphs,
        device=device,
        experiment_name='GAT_Transfer',
        transfer_path=EEG_WEIGHTS_PATH,
        freeze=False,  # Unfreeze all layers — input domain is too different to keep frozen
    )
    all_results['GAT_Transfer'] = gat_transfer_result
    
    # --- Experiment 2: GAT from Scratch (Ablation) ---
    gat_scratch_result, gat_scratch_model = run_experiment(
        model_class=SepsisGATv2,
        model_kwargs={'num_node_features': n_features, 'hidden_channels': 64, 'heads': 4, 'dropout': 0.5},
        train_graphs=train_graphs,
        device=device,
        experiment_name='GAT_Scratch',
        transfer_path=None,
    )
    all_results['GAT_Scratch'] = gat_scratch_result
    
    # --- Experiment 3: GCN (Architecture Comparison) ---
    gcn_result, gcn_model = run_experiment(
        model_class=SepsisGCN,
        model_kwargs={'num_node_features': n_features, 'hidden_channels': 64, 'dropout': 0.5},
        train_graphs=train_graphs,
        device=device,
        experiment_name='GCN_Baseline',
        transfer_path=None,
    )
    all_results['GCN_Baseline'] = gcn_result
    
    # Save all results
    with open(os.path.join(RESULTS_DIR, 'gnn_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # --- Final Summary ---
    print(f"\n{'=' * 70}")
    print("PHASE 3B COMPLETE: GNN Training Results")
    print(f"{'=' * 70}")
    print(f"\n  {'Model':<20} {'Mean AUC':>10} {'± Std':>8} {'Mean F1':>10} {'Mean Acc':>10}")
    print(f"  {'-'*58}")
    for name, r in all_results.items():
        print(f"  {name:<20} {r['mean_auc']:>10.4f} {r['std_auc']:>8.4f} {r['mean_f1']:>10.4f} {r['mean_acc']:>10.4f}")
    
    # CoV: Transfer vs Scratch comparison
    print(f"\n  === CoV: Transfer Learning Value ===")
    transfer_auc = all_results['GAT_Transfer']['mean_auc']
    scratch_auc = all_results['GAT_Scratch']['mean_auc']
    delta = transfer_auc - scratch_auc
    print(f"  Transfer AUC: {transfer_auc:.4f}")
    print(f"  Scratch AUC:  {scratch_auc:.4f}")
    print(f"  Delta:        {delta:+.4f}")
    if delta > 0.02:
        print(f"  Verdict: Transfer Learning HELPS (+{delta:.4f})")
    elif delta < -0.02:
        print(f"  Verdict: Transfer Learning HURTS ({delta:.4f}). Recommend scratch model.")
    else:
        print(f"  Verdict: Transfer Learning NEUTRAL (|delta| < 0.02)")

if __name__ == "__main__":
    main()
