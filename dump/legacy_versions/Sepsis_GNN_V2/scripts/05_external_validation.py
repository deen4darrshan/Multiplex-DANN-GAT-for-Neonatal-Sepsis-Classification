"""
Phase 4: External Validation
==============================
Evaluates the best GNN model on the held-out GSE26440 pediatric cohort.
This is a ONE-SHOT test — no tuning allowed after this point.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
import pickle

from torch_geometric.loader import DataLoader
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, 
                            classification_report, roc_curve, precision_recall_curve)

V2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(V2_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(V2_ROOT, 'models')
RESULTS_DIR = os.path.join(V2_ROOT, 'results')
FIGURES_DIR = os.path.join(V2_ROOT, 'figures')

# Import model definitions from training script
sys.path.insert(0, os.path.dirname(__file__))
import importlib
train_module = importlib.import_module('04_train_gnn')
SepsisGATv2 = train_module.SepsisGATv2
SepsisGCN = train_module.SepsisGCN


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        labels = data.y.cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels)
    
    return np.array(all_labels), np.array(all_probs)


def plot_roc_curves(results, save_path):
    """Plot ROC curves for all models."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = {'GAT_Transfer': '#e74c3c', 'GAT_Scratch': '#3498db', 'GCN_Baseline': '#2ecc71'}
    
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(r['y_true'], r['y_prob'])
        ax.plot(fpr, tpr, label=f'{name} (AUC={r["auc"]:.3f})', 
                color=colors.get(name, 'gray'), linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('External Validation ROC — GSE26440 Pediatric', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("=" * 70)
    print("PHASE 4: External Validation (GSE26440 Pediatric)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load external graphs
    ext_graphs = torch.load(os.path.join(PROCESSED_DIR, 'external_graphs.pt'), weights_only=False)
    ext_loader = DataLoader(ext_graphs, batch_size=32)
    
    n_features = ext_graphs[0].x.shape[1]
    print(f"  External samples: {len(ext_graphs)}")
    print(f"  Labels: {sum(g.y.item() for g in ext_graphs)} Sepsis, {len(ext_graphs) - sum(g.y.item() for g in ext_graphs)} Control")
    
    # Model configs
    model_configs = {
        'GAT_Transfer': {
            'class': SepsisGATv2,
            'kwargs': {'num_node_features': n_features, 'hidden_channels': 64, 'heads': 4, 'dropout': 0.5},
            'path': os.path.join(MODELS_DIR, 'gat_transfer_best.pt'),
        },
        'GAT_Scratch': {
            'class': SepsisGATv2,
            'kwargs': {'num_node_features': n_features, 'hidden_channels': 64, 'heads': 4, 'dropout': 0.5},
            'path': os.path.join(MODELS_DIR, 'gat_scratch_best.pt'),
        },
        'GCN_Baseline': {
            'class': SepsisGCN,
            'kwargs': {'num_node_features': n_features, 'hidden_channels': 64, 'dropout': 0.5},
            'path': os.path.join(MODELS_DIR, 'gcn_baseline_best.pt'),
        },
    }
    
    results = {}
    
    for name, cfg in model_configs.items():
        print(f"\n  --- {name} ---")
        
        if not os.path.exists(cfg['path']):
            print(f"    SKIP: Model file not found at {cfg['path']}")
            continue
        
        model = cfg['class'](**cfg['kwargs']).to(device)
        model.load_state_dict(torch.load(cfg['path'], map_location=device, weights_only=False))
        
        y_true, y_prob = predict(model, ext_loader, device)
        y_pred = (y_prob > 0.5).astype(int)
        
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        results[name] = {
            'auc': float(auc),
            'f1': float(f1),
            'accuracy': float(acc),
            'y_true': y_true.tolist(),
            'y_prob': y_prob.tolist(),
        }
        
        print(f"    AUC:      {auc:.4f}")
        print(f"    F1:       {f1:.4f}")
        print(f"    Accuracy: {acc:.4f}")
        print(f"    {classification_report(y_true, y_pred, target_names=['Control', 'Sepsis'], zero_division=0)}")
    
    # Plot ROC curves
    if results:
        plot_roc_curves(results, os.path.join(FIGURES_DIR, 'roc_external_validation.png'))
    
    # Save results
    save_results = {k: {kk: vv for kk, vv in v.items() if kk != 'y_true' and kk != 'y_prob'} 
                    for k, v in results.items()}
    with open(os.path.join(RESULTS_DIR, 'external_validation_results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)
    
    # Final Summary
    print(f"\n{'=' * 70}")
    print("PHASE 4 COMPLETE: External Validation")
    print(f"{'=' * 70}")
    print(f"\n  {'Model':<20} {'AUC':>8} {'F1':>8} {'Accuracy':>10}")
    print(f"  {'-'*46}")
    for name, r in results.items():
        print(f"  {name:<20} {r['auc']:>8.4f} {r['f1']:>8.4f} {r['accuracy']:>10.4f}")
    
    # CoV: Is external AUC > random?
    best_model = max(results, key=lambda k: results[k]['auc'])
    best_auc = results[best_model]['auc']
    print(f"\n  Best external model: {best_model} (AUC = {best_auc:.4f})")
    if best_auc > 0.60:
        print(f"  ✓ PASS: Model generalizes to pediatric cohort")
    elif best_auc > 0.55:
        print(f"  ~ MARGINAL: Some generalization signal detected")
    else:
        print(f"  ✗ FAIL: Near-random performance on external data")


if __name__ == "__main__":
    main()
