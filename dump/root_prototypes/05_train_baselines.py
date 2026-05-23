"""
Module D - Step D1: Baseline Benchmark

Train Random Forest and Logistic Regression on flattened graph features.
This establishes the number to beat for GNN models.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import os

# Paths
GRAPH_DIR = "data/graphs"
OUT_DIR = "data/processed"

def load_and_flatten_graphs():
    """Load patient graphs and flatten to (N_samples, N_features) matrix."""
    print("Loading patient graphs...")
    
    with open(os.path.join(GRAPH_DIR, "patient_graphs_3d.pkl"), 'rb') as f:
        data = pickle.load(f)
    
    data_list = data['data_list']
    print(f"Loaded {len(data_list)} patient graphs")
    print(f"Node features per graph: {data_list[0].x.shape}")
    
    # Flatten: use mean of each feature across nodes
    # Shape: (N_samples, 3) - mean expression, mean degree, mean variance
    X_mean = np.array([d.x.mean(dim=0).numpy() for d in data_list])
    
    # Also try: use all node features flattened (more expressive but higher dim)
    # For 1047 nodes × 3 features = 3141 features - still manageable
    # But let's use a more compact representation: statistics per feature
    X_stats = []
    for d in data_list:
        x = d.x.numpy()
        # Per feature: mean, std, min, max, median
        stats = []
        for feat_idx in range(x.shape[1]):
            feat = x[:, feat_idx]
            stats.extend([
                feat.mean(),
                feat.std(),
                np.percentile(feat, 25),
                np.percentile(feat, 75),
            ])
        X_stats.append(stats)
    
    X = np.array(X_stats)  # (319, 12) - compact representation
    y = np.array([d.y.item() for d in data_list])
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels: Control={np.sum(y==0)}, Sepsis={np.sum(y==1)}")
    
    return X, y

def train_baseline(X, y, model_class, model_name, **kwargs):
    """Train baseline model with 5-fold CV."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_aurocs = []
    fold_accs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Train model
        model = model_class(**kwargs)
        model.fit(X_train, y_train)
        
        # Predict
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_val)[:, 1]
        else:
            probs = model.decision_function(X_val)
        preds = model.predict(X_val)
        
        # Metrics
        auroc = roc_auc_score(y_val, probs)
        acc = accuracy_score(y_val, preds)
        
        fold_aurocs.append(auroc)
        fold_accs.append(acc)
        print(f"  Fold {fold+1}: AUROC={auroc:.4f}, Accuracy={acc:.4f}")
    
    mean_auroc = np.mean(fold_aurocs)
    std_auroc = np.std(fold_aurocs)
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    
    print(f"\n{model_name} Summary:")
    print(f"  Mean AUROC: {mean_auroc:.4f} (+/- {std_auroc:.4f})")
    print(f"  Mean Accuracy: {mean_acc*100:.1f}% (+/- {std_acc*100:.1f}%)")
    
    return mean_auroc, std_auroc, mean_acc, std_acc

def main():
    # Load data
    X, y = load_and_flatten_graphs()
    
    results = {}
    
    # Random Forest
    rf_auroc, rf_std, rf_acc, rf_acc_std = train_baseline(
        X, y, 
        RandomForestClassifier, 
        "Random Forest",
        n_estimators=100, 
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    results['Random Forest'] = {
        'auroc': rf_auroc, 'auroc_std': rf_std,
        'acc': rf_acc, 'acc_std': rf_acc_std
    }
    
    # Logistic Regression
    lr_auroc, lr_std, lr_acc, lr_acc_std = train_baseline(
        X, y, 
        LogisticRegression, 
        "Logistic Regression",
        max_iter=1000,
        random_state=42
    )
    results['Logistic Regression'] = {
        'auroc': lr_auroc, 'auroc_std': lr_std,
        'acc': lr_acc, 'acc_std': lr_acc_std
    }
    
    # Save results
    results_df = pd.DataFrame([
        {'Model': 'Random Forest', 'AUROC': rf_auroc, 'Std': rf_std, 'Accuracy': rf_acc},
        {'Model': 'Logistic Regression', 'AUROC': lr_auroc, 'Std': lr_std, 'Accuracy': lr_acc}
    ])
    results_df.to_csv(os.path.join(OUT_DIR, "baseline_results_optimized.csv"), index=False)
    
    # Summary
    print(f"\n{'='*60}")
    print("=== BASELINE BENCHMARK COMPLETE ===")
    print(f"{'='*60}")
    print(f"\n| Model | Mean AUROC | Std Dev | Accuracy |")
    print(f"|-------|------------|---------|----------|")
    print(f"| Random Forest | {rf_auroc:.4f} | {rf_std:.4f} | {rf_acc*100:.1f}% |")
    print(f"| Logistic Regression | {lr_auroc:.4f} | {lr_std:.4f} | {lr_acc*100:.1f}% |")
    print(f"\n*** NUMBER TO BEAT: {max(rf_auroc, lr_auroc):.4f} ***")

if __name__ == "__main__":
    main()
