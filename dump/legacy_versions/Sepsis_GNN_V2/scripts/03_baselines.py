"""
Phase 3A: Baseline Models (LR + RF)
====================================
Establishes performance ceiling using tabular expression data.
These baselines ignore graph topology — used to benchmark GNN value.
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

V2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(V2_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(V2_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print("=" * 70)
    print("PHASE 3A: Baseline Models (Tabular)")
    print("=" * 70)
    
    # Load ComBat-corrected training data
    train_expr = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_expression_combat.csv'), index_col=0)
    train_meta = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_metadata.csv'))
    
    # Load gene list (same genes used for graphs)
    with open(os.path.join(PROCESSED_DIR, 'final_genes.txt'), 'r') as f:
        gene_list = [line.strip() for line in f if line.strip()]
    
    # Prepare tabular data (samples × genes)
    X = train_expr.loc[gene_list].T.values  # [N_samples, N_genes]
    y = train_meta['Label'].values
    batches = train_meta['Batch'].values
    
    # Create stratification key
    strat_key = [f"{l}_{b}" for l, b in zip(y, batches)]
    
    print(f"\n  Features: {X.shape[1]} genes")
    print(f"  Samples: {X.shape[0]} (Sepsis: {sum(y)}, Control: {len(y)-sum(y)})")
    
    # Models to evaluate
    models = {
        'LogisticRegression': LogisticRegression(
            C=1.0, max_iter=2000, class_weight='balanced', solver='lbfgs', random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1
        )
    }
    
    # 5-Fold Stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    
    for name, model in models.items():
        print(f"\n  --- {name} ---")
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, strat_key)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            
            model.fit(X_train, y_train)
            
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)
            
            auc = roc_auc_score(y_val, y_prob)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            acc = accuracy_score(y_val, y_pred)
            
            fold_metrics.append({'fold': fold+1, 'auc': auc, 'f1': f1, 'accuracy': acc})
            print(f"    Fold {fold+1}: AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}")
        
        df = pd.DataFrame(fold_metrics)
        results[name] = {
            'mean_auc': df['auc'].mean(),
            'std_auc': df['auc'].std(),
            'mean_f1': df['f1'].mean(),
            'mean_acc': df['accuracy'].mean(),
            'folds': fold_metrics
        }
        print(f"  Mean AUC: {df['auc'].mean():.4f} ± {df['auc'].std():.4f}")
    
    # Save results
    with open(os.path.join(RESULTS_DIR, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print("BASELINES COMPLETE")
    print(f"{'=' * 70}")
    for name, r in results.items():
        print(f"  {name}: AUC = {r['mean_auc']:.4f} ± {r['std_auc']:.4f}")

if __name__ == "__main__":
    main()
