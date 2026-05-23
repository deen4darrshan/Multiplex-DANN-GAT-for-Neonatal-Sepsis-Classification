"""
Module D - Task D.1: Baseline Models (RF/XGB)

Trains Random Forest and XGBoost baselines on flat gene expression.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

# Try to import xgboost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed, skipping...")

# Paths
DATA_DIR = "data/processed"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def load_data():
    """Load training data."""
    print("Loading training data...")
    
    # Load combined training (samples x genes)
    df = pd.read_csv(os.path.join(DATA_DIR, "combined_training.csv"), index_col=0)
    print(f"Shape: {df.shape}")
    
    # Separate features and labels
    X = df.drop(['Condition', 'Batch'], axis=1).values
    y_raw = df['Condition'].values
    
    # Filter to only Sepsis/Control (exclude Unknown)
    mask = (y_raw == 'Sepsis') | (y_raw == 'Control')
    X = X[mask]
    y_raw = y_raw[mask]
    
    # Convert labels
    y = np.array([1 if label == 'Sepsis' else 0 for label in y_raw])
    
    print(f"After filtering Unknown: X={X.shape}, y={len(y)}")
    print(f"Class distribution: Sepsis={sum(y)}, Control={sum(1-y)}")
    
    return X, y

def train_evaluate_cv(X, y, model_fn, model_name, n_splits=5):
    """Train and evaluate using cross-validation."""
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    aurocs = []
    accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Train
        model = model_fn()
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Metrics
        auroc = roc_auc_score(y_val, y_prob)
        acc = accuracy_score(y_val, y_pred)
        
        aurocs.append(auroc)
        accuracies.append(acc)
        
        print(f"Fold {fold+1}: AUROC={auroc:.4f}, Accuracy={acc:.4f}")
    
    print(f"\n{model_name} Summary:")
    print(f"  Mean AUROC: {np.mean(aurocs):.4f} (+/- {np.std(aurocs):.4f})")
    print(f"  Mean Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    
    return aurocs, accuracies

def main():
    # Load data
    X, y = load_data()
    
    results = {}
    
    # 1. Random Forest
    rf_aurocs, rf_accs = train_evaluate_cv(
        X, y,
        lambda: RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
        "Random Forest"
    )
    results['Random Forest'] = {'auroc': np.mean(rf_aurocs), 'acc': np.mean(rf_accs)}
    
    # 2. Logistic Regression
    lr_aurocs, lr_accs = train_evaluate_cv(
        X, y,
        lambda: LogisticRegression(C=0.1, penalty='l2', class_weight='balanced', max_iter=1000, random_state=42),
        "Logistic Regression"
    )
    results['Logistic Regression'] = {'auroc': np.mean(lr_aurocs), 'acc': np.mean(lr_accs)}
    
    # 3. XGBoost (if available)
    if HAS_XGB:
        xgb_aurocs, xgb_accs = train_evaluate_cv(
            X, y,
            lambda: xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss'),
            "XGBoost"
        )
        results['XGBoost'] = {'auroc': np.mean(xgb_aurocs), 'acc': np.mean(xgb_accs)}
    
    # Summary
    print(f"\n{'='*60}")
    print("=== BASELINE RESULTS SUMMARY ===")
    print(f"{'='*60}")
    
    for model_name, metrics in results.items():
        print(f"{model_name}: AUROC={metrics['auroc']:.4f}, Accuracy={metrics['acc']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(DATA_DIR, "baseline_results.csv"))
    
    # Verification
    print(f"\n{'='*60}")
    print("=== VERIFICATION (CoVe) ===")
    print(f"{'='*60}")
    
    best_auroc = max([m['auroc'] for m in results.values()])
    if best_auroc >= 0.60:
        print(f"✓ PASS: Best baseline AUROC >= 0.60 ({best_auroc:.4f})")
    else:
        print(f"⚠ WARNING: Best baseline AUROC < 0.60 ({best_auroc:.4f})")

if __name__ == "__main__":
    main()
