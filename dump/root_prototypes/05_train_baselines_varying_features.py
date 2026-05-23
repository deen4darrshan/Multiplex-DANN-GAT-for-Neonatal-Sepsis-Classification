"""
Optimization Phase: Baseline Model Feature Selection

Systematically tests Logistic Regression and Random Forest models with varying numbers 
of input features (genes) selected by variance.

Steps:
1. Load combined expression data (ComBat corrected)
2. Calculate variance for all genes
3. Iterate through feature counts: [All, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10]
4. For each count:
    - Select top N genes
    - Train LR and RF using Stratified 5-Fold CV
    - Log results
5. Save best model and full report
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Paths
DATA_DIR = "data/processed"
MODELS_DIR = "models"
LOGS_DIR = "logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Feature counts to test
FEATURE_COUNTS = [10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10]

def load_data():
    """Load expression data and metadata."""
    print("Loading data...")
    expression = pd.read_csv(os.path.join(DATA_DIR, "combined_expression.csv"), index_col=0)
    metadata = pd.read_csv(os.path.join(DATA_DIR, "combined_metadata.csv"))
    
    # Filter metadata to rows present in expression columns
    common_samples = [s for s in metadata['SampleID'] if s in expression.columns]
    metadata = metadata[metadata['SampleID'].isin(common_samples)]
    
    # Align expression columns to metadata rows
    expression = expression[metadata['SampleID']]
    
    y = metadata['Condition'].map({'Control': 0, 'Sepsis': 1}).values
    
    # Handle NaN in targets if any (shouldn't be, but good to check)
    if np.isnan(y).any():
        print("Warning: NaN labels found, dropping...")
        valid_mask = ~np.isnan(y)
        y = y[valid_mask]
        expression = expression.iloc[:, valid_mask]
        
    return expression, y

def get_top_variance_features(expression, n_features):
    """Select top N genes by variance."""
    if n_features is None or n_features >= len(expression):
        return expression.T.values, expression.index.tolist()
        
    variances = expression.var(axis=1)
    top_genes = variances.nlargest(n_features).index
    return expression.loc[top_genes].T.values, top_genes.tolist()

def train_cv(X, y, model_class, seed=42, **kwargs):
    """Train model with 5-fold CV and return metrics."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aurocs = []
    accs = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Train
        model = model_class(**kwargs)
        model.fit(X_train, y_train)
        
        # Evaluate
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val)[:, 1]
        else:
            probs = model.decision_function(X_val)
        preds = model.predict(X_val)
        
        aurocs.append(roc_auc_score(y_val, probs))
        accs.append(accuracy_score(y_val, preds))
        
    return np.mean(aurocs), np.std(aurocs), np.mean(accs), np.std(accs)

def main():
    expression, y = load_data()
    print(f"Data loaded: {expression.shape[0]} genes, {expression.shape[1]} samples")
    print(f"Class balance: {np.bincount(y.astype(int))}")
    
    results = []
    best_lr_score = 0
    best_rf_score = 0
    best_lr_features = 0
    best_rf_features = 0
    
    md_output = "# Baseline Model Optimization Results\n\n"
    md_output += "| Features | LR AUROC | LR Std | LR Acc | RF AUROC | RF Std | RF Acc |\n"
    md_output += "|----------|----------|--------|--------|----------|--------|--------|\n"
    
    for n_feat in FEATURE_COUNTS:
        print(f"\nProcessing {n_feat} features...")
        
        X, selected_genes = get_top_variance_features(expression, n_feat)
        
        # Logistic Regression
        lr_auroc, lr_std, lr_acc, lr_std_acc = train_cv(
            X, y, LogisticRegression, max_iter=2000, random_state=42, class_weight='balanced'
        )
        
        # Random Forest
        rf_auroc, rf_std, rf_acc, rf_std_acc = train_cv(
            X, y, RandomForestClassifier, n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1
        )
        
        print(f"  LR: {lr_auroc:.4f} (±{lr_std:.4f})")
        print(f"  RF: {rf_auroc:.4f} (±{rf_std:.4f})")
        
        md_output += f"| {n_feat} | {lr_auroc:.4f} | {lr_std:.4f} | {lr_acc:.4f} | {rf_auroc:.4f} | {rf_std:.4f} | {rf_acc:.4f} |\n"
        
        results.append({
            'n_features': n_feat,
            'lr_auroc': lr_auroc,
            'rf_auroc': rf_auroc
        })
        
        # Save best models (retrained on full data)
        if lr_auroc > best_lr_score:
            best_lr_score = lr_auroc
            best_lr_features = n_feat
            
            # Retrain on all data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            final_lr = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
            final_lr.fit(X_scaled, y)
            joblib.dump({
                'model': final_lr,
                'scaler': scaler, 
                'genes': selected_genes,
                'auroc': lr_auroc
            }, os.path.join(MODELS_DIR, "lr_best_optimized.pkl"))
            
        if rf_auroc > best_rf_score:
            best_rf_score = rf_auroc
            best_rf_features = n_feat
            
            # Retrain on all data
            scaler = StandardScaler() # RF doesn't strictly need scaling but consistent API helps
            X_scaled = scaler.fit_transform(X)
            final_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)
            final_rf.fit(X_scaled, y)
            joblib.dump({
                'model': final_rf,
                'scaler': scaler,
                'genes': selected_genes,
                'auroc': rf_auroc
            }, os.path.join(MODELS_DIR, "rf_best_optimized.pkl"))

    md_output += "\n\n## Best Models\n"
    md_output += f"- **Best Logistic Regression:** {best_lr_features} features, AUROC = {best_lr_score:.4f}\n"
    md_output += f"- **Best Random Forest:** {best_rf_features} features, AUROC = {best_rf_score:.4f}\n"
    
    with open(os.path.join(LOGS_DIR, "baseline_optimization_results.md"), "w") as f:
        f.write(md_output)
        
    print(f"\nCompleted. Best LR: {best_lr_score:.4f} ({best_lr_features} feats), Best RF: {best_rf_score:.4f} ({best_rf_features} feats)")

if __name__ == "__main__":
    main()
