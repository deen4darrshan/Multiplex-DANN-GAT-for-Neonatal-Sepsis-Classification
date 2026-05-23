import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_DIR = os.path.join(ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(ROOT, 'results')
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def plot_roc(y_true, y_prob, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    expr = pd.read_csv(os.path.join(PROC_DIR, 'expression_combat.csv'), index_col=0)
    meta = pd.read_csv(os.path.join(PROC_DIR, 'metadata_combat.csv'))

    with open(os.path.join(PROC_DIR, 'final_genes.txt'), 'r') as f:
        genes = [line.strip() for line in f if line.strip()]

    X = expr.loc[genes].T.values
    y = meta['Label'].values
    batches = meta['Batch'].values

    # Determine folds
    class_counts = np.bincount(y)
    n_splits = int(min(5, class_counts.min()))
    if n_splits < 2:
        raise ValueError('Not enough samples per class for CV')

    strat_key = [f"{l}_{b}" for l, b in zip(y, batches)]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = {
        'LogisticRegression': LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced', solver='lbfgs', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    }

    results = {}

    for name, model in models.items():
        fold_metrics = []
        y_true_all = []
        y_prob_all = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, strat_key)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            auc = roc_auc_score(y_val, y_prob)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            acc = accuracy_score(y_val, y_pred)

            fold_metrics.append({'fold': fold + 1, 'auc': auc, 'f1': f1, 'accuracy': acc})
            y_true_all.extend(y_val)
            y_prob_all.extend(y_prob)

        df = pd.DataFrame(fold_metrics)
        results[name] = {
            'mean_auc': float(df['auc'].mean()),
            'std_auc': float(df['auc'].std()),
            'mean_f1': float(df['f1'].mean()),
            'mean_acc': float(df['accuracy'].mean()),
            'folds': fold_metrics
        }

        # Plot ROC
        plot_roc(np.array(y_true_all), np.array(y_prob_all), f'{name} ROC (CV)', os.path.join(FIG_DIR, f'roc_{name.lower()}.png'))

    with open(os.path.join(RESULTS_DIR, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('Baseline results saved.')


if __name__ == '__main__':
    main()
