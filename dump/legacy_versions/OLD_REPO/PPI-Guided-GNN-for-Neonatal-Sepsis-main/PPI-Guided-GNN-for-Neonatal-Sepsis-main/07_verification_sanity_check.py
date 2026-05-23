import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import os

# Paths
GRAPH_PATH = "data/graphs/patient_graphs_3d.pkl"

def load_data():
    print(f"Loading data from {GRAPH_PATH}...")
    with open(GRAPH_PATH, 'rb') as f:
        data = pickle.load(f)
    
    data_list = data['data_list']
    print(f"Loaded {len(data_list)} samples.")
    
    # Extract features and labels
    # Flatten assumption: (Num_Nodes * 3) features per sample
    X = []
    y = []
    
    for d in data_list:
        # d.x shape is (Num_Nodes, 3)
        features = d.x.numpy().flatten()
        X.append(features)
        y.append(d.y.item())
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Feature Matrix Shape: {X.shape}")
    print(f"Label Distribution: {np.bincount(y)}")
    
    return X, y

def run_baseline(name, model, X, y):
    print(f"\nRunning {name}...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
    
    print(f"  Fold Scores: {scores}")
    print(f"  Mean AUROC: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores.mean()

def main():
    if not os.path.exists(GRAPH_PATH):
        print(f"Error: {GRAPH_PATH} not found.")
        return

    X, y = load_data()
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, solver='liblinear') # robust for small datasets
    lr_score = run_baseline("Logistic Regression", lr, X, y)
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_score = run_baseline("Random Forest", rf, X, y)
    
    print(f"\n{'='*40}")
    print("DIAGNOSTIC VERDICT")
    print(f"{'='*40}")
    
    threshold = 0.70
    if lr_score > threshold or rf_score > threshold:
        print("✅ PASS: Data has signal. The GNN is the problem.")
        print("Recommendation: Tune GNN, reduce regularization, or simplify architecture.")
    else:
        print("❌ FAIL: Data has lost signal. The Feature Selection is the problem.")
        print("Recommendation: Revert to top-2000 genes or change selection method.")

if __name__ == "__main__":
    main()
