"""
Phase 4: External Validation of V11 Multiplex DANN
===================================================
Evaluates the best V11 Multiplex DANN model on the held-out GSE26440 pediatric cohort.
This tests true out-of-distribution (OOD) biological generalization since the pediatric 
cohort was not included in the Combat normalization nor the age bracket of the training set.
"""

import os, sys, time, json, warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, classification_report)
from scipy.stats import median_abs_deviation

# Add training script path to import the model definitions
sys.path.insert(0, os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "scripts"))
import importlib
train_module = importlib.import_module('19_train_v11_multiplex_dann')

MultiplexGNNGuidedDANN = train_module.MultiplexGNNGuidedDANN
build_kegg_hyperedges = train_module.build_kegg_hyperedges
build_string_hyperedges = train_module.build_string_hyperedges
build_coexpr_hyperedges = train_module.build_coexpr_hyperedges
make_data_list = train_module.make_data_list
collate_multiplex = train_module.collate_multiplex
compute_metrics = train_module.compute_metrics

# ============================================================================
# CONFIGURATION
# ============================================================================
OUT_DIR     = os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "results")
MODEL_DIR   = os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "models")
PROC_DIR    = os.path.join(PROJECT_ROOT, "data", "processed")
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TOP_K = 2000
H_DIM = 64
DROPOUT = 0.3
COEXPR_THR = 0.7

def main():
    print("=" * 80)
    print("  EXTERNAL VALIDATION: V11 Multiplex DANN on GSE26440 Pediatric Cohort")
    print("=" * 80)

    # 1. Rebuild Node Features (Top 2000 genes from training)
    print("\n[1/5] Extracting Top Genes from Training Set...")
    train_expr_path = os.path.join(OUT_DIR, "expression_combat_v2.csv")
    if not os.path.exists(train_expr_path):
        print(f"ERROR: Cannot find {train_expr_path}")
        return
        
    expr_train = pd.read_csv(train_expr_path, index_col=0)
    mad = expr_train.apply(median_abs_deviation, axis=1)
    top_genes = mad.sort_values(ascending=False).head(TOP_K).index.tolist()
    print(f"  Selected {len(top_genes)} highest variance genes.")

    # 2. Build Static Hyperedges
    print("\n[2/5] Building Static Hyperedges...")
    kegg_hei, n_kegg = build_kegg_hyperedges(top_genes)
    string_hei, n_string = build_string_hyperedges(top_genes)
    print(f"  KEGG: {n_kegg} pathways")
    print(f"  STRING: {n_string} edges")

    # 3. Build Co-expression Hyperedges (USING TRAINING SET ONLY)
    print("\n[3/5] Building Co-expression Hyperedges (Training Data Only)...")
    train_sids = expr_train.columns.tolist()
    coexpr_hei, n_coexpr = build_coexpr_hyperedges(expr_train.loc[top_genes], top_genes, train_sids)
    print(f"  CoExpr: {n_coexpr} edges (|ρ| > {COEXPR_THR})")

    # 4. Load GSE26440 External Dataset
    print("\n[4/5] Loading External GSE26440 Validation Dataset...")
    ext_expr_path = os.path.join(PROC_DIR, "GSE26440_Valid_mapped.csv")
    ext_meta_path = os.path.join(PROC_DIR, "GSE26440_Valid_phenotype.csv")
    
    expr_ext = pd.read_csv(ext_expr_path, index_col=0)
    meta_ext = pd.read_csv(ext_meta_path)
    
    # Filter external expression to exactly `top_genes`
    # Handle missing genes by padding with 0
    missing_genes = set(top_genes) - set(expr_ext.index)
    if missing_genes:
        print(f"  Warning: {len(missing_genes)} genes from training missing in external set. Padding with 0s.")
        for mg in missing_genes:
            expr_ext.loc[mg] = 0.0
            
    # Subset and reorder to exactly match top_genes
    expr_ext_f = expr_ext.loc[top_genes]
    
    # Standardize column naming for condition (some scripts use specific values)
    # Ensure they map to Control / Sepsis
    meta_ext['Condition'] = meta_ext['title'].apply(lambda x: 'Control' if 'control' in str(x).lower() else 'Sepsis')
    meta_ext['SampleID'] = meta_ext.iloc[:, 0] # GSM ID is first column
    meta_ext['Batch'] = 'GSE26440_Valid' # Dummy batch for DANN forward pass

    # Prepare graphs
    # We will spoof the domain_y / batch processing by appending this dummy batch to the meta
    data_list = make_data_list(expr_ext_f, meta_ext, top_genes, kegg_hei, string_hei)
    print(f"  External samples prepared: {len(data_list)}")
    
    # Validation cohort labels
    y_true = [d.y.item() for d in data_list]
    print(f"  Labels: {y_true.count(1)} Sepsis, {y_true.count(0)} Control")

    # 5. Load Model and Evaluate
    print("\n[5/5] Running Model Inference...")
    model_path = os.path.join(MODEL_DIR, "v11_multiplex_dann_best.pt")
    if not os.path.exists(model_path):
        print(f"ERROR: Model weights not found at {model_path}")
        return

    # Initialize model
    model = MultiplexGNNGuidedDANN(n_genes=len(top_genes), h_dim=H_DIM, dropout=DROPOUT).to(DEVICE)
    # Ensure we don't hit strict matching errors if DANN discriminator output dims slightly differ due to num batches
    # We can load with strict=False or adjust the domain classifier layer if needed.
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    # V11 trained with a 4-class or 3-class domain discriminator. 
    # Let's peek at the loaded state dict to dynamically adjust the discriminator shape before loading if needed
    dom_weight = state_dict.get('domain_discriminator.4.weight')
    if dom_weight is not None:
        num_domain_classes = dom_weight.shape[0]
        model.domain_discriminator[4] = torch.nn.Linear(H_DIM, num_domain_classes).to(DEVICE)
        
    model.load_state_dict(state_dict)
    model.eval()

    # Inference (all in one batch since N is small ~100)
    batch_data = collate_multiplex(data_list, coexpr_hei).to(DEVICE)
    
    with torch.no_grad():
        out, dom_out, attn_weights = model(
            batch_data.x, 
            batch_data.hedge_indices, 
            batch_data.batch, 
            batch_data.global_feat
        )
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        
    auc = roc_auc_score(y_true, probs)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    
    mean_attn = attn_weights.mean(dim=0).cpu().numpy()

    print("\n" + "="*50)
    print(" EXTERNAL VALIDATION RESULTS (GSE26440)")
    print("="*50)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  AUROC:     {auc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"\n  Relation Attention: KEGG={mean_attn[0]:.3f}, STRING={mean_attn[1]:.3f}, CoExpr={mean_attn[2]:.3f}")
    
    print("\n  " + classification_report(y_true, preds, target_names=["Control", "Sepsis"]).replace('\n', '\n  '))
    
    # Save results
    res_dict = {
        'dataset': 'GSE26440_Valid',
        'n_samples': len(y_true),
        'accuracy': acc,
        'auroc': auc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'attn_kegg': float(mean_attn[0]),
        'attn_string': float(mean_attn[1]),
        'attn_coexpr': float(mean_attn[2])
    }
    
    with open(os.path.join(OUT_DIR, 'v11_gse26440_external_results.json'), 'w') as f:
        json.dump(res_dict, f, indent=4)
    print(f"\n  Results saved to {os.path.join(OUT_DIR, 'v11_gse26440_external_results.json')}")

if __name__ == "__main__":
    main()
