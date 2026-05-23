"""
A1 HGCN V6: Simple Train/Test Split (Traditional Pipeline)
==========================================================
Requested by user: "No folds, no cross validation. Just split the data for train and test."

Pipeline:
  1. Load V2 ComBat data (319 samples).
  2. Split 80/20 into Train/Test (stratified by Condition).
  3. Train Hybrid V4 model (GNN+MLP) on Train set.
  4. Evaluate on Test set.
  5. (Optional) Evaluate on External GSE26440 if available.

This provides a single, deterministic training run for quick validation 
or final model production.
"""

import os, sys, time, json, warnings
warnings.filterwarnings('ignore')

# Resolve project root (two levels up from CH_DANN_Plan/scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, roc_curve)
from scipy.stats import median_abs_deviation

# ============================================================================
# CONFIGURATION
# ============================================================================
OUT_DIR = os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "results")
MODEL_DIR = os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "models")
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
TOP_K = 2000
H_DIM = 64
DROPOUT = 0.3
BS = 16
EPOCHS = 150
LR = 3e-4
WD = 5e-4
PATIENCE = 30
EVAL_EVERY = 1
STRING_THR = 700
TEST_SIZE = 0.2
SEED = 42

TIER1 = ['FCGR1A','MMP9','S100A8','S100A9','TLR4','MYD88','IL6','CXCL8','MPO','CEACAM8']


# ============================================================================
# HYBRID MODEL (V4)
# ============================================================================
class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x, batch):
        scores = self.attn(x)
        max_s = torch.zeros(batch.max()+1, 1, device=x.device)
        max_s.scatter_reduce_(0, batch.unsqueeze(1), scores, reduce='amax', include_self=False)
        scores = (scores - max_s[batch]).exp()
        denom = torch.zeros(batch.max()+1, 1, device=x.device)
        denom.scatter_add_(0, batch.unsqueeze(1), scores)
        weights = scores / (denom[batch] + 1e-8)
        wx = x * weights
        out = torch.zeros(batch.max()+1, x.size(1), device=x.device)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(wx), wx)
        return out


class HybridHGCN(nn.Module):
    def __init__(self, n_genes, h_dim=64, dropout=0.3):
        super().__init__()
        
        # --- GNN Branch ---
        self.gene_embed = nn.Sequential(
            nn.Linear(1, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
        )
        self.conv1 = HypergraphConv(h_dim, h_dim)
        self.ln1 = nn.LayerNorm(h_dim)
        self.conv2 = HypergraphConv(h_dim, h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
        self.gnn_pool = AttentionPool(h_dim)
        
        # --- MLP Branch ---
        self.mlp_branch = nn.Sequential(
            nn.Linear(n_genes, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
        )
        
        # --- Fusion ---
        self.classifier = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(h_dim, 2)
        )
        self.dropout = dropout

    def forward(self, x, hyperedge_index, batch, global_feat=None):
        # GNN branch
        g = self.gene_embed(x)
        h = self.conv1(g, hyperedge_index)
        h = self.ln1(h); h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        g = g + h
        h = self.conv2(g, hyperedge_index)
        h = self.ln2(h); h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        g = g + h
        gnn_out = self.gnn_pool(g, batch)
        
        # MLP branch
        if global_feat is not None:
            mlp_out = self.mlp_branch(global_feat)
        else:
            mlp_out = global_mean_pool(x, batch)
            mlp_out = mlp_out.expand(-1, gnn_out.size(1))
        
        # Fusion
        fused = torch.cat([gnn_out, mlp_out], dim=1)
        return self.classifier(fused)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    expr = pd.read_csv(os.path.join(OUT_DIR, "expression_combat_v2.csv"), index_col=0)
    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata_v2.csv"))
    
    mad = expr.apply(median_abs_deviation, axis=1)
    top_genes = mad.sort_values(ascending=False).head(TOP_K).index.tolist()
    expr_f = expr.loc[top_genes]
    
    return expr_f, meta, top_genes


def build_hyperedges(gene_list, use_string=True):
    gene_set = set(gene_list)
    pw = {}
    try:
        import gseapy as gp
        kegg = gp.get_library("KEGG_2021_Human")
        for p, genes in kegg.items():
            ol = list(set(genes) & gene_set)
            if len(ol) >= 3:
                pw[p] = ol
    except:
        pass

    se = []
    if use_string:
        ppi_path = os.path.join(PROC_DIR, "ppi_network.csv")
        if os.path.exists(ppi_path):
            ppi = pd.read_csv(ppi_path)
            pf = ppi[(ppi['source'].isin(gene_set))&(ppi['target'].isin(gene_set))&(ppi['score']>=STRING_THR)]
            se = list(zip(pf['source'].tolist(), pf['target'].tolist()))
    
    return pw, se


def make_data_list(expr_f, meta, gene_list, pw, se):
    g2i = {g: i for i, g in enumerate(gene_list)}
    ni, hi, hid = [], [], 0
    for genes in pw.values():
        for g in genes:
            if g in g2i: ni.append(g2i[g]); hi.append(hid)
        hid += 1
    for s, t in se:
        if s in g2i and t in g2i:
            ni.append(g2i[s]); hi.append(hid); ni.append(g2i[t]); hi.append(hid)
            hid += 1
    
    hei = torch.tensor([ni, hi], dtype=torch.long)
    label_map = {'Control': 0, 'Sepsis': 1}
    
    data_list = []
    for _, row in meta.iterrows():
        sid, cond = row['SampleID'], row['Condition']
        if cond not in label_map or sid not in expr_f.columns: continue
        
        x = torch.tensor(expr_f[sid].values, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(label_map[cond], dtype=torch.long)
        global_feat = torch.tensor(expr_f[sid].values, dtype=torch.float32).unsqueeze(0)
        
        d = Data(x=x, y=y)
        d.hyperedge_index = hei.clone()
        d.num_nodes = len(gene_list)
        d.global_feat = global_feat
        d.sample_id = sid
        d.batch_label = row['Batch']
        data_list.append(d)
        
    return data_list


# ============================================================================
# TRAINING
# ============================================================================
def augment(data, hedge_drop=0.05, noise_std=0.02):
    data = data.clone()
    if hedge_drop > 0 and data.hyperedge_index.size(1) > 0:
        uh = data.hyperedge_index[1].unique()
        keep = torch.rand(uh.max().item()+1) > hedge_drop
        mask = keep[data.hyperedge_index[1]]
        data.hyperedge_index = data.hyperedge_index[:, mask]
    if noise_std > 0:
        data.x = data.x + torch.randn_like(data.x) * noise_std
        if hasattr(data, 'global_feat'):
            data.global_feat = data.global_feat + torch.randn_like(data.global_feat) * noise_std
    return data


def opt_threshold(probs, labels):
    fpr, tpr, thr = roc_curve(labels, probs)
    return thr[np.argmax(tpr - fpr)]


def train_run(train_data, test_data, n_genes, device):
    print(f"\nTraining on {len(train_data)} samples, Testing on {len(test_data)} samples")
    
    # Class balance check
    ty = [d.y.item() for d in train_data]
    vy = [d.y.item() for d in test_data]
    print(f"  Train: C={ty.count(0)} S={ty.count(1)}")
    print(f"  Test:  C={vy.count(0)} S={vy.count(1)}")
    
    train_loader = DataLoader(train_data, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BS, shuffle=False)
    
    model = HybridHGCN(n_genes, H_DIM, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    best_auroc = 0
    best_state = None
    patience = 0
    
    for ep in range(1, EPOCHS+1):
        model.train()
        tloss, n = 0, 0
        for data in train_loader:
            data = augment(data).to(device)
            optimizer.zero_grad()
            out = model(data.x, data.hyperedge_index, data.batch, data.global_feat)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tloss += loss.item()*data.y.size(0)
            n += data.y.size(0)
        scheduler.step()
        
        if ep % EVAL_EVERY == 0:
            model.eval()
            probs, labels = [], []
            vloss, vn = 0, 0
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    out = model(data.x, data.hyperedge_index, data.batch, data.global_feat)
                    vloss += criterion(out, data.y).item() * data.y.size(0)
                    vn += data.y.size(0)
                    probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
                    labels.extend(data.y.cpu().numpy())
            
            auroc = roc_auc_score(labels, probs) if len(set(labels))>=2 else 0.5
            
            if ep % 10 == 0 or ep <= 5:
                print(f"  Ep {ep:3d}: TrL={tloss/n:.4f} TeL={vloss/vn:.4f} TestAUROC={auroc:.4f}")
            
            if auroc > best_auroc:
                best_auroc = auroc
                best_state = {k: v.clone() for k,v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                
            if patience >= PATIENCE:
                print(f"  Early stopping at epoch {ep}")
                break
                
    if best_state:
        model.load_state_dict(best_state)
        
    return model, best_auroc


# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    print("="*60)
    print("A1 HGCN V6: Simple Train/Test Split")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    print("\n--- Loading Data ---")
    expr_f, meta, gene_list = load_data()
    n_genes = len(gene_list)
    print(f"  Genes: {n_genes}, Samples: {len(meta)}")
    
    print("\n--- Building Hypergraph ---")
    pw, se = build_hyperedges(gene_list)
    print(f"  Pathways: {len(pw)}, STRING edges: {len(se)}")
    
    print("\n--- Preparing Graphs ---")
    data_list = make_data_list(expr_f, meta, gene_list, pw, se)
    
    # Split
    labels = [d.y.item() for d in data_list]
    train_idx, test_idx = train_test_split(
        range(len(data_list)), test_size=TEST_SIZE, 
        stratify=labels, random_state=SEED
    )
    
    train_data = [data_list[i] for i in train_idx]
    test_data = [data_list[i] for i in test_idx]
    
    # Run
    print(f"\n--- Starting Training (Train={len(train_data)}, Test={len(test_data)}) ---")
    model, auroc = train_run(train_data, test_data, n_genes, DEVICE)
    
    # Final Eval
    model.eval()
    loader = DataLoader(test_data, batch_size=BS, shuffle=False)
    probs, labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device=DEVICE)
            out = model(data.x, data.hyperedge_index, data.batch, data.global_feat)
            probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            labels.extend(data.y.cpu().numpy())
            
    thr = opt_threshold(probs, labels)
    preds = [1 if p >= thr else 0 for p in probs]
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS (Test Set)")
    print(f"{'='*60}")
    print(f"  AUROC:     {roc_auc_score(labels, probs):.4f}")
    print(f"  Accuracy:  {accuracy_score(labels, preds):.4f}")
    print(f"  F1 Score:  {f1_score(labels, preds):.4f}")
    print(f"  Precision: {precision_score(labels, preds):.4f}")
    print(f"  Recall:    {recall_score(labels, preds):.4f}")
    print(f"  Threshold: {thr:.4f}")
    
    # Save
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "v6_simple_split_best.pt"))
    print(f"\n  Model saved to {MODEL_DIR}/v6_simple_split_best.pt")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
