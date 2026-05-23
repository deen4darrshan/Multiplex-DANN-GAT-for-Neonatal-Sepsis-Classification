"""
A1 HGCN V4: Hybrid GNN-MLP Architecture
=========================================
Uses pre-computed V2 ComBat data (expression_combat_v2.csv, metadata_v2.csv).

KEY INSIGHT: The GNN alone (AUROC ~0.68) cannot match LR (0.998) because:
  - Fixed topology + scalar features loses multivariate signal
  - GNN learns local graph statistics, missing global patterns

SOLUTION: Hybrid architecture that combines:
  1. GNN branch: learns pathway-aware representations via HypergraphConv
  2. MLP branch: learns directly from the full expression vector (like LR)
  3. Fusion: concatenate both branches → final classifier

This way the GNN adds biological structure on top of LR-quality features.

FIXES APPLIED:
  H1+H2: MLP gene embedding for GNN branch
  H3:    Attention readout for GNN branch
  H4:    Extended training (300 epochs, patience=50, min 80 epochs)
  M3:    Runs pathway-only and pathway+STRING ablations
  NEW:   Hybrid GNN+MLP fusion architecture
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve
from scipy.stats import median_abs_deviation

OUT_DIR = os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "results")
MODEL_DIR = os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "models")
PROC_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TOP_K = 2000
H_DIM = 64
DROPOUT = 0.3
BS = 16
EPOCHS = 300
LR = 3e-4
WD = 5e-4
PATIENCE = 50
EVAL_EVERY = 3
MIN_EPOCHS = 80
N_FOLDS = 5
STRING_THR = 700

TIER1 = ['FCGR1A','MMP9','S100A8','S100A9','TLR4','MYD88','IL6','CXCL8','MPO','CEACAM8']


# ============================================================================
# HYBRID MODEL
# ============================================================================
class AttentionPool(nn.Module):
    """Gated attention pooling for graph-level representation."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x, batch):
        scores = self.attn(x)  # (N, 1)
        # Stable softmax per graph
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
    """
    Hybrid model: GNN branch + MLP branch + Fusion.
    
    GNN branch: gene embedding → HypergraphConv × 2 → AttentionPool
    MLP branch: full expression vector → MLP → compact representation
    Fusion: concatenate → classifier
    """
    def __init__(self, n_genes, h_dim=64, dropout=0.3):
        super().__init__()
        
        # --- GNN Branch ---
        self.gene_embed = nn.Sequential(
            nn.Linear(1, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
        )
        self.conv1 = HypergraphConv(h_dim, h_dim)
        self.ln1 = nn.LayerNorm(h_dim)
        self.conv2 = HypergraphConv(h_dim, h_dim)
        self.ln2 = nn.LayerNorm(h_dim)
        self.gnn_pool = AttentionPool(h_dim)
        
        # --- MLP Branch (direct expression features) ---
        self.mlp_branch = nn.Sequential(
            nn.Linear(n_genes, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
        )
        
        # --- Fusion + Classifier ---
        self.fusion = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, 2)
        )
        
        self.dropout = dropout

    def forward(self, x, hyperedge_index, batch, global_feat=None):
        # GNN branch
        g = self.gene_embed(x)
        h = self.conv1(g, hyperedge_index)
        h = self.ln1(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        g = g + h
        
        h = self.conv2(g, hyperedge_index)
        h = self.ln2(h)
        h = F.gelu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        g = g + h
        
        gnn_out = self.gnn_pool(g, batch)  # (B, h_dim)
        
        # MLP branch — uses the full expression vector
        if global_feat is not None:
            mlp_out = self.mlp_branch(global_feat)  # (B, h_dim)
        else:
            # Reconstruct from per-node features — flatten per graph
            # This is less ideal but serves as fallback
            mlp_out = global_mean_pool(x, batch)
            mlp_out = mlp_out.expand(-1, gnn_out.size(1))
        
        # Fusion
        fused = torch.cat([gnn_out, mlp_out], dim=1)
        return self.fusion(fused)


# ============================================================================
# DATA
# ============================================================================
def load_and_prepare():
    """Load V2 data. Returns expression, metadata, filtered expression, gene list."""
    expr = pd.read_csv(os.path.join(OUT_DIR, "expression_combat_v2.csv"), index_col=0)
    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata_v2.csv"))
    
    mad = expr.apply(median_abs_deviation, axis=1)
    top_genes = mad.sort_values(ascending=False).head(TOP_K).index.tolist()
    expr_f = expr.loc[top_genes]
    
    found = [g for g in TIER1 if g in top_genes]
    print(f"  Expr: {expr.shape} → filtered to {expr_f.shape}")
    print(f"  Conditions: {meta['Condition'].value_counts().to_dict()}")
    print(f"  Tier1 biomarkers: {len(found)}/10: {found}")
    
    return expr, meta, expr_f, top_genes


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
    
    print(f"  Pathways: {len(pw)}, STRING: {len(se)}")
    return pw, se


def make_data_list(expr_f, meta, gene_list, pw, se):
    """Build Data objects with BOTH per-node features AND global expression vector."""
    g2i = {g: i for i, g in enumerate(gene_list)}
    
    ni, hi, hid = [], [], 0
    for genes in pw.values():
        for g in genes:
            if g in g2i:
                ni.append(g2i[g])
                hi.append(hid)
        hid += 1
    pw_count = hid
    for s, t in se:
        if s in g2i and t in g2i:
            ni.append(g2i[s]); hi.append(hid)
            ni.append(g2i[t]); hi.append(hid)
            hid += 1
    
    hei = torch.tensor([ni, hi], dtype=torch.long)
    label_map = {'Control': 0, 'Sepsis': 1}
    
    data_list = []
    for _, row in meta.iterrows():
        sid, cond = row['SampleID'], row['Condition']
        if cond not in label_map or sid not in expr_f.columns:
            continue
        
        # Per-node features (for GNN)
        x = torch.tensor(expr_f[sid].values, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(label_map[cond], dtype=torch.long)
        
        # Global expression vector (for MLP branch) — stored as (1, n_genes)
        # so PyG concatenates to (BS, n_genes) properly
        global_feat = torch.tensor(expr_f[sid].values, dtype=torch.float32).unsqueeze(0)
        
        d = Data(x=x, y=y)
        d.hyperedge_index = hei.clone()
        d.num_nodes = len(gene_list)
        d.global_feat = global_feat
        d.sample_id = sid
        d.batch_label = row['Batch']
        data_list.append(d)
    
    labels = [d.y.item() for d in data_list]
    print(f"  Graphs: {len(data_list)} (C={labels.count(0)}, S={labels.count(1)})")
    print(f"  Hyperedges: {pw_count} pathway + {hid-pw_count} STRING = {hid}")
    
    # Verify variation
    if len(data_list) >= 2:
        d = np.abs(data_list[0].x.numpy()-data_list[1].x.numpy()).mean()
        assert d > 0.001, f"Features identical! diff={d}"
    
    return data_list


# PyG handles global_feat automatically since it's stored as (1, n_genes)
# and gets concatenated to (BS, n_genes) during batching.


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


def run_fold(fold, train_data, val_data, n_genes, device, tag):
    print(f"\n--- Fold {fold+1}/{N_FOLDS} [{tag}] ---")
    tl = [d.y.item() for d in train_data]
    vl = [d.y.item() for d in val_data]
    print(f"  Train: {len(train_data)} (C={tl.count(0)}, S={tl.count(1)})")
    print(f"  Val:   {len(val_data)} (C={vl.count(0)}, S={vl.count(1)})")

    train_loader = DataLoader(train_data, batch_size=BS, shuffle=True, )
    val_loader = DataLoader(val_data, batch_size=BS, shuffle=False, )

    model = HybridHGCN(n_genes, H_DIM, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_auroc, best_state, pat = 0, None, 0

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
            probs_all, labels_all, vloss, vn = [], [], 0, 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    out = model(data.x, data.hyperedge_index, data.batch, data.global_feat)
                    vloss += criterion(out, data.y).item()*data.y.size(0)
                    vn += data.y.size(0)
                    p = F.softmax(out, dim=1)[:, 1]
                    probs_all.extend(p.cpu().numpy())
                    labels_all.extend(data.y.cpu().numpy())

            auroc = roc_auc_score(labels_all, probs_all) if len(set(labels_all))>=2 else 0.5

            if ep % 30 == 0 or ep <= 9:
                print(f"    Ep {ep:3d}: TrL={tloss/n:.4f} VaL={vloss/vn:.4f} AUROC={auroc:.4f}")

            if auroc > best_auroc:
                best_auroc = auroc
                best_state = {k: v.clone() for k,v in model.state_dict().items()}
                pat = 0
            else:
                pat += 1

            if ep >= MIN_EPOCHS and pat >= PATIENCE:
                print(f"    Early stop at ep {ep}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Final eval
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.hyperedge_index, data.batch, data.global_feat)
            p = F.softmax(out, dim=1)[:, 1]
            probs_all.extend(p.cpu().numpy())
            labels_all.extend(data.y.cpu().numpy())

    auroc = roc_auc_score(labels_all, probs_all) if len(set(labels_all))>=2 else 0.5
    thr = opt_threshold(probs_all, labels_all)
    preds = [1 if p >= thr else 0 for p in probs_all]
    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds, zero_division=0)
    prec = precision_score(labels_all, preds, zero_division=0)
    rec = recall_score(labels_all, preds, zero_division=0)

    print(f"  → AUROC={auroc:.4f} Acc={acc:.3f} F1={f1:.3f} P={prec:.3f} R={rec:.3f} thr={thr:.3f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"hybrid_v4_{tag}_fold{fold+1}.pt"))

    return model, {'fold': fold+1, 'auroc': auroc, 'accuracy': acc, 'f1': f1,
                   'precision': prec, 'recall': rec, 'threshold': thr}


def run_experiment(data_list, n_genes, tag, device):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {tag}")
    print(f"{'='*60}")

    labels = np.array([d.y.item() for d in data_list])
    batches = np.array([d.batch_label for d in data_list])
    strat = np.array([f"{l}_{b}" for l,b in zip(labels, batches)])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    results = []
    best_a, best_s = 0, None

    for fold, (ti, vi) in enumerate(skf.split(range(len(data_list)), strat)):
        td = [data_list[i] for i in ti]
        vd = [data_list[i] for i in vi]
        model, res = run_fold(fold, td, vd, n_genes, device, tag)
        results.append(res)
        if res['auroc'] > best_a:
            best_a = res['auroc']
            best_s = {k: v.clone().cpu() for k,v in model.state_dict().items()}

    if best_s:
        torch.save(best_s, os.path.join(MODEL_DIR, f"hybrid_v4_{tag}_best.pt"))

    aurocs = [r['auroc'] for r in results]
    f1s = [r['f1'] for r in results]
    print(f"\n  {tag} SUMMARY:")
    print(f"  {'Fold':<6} {'AUROC':<8} {'F1':<8} {'Acc':<8}")
    for r in results:
        s = " ★" if r['auroc']==max(aurocs) else ""
        print(f"  {r['fold']:<6} {r['auroc']:<8.4f} {r['f1']:<8.3f} {r['accuracy']:<8.3f}{s}")
    print(f"  {'Mean':<6} {np.mean(aurocs):<8.4f} {np.mean(f1s):<8.3f}")
    print(f"  {'Std':<6} {np.std(aurocs):<8.4f} {np.std(f1s):<8.3f}")

    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    print("="*60)
    print("A1 HGCN V4: Hybrid GNN+MLP Architecture")
    print(f"Device: {DEVICE}")
    print("="*60)

    print("\n--- Loading data ---")
    expr, meta, expr_f, gene_list = load_and_prepare()
    n_genes = len(gene_list)

    # Experiment: Pathway + STRING
    print("\n--- Building hypergraph (pathway+STRING) ---")
    pw, se = build_hyperedges(gene_list, use_string=True)
    dl_full = make_data_list(expr_f, meta, gene_list, pw, se)
    res_full = run_experiment(dl_full, n_genes, "Hybrid_PW_STR", DEVICE)

    # Experiment: Pathway only
    print("\n--- Building hypergraph (pathway only) ---")
    pw2, _ = build_hyperedges(gene_list, use_string=False)
    dl_pw = make_data_list(expr_f, meta, gene_list, pw2, [])
    res_pw = run_experiment(dl_pw, n_genes, "Hybrid_PW", DEVICE)

    # LR baseline
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = expr_f.T.values
    y = (meta['Condition']=='Sepsis').astype(int).values
    X_s = StandardScaler().fit_transform(X)
    strat = np.array([f"{l}_{b}" for l,b in zip(y, meta['Batch'].values)])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_a = []
    for ti,vi in skf.split(X_s, strat):
        lr = LogisticRegression(max_iter=1000, C=0.01)
        lr.fit(X_s[ti], y[ti])
        lr_a.append(roc_auc_score(y[vi], lr.predict_proba(X_s[vi])[:,1]))

    a_a = [r['auroc'] for r in res_full]
    b_a = [r['auroc'] for r in res_pw]

    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"\n  {'Model':<28} {'AUROC':<15} {'Std':<8}")
    print(f"  {'-'*51}")
    print(f"  {'LR baseline':<28} {np.mean(lr_a):<15.4f} {np.std(lr_a):<8.4f}")
    print(f"  {'Hybrid V4 PW+STR':<28} {np.mean(a_a):<15.4f} {np.std(a_a):<8.4f}")
    print(f"  {'Hybrid V4 PW-only':<28} {np.mean(b_a):<15.4f} {np.std(b_a):<8.4f}")
    print(f"  {'HGCN V3 (attn only)':<28} {'~0.68':<15} {'---':<8}")
    print(f"  {'HGCN V2 (1D scalar)':<28} {'~0.65':<15} {'---':<8}")
    print(f"  {'HGCN V1 (152 uncorr)':<28} {'0.844':<15} {'0.026':<8}")
    print(f"  {'Phase 2 GCN':<28} {'0.681':<15} {'0.048':<8}")

    elapsed = time.time() - t0

    all_res = {
        'experiment': 'A1_Hybrid_V4',
        'hybrid_pw_str': {'mean_auroc': float(np.mean(a_a)), 'std_auroc': float(np.std(a_a)),
                          'mean_f1': float(np.mean([r['f1'] for r in res_full])), 'folds': res_full},
        'hybrid_pw': {'mean_auroc': float(np.mean(b_a)), 'std_auroc': float(np.std(b_a)),
                      'mean_f1': float(np.mean([r['f1'] for r in res_pw])), 'folds': res_pw},
        'lr_baseline': {'mean_auroc': float(np.mean(lr_a)), 'std_auroc': float(np.std(lr_a))},
        'n_samples': len(dl_full), 'n_genes': n_genes,
        'elapsed_min': elapsed/60
    }
    with open(os.path.join(OUT_DIR, "a1_v4_summary.json"), 'w') as f:
        json.dump(all_res, f, indent=2)

    for tag, r in [('pw_str', res_full), ('pw', res_pw)]:
        pd.DataFrame(r).to_csv(os.path.join(OUT_DIR, f"a1_v4_{tag}.csv"), index=False)

    print(f"\n  Time: {elapsed/60:.1f} min")
    print(f"  Results: {OUT_DIR}/a1_v4_*.json/csv")

    # CoVe
    print(f"\n{'='*60}")
    print("CoVe GATE")
    print(f"{'='*60}")
    ma = np.mean(a_a)
    mf = np.mean([r['f1'] for r in res_full])
    if ma >= 0.78: print(f"  ✓ AUROC {ma:.4f} >= 0.78")
    elif ma >= 0.68: print(f"  ⚠ AUROC {ma:.4f} >= 0.68")
    else: print(f"  ✗ AUROC {ma:.4f} < 0.68")
    if mf >= 0.5: print(f"  ✓ F1 {mf:.4f}")
    else: print(f"  ⚠ F1 {mf:.4f}")
    if np.std(a_a) < 0.05: print(f"  ✓ Std {np.std(a_a):.4f}")
    else: print(f"  ⚠ Std {np.std(a_a):.4f}")


if __name__ == "__main__":
    main()
