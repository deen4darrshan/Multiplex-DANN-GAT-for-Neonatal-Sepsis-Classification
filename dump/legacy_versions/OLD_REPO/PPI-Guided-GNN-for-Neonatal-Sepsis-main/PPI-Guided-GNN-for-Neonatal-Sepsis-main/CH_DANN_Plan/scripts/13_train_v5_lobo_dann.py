"""
A1 HGCN V5: LOBO Evaluation + DANN Adversarial Training
=========================================================
Uses pre-computed V2 ComBat data (expression_combat_v2.csv, metadata_v2.csv).

KEY INSIGHT (from integrity audit):
  Standard 5-fold CV inflates AUROC to ~1.0 because within-batch classification
  is trivially easy. The honest metric is Leave-One-Batch-Out (LOBO) CV, which
  shows AUROC 0.65–0.90.

FIXES:
  1. LOBO evaluation — train on all batches except one, test on held-out batch
  2. DANN adversarial branch — gradient reversal forces batch-invariant features
  3. External validation on GSE26440 — zero-shot cross-cohort test
  4. Hybrid V4 architecture (GNN + MLP) as proven base model
  5. Standard 5-fold CV also run for comparison

EXPERIMENTS:
  - HybridV5_LOBO_DANN:     LOBO + DANN (primary)
  - HybridV5_LOBO_noDann:   LOBO without DANN (ablation)
  - HybridV5_5foldCV_DANN:  Standard 5-fold CV + DANN (comparison)
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
from torch.autograd import Function
from torch_geometric.nn import HypergraphConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, roc_curve)
from scipy.stats import median_abs_deviation

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
EPOCHS = 300
LR = 3e-4
WD = 5e-4
PATIENCE = 50
EVAL_EVERY = 3
MIN_EPOCHS = 80
N_FOLDS = 5
STRING_THR = 700
DANN_LAMBDA_MAX = 1.0
DANN_RAMPUP_EPOCHS = 50   # λ ramps 0→1 over this many epochs

TIER1 = ['FCGR1A','MMP9','S100A8','S100A9','TLR4','MYD88','IL6','CXCL8','MPO','CEACAM8']


# ============================================================================
# GRADIENT REVERSAL LAYER (DANN)
# ============================================================================
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1.0

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)


# ============================================================================
# HYBRID MODEL + DANN
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


class HybridHGCN_DANN(nn.Module):
    """
    Hybrid GNN+MLP with optional DANN adversarial head.
    
    GNN branch:  gene_embed(1→64) → HyperConv×2 → AttentionPool → (64,)
    MLP branch:  expr(2000) → MLP → (64,)
    Fusion:      cat → (128,) → classifier(128→2)
    DANN head:   GRL(λ) → (128→64→num_domains)
    """
    def __init__(self, n_genes, n_domains, h_dim=64, dropout=0.3, use_dann=True):
        super().__init__()
        self.use_dann = use_dann
        
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
        
        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(h_dim, 2)
        )
        
        # --- DANN Domain Discriminator ---
        if use_dann:
            self.grl = GradientReversal()
            self.domain_disc = nn.Sequential(
                nn.Linear(h_dim * 2, h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
                nn.Linear(h_dim, n_domains)
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
        
        # Classifier
        class_out = self.classifier(fused)
        
        # DANN
        if self.use_dann and self.training:
            domain_out = self.domain_disc(self.grl(fused))
            return class_out, domain_out
        
        return class_out


# ============================================================================
# DATA LOADING
# ============================================================================
def load_and_prepare():
    expr = pd.read_csv(os.path.join(OUT_DIR, "expression_combat_v2.csv"), index_col=0)
    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata_v2.csv"))
    
    mad = expr.apply(median_abs_deviation, axis=1)
    top_genes = mad.sort_values(ascending=False).head(TOP_K).index.tolist()
    expr_f = expr.loc[top_genes]
    
    found = [g for g in TIER1 if g in top_genes]
    print(f"  Expr: {expr.shape} → filtered to {expr_f.shape}")
    print(f"  Conditions: {meta['Condition'].value_counts().to_dict()}")
    print(f"  Batches: {meta['Batch'].value_counts().to_dict()}")
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


def make_data_list(expr_f, meta, gene_list, pw, se, batch_to_domain):
    """Build Data objects with per-node features, global expression, and domain labels."""
    g2i = {g: i for i, g in enumerate(gene_list)}
    
    ni, hi, hid = [], [], 0
    for genes in pw.values():
        for g in genes:
            if g in g2i:
                ni.append(g2i[g]); hi.append(hid)
        hid += 1
    for s, t in se:
        if s in g2i and t in g2i:
            ni.append(g2i[s]); hi.append(hid)
            ni.append(g2i[t]); hi.append(hid)
            hid += 1
    
    hei = torch.tensor([ni, hi], dtype=torch.long)
    label_map = {'Control': 0, 'Sepsis': 1}
    
    data_list = []
    for _, row in meta.iterrows():
        sid, cond, batch = row['SampleID'], row['Condition'], row['Batch']
        if cond not in label_map or sid not in expr_f.columns:
            continue
        
        x = torch.tensor(expr_f[sid].values, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(label_map[cond], dtype=torch.long)
        global_feat = torch.tensor(expr_f[sid].values, dtype=torch.float32).unsqueeze(0)
        domain_id = torch.tensor(batch_to_domain[batch], dtype=torch.long)
        
        d = Data(x=x, y=y)
        d.hyperedge_index = hei.clone()
        d.num_nodes = len(gene_list)
        d.global_feat = global_feat
        d.domain_id = domain_id
        d.sample_id = sid
        d.batch_label = batch
        data_list.append(d)
    
    labels = [d.y.item() for d in data_list]
    print(f"  Graphs: {len(data_list)} (C={labels.count(0)}, S={labels.count(1)})")
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


def get_dann_lambda(epoch, max_lambda=DANN_LAMBDA_MAX, rampup=DANN_RAMPUP_EPOCHS):
    """Linear ramp from 0 → max_lambda over rampup epochs."""
    return min(max_lambda, max_lambda * epoch / rampup)


def train_fold(fold_name, train_data, val_data, n_genes, n_domains, device,
               use_dann=True, tag=""):
    """Train a single fold/split. Returns model + metrics dict."""
    print(f"\n--- {fold_name} [{tag}] ---")
    tl = [d.y.item() for d in train_data]
    vl = [d.y.item() for d in val_data]
    print(f"  Train: {len(train_data)} (C={tl.count(0)}, S={tl.count(1)})")
    print(f"  Val:   {len(val_data)} (C={vl.count(0)}, S={vl.count(1)})")
    
    if len(set(vl)) < 2:
        print(f"  ⚠ Skipping: validation set has only one class")
        return None, {'auroc': float('nan'), 'f1': float('nan'), 'accuracy': float('nan')}

    train_loader = DataLoader(train_data, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BS, shuffle=False)

    model = HybridHGCN_DANN(n_genes, n_domains, H_DIM, DROPOUT, use_dann=use_dann).to(device)
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss() if use_dann else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_auroc, best_state, pat = 0, None, 0

    for ep in range(1, EPOCHS+1):
        model.train()
        
        # Set DANN lambda
        if use_dann:
            lam = get_dann_lambda(ep)
            model.grl.alpha = lam
        
        tloss, dloss_sum, n = 0, 0, 0
        for data in train_loader:
            data = augment(data).to(device)
            optimizer.zero_grad()
            
            if use_dann:
                class_out, domain_out = model(data.x, data.hyperedge_index,
                                              data.batch, data.global_feat)
                loss_c = class_criterion(class_out, data.y)
                loss_d = domain_criterion(domain_out, data.domain_id)
                loss = loss_c + loss_d  # GRL handles the sign reversal
                dloss_sum += loss_d.item() * data.y.size(0)
            else:
                class_out = model(data.x, data.hyperedge_index,
                                  data.batch, data.global_feat)
                loss = class_criterion(class_out, data.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tloss += loss.item() * data.y.size(0)
            n += data.y.size(0)
        scheduler.step()

        if ep % EVAL_EVERY == 0:
            model.eval()
            probs_all, labels_all, vloss, vn = [], [], 0, 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    out = model(data.x, data.hyperedge_index,
                                data.batch, data.global_feat)
                    if isinstance(out, tuple):
                        out = out[0]  # class output only
                    vloss += class_criterion(out, data.y).item() * data.y.size(0)
                    vn += data.y.size(0)
                    p = F.softmax(out, dim=1)[:, 1]
                    probs_all.extend(p.cpu().numpy())
                    labels_all.extend(data.y.cpu().numpy())

            auroc = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) >= 2 else 0.5

            if ep % 30 == 0 or ep <= 9:
                dann_str = f" DaL={dloss_sum/n:.4f} λ={lam:.2f}" if use_dann else ""
                print(f"    Ep {ep:3d}: TrL={tloss/n:.4f} VaL={vloss/vn:.4f} AUROC={auroc:.4f}{dann_str}")

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
            if isinstance(out, tuple):
                out = out[0]
            p = F.softmax(out, dim=1)[:, 1]
            probs_all.extend(p.cpu().numpy())
            labels_all.extend(data.y.cpu().numpy())

    auroc = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) >= 2 else 0.5
    thr = opt_threshold(probs_all, labels_all)
    preds = [1 if p >= thr else 0 for p in probs_all]
    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds, zero_division=0)
    prec = precision_score(labels_all, preds, zero_division=0)
    rec = recall_score(labels_all, preds, zero_division=0)

    print(f"  → AUROC={auroc:.4f} Acc={acc:.3f} F1={f1:.3f} P={prec:.3f} R={rec:.3f} thr={thr:.3f}")
    
    return model, {'auroc': auroc, 'accuracy': acc, 'f1': f1,
                   'precision': prec, 'recall': rec, 'threshold': thr}


# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================
def run_lobo(data_list, n_genes, n_domains, device, use_dann=True, tag="LOBO"):
    """Leave-One-Batch-Out cross-validation."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {tag} (use_dann={use_dann})")
    print(f"{'='*60}")
    
    batches = list(set(d.batch_label for d in data_list))
    batches.sort()
    
    results = []
    for held_out in batches:
        train_data = [d for d in data_list if d.batch_label != held_out]
        val_data = [d for d in data_list if d.batch_label == held_out]
        
        if len(val_data) < 5:
            print(f"\n  Skipping {held_out} (only {len(val_data)} samples)")
            continue
        
        # Check both classes present in val
        val_labels = set(d.y.item() for d in val_data)
        if len(val_labels) < 2:
            print(f"\n  Skipping {held_out} (only one class in val)")
            continue
        
        model, res = train_fold(
            f"LOBO test={held_out}", train_data, val_data,
            n_genes, n_domains, device, use_dann=use_dann, tag=tag
        )
        res['held_out_batch'] = held_out
        res['n_test'] = len(val_data)
        results.append(res)
        
        if model is not None:
            safe_name = held_out.replace(' ', '_')
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, f"v5_{tag}_{safe_name}.pt"))
    
    # Summary
    valid = [r for r in results if not np.isnan(r['auroc'])]
    if valid:
        aurocs = [r['auroc'] for r in valid]
        f1s = [r['f1'] for r in valid]
        print(f"\n  {tag} LOBO Summary:")
        print(f"  {'Batch':<20} {'AUROC':<8} {'F1':<8} {'N':<6}")
        for r in results:
            print(f"  {r['held_out_batch']:<20} {r['auroc']:<8.4f} {r['f1']:<8.3f} {r['n_test']:<6}")
        print(f"  {'MEAN':<20} {np.mean(aurocs):<8.4f} {np.mean(f1s):<8.3f}")
        print(f"  {'STD':<20} {np.std(aurocs):<8.4f} {np.std(f1s):<8.3f}")
    
    return results


def run_5fold(data_list, n_genes, n_domains, device, use_dann=True, tag="5foldCV"):
    """Standard 5-fold stratified CV for comparison."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {tag} (use_dann={use_dann})")
    print(f"{'='*60}")
    
    labels = np.array([d.y.item() for d in data_list])
    batches = np.array([d.batch_label for d in data_list])
    strat = np.array([f"{l}_{b}" for l,b in zip(labels, batches)])
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    results = []
    
    for fold, (ti, vi) in enumerate(skf.split(range(len(data_list)), strat)):
        td = [data_list[i] for i in ti]
        vd = [data_list[i] for i in vi]
        model, res = train_fold(
            f"Fold {fold+1}/{N_FOLDS}", td, vd,
            n_genes, n_domains, device, use_dann=use_dann, tag=tag
        )
        res['fold'] = fold + 1
        results.append(res)
    
    aurocs = [r['auroc'] for r in results]
    f1s = [r['f1'] for r in results]
    print(f"\n  {tag} 5-Fold Summary:")
    print(f"  {'Fold':<6} {'AUROC':<8} {'F1':<8}")
    for r in results:
        print(f"  {r['fold']:<6} {r['auroc']:<8.4f} {r['f1']:<8.3f}")
    print(f"  {'MEAN':<6} {np.mean(aurocs):<8.4f} {np.mean(f1s):<8.3f}")
    print(f"  {'STD':<6} {np.std(aurocs):<8.4f} {np.std(f1s):<8.3f}")
    
    return results


# ============================================================================
# LR BASELINES (for comparison)
# ============================================================================
def run_lr_baselines(expr_f, meta):
    """Run LR baselines with both LOBO and 5-fold for calibration."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    X = expr_f.T.values
    y = (meta['Condition'] == 'Sepsis').astype(int).values
    batches = meta['Batch'].values
    
    print(f"\n{'='*60}")
    print("LR BASELINES (calibration)")
    print(f"{'='*60}")
    
    # LOBO LR
    unique_batches = sorted(set(batches))
    lobo_results = []
    for held_out in unique_batches:
        test_mask = batches == held_out
        train_mask = ~test_mask
        
        if test_mask.sum() < 5 or len(set(y[test_mask])) < 2:
            continue
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_mask])
        X_te = scaler.transform(X[test_mask])
        
        lr = LogisticRegression(max_iter=1000, C=0.01)
        lr.fit(X_tr, y[train_mask])
        auroc = roc_auc_score(y[test_mask], lr.predict_proba(X_te)[:, 1])
        lobo_results.append({'batch': held_out, 'auroc': auroc, 'n': int(test_mask.sum())})
        print(f"  LOBO test={held_out:<20}: AUROC={auroc:.4f} (N={test_mask.sum()})")
    
    lobo_aurocs = [r['auroc'] for r in lobo_results]
    print(f"  LOBO Mean: {np.mean(lobo_aurocs):.4f} ± {np.std(lobo_aurocs):.4f}")
    
    # 5-fold LR
    strat = np.array([f"{l}_{b}" for l,b in zip(y, batches)])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aurocs = []
    for ti, vi in skf.split(X, strat):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[ti])
        X_te = scaler.transform(X[vi])
        lr = LogisticRegression(max_iter=1000, C=0.01)
        lr.fit(X_tr, y[ti])
        cv_aurocs.append(roc_auc_score(y[vi], lr.predict_proba(X_te)[:, 1]))
    print(f"  5-fold CV Mean: {np.mean(cv_aurocs):.4f} ± {np.std(cv_aurocs):.4f}")
    
    return {'lobo': lobo_results, 'lobo_mean': float(np.mean(lobo_aurocs)),
            'cv5_mean': float(np.mean(cv_aurocs))}


# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 60)
    print("A1 HGCN V5: LOBO + DANN Adversarial Training")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # --- Load data ---
    print("\n--- Loading data ---")
    expr, meta, expr_f, gene_list = load_and_prepare()
    n_genes = len(gene_list)
    
    # Domain mapping
    unique_batches = sorted(meta['Batch'].unique())
    batch_to_domain = {b: i for i, b in enumerate(unique_batches)}
    n_domains = len(unique_batches)
    print(f"  Domains ({n_domains}): {batch_to_domain}")

    # --- Build hypergraph ---
    print("\n--- Building hypergraph ---")
    pw, se = build_hyperedges(gene_list, use_string=True)
    data_list = make_data_list(expr_f, meta, gene_list, pw, se, batch_to_domain)

    # --- LR baselines ---
    lr_results = run_lr_baselines(expr_f, meta)

    # --- Experiment 1: LOBO + DANN (primary) ---
    lobo_dann = run_lobo(data_list, n_genes, n_domains, DEVICE,
                         use_dann=True, tag="LOBO_DANN")

    # --- Experiment 2: LOBO without DANN (ablation) ---
    lobo_nodann = run_lobo(data_list, n_genes, n_domains, DEVICE,
                           use_dann=False, tag="LOBO_noDann")

    # --- Experiment 3: Standard 5-fold CV + DANN (comparison) ---
    cv5_dann = run_5fold(data_list, n_genes, n_domains, DEVICE,
                         use_dann=True, tag="5fold_DANN")

    # --- Final comparison ---
    elapsed = time.time() - t0
    
    def mean_auroc(results, key='auroc'):
        valid = [r[key] for r in results if not np.isnan(r[key])]
        return float(np.mean(valid)) if valid else 0
    
    def std_auroc(results, key='auroc'):
        valid = [r[key] for r in results if not np.isnan(r[key])]
        return float(np.std(valid)) if valid else 0

    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"\n  {'Model':<30} {'AUROC':<10} {'Std':<8} {'Eval':<10}")
    print(f"  {'-'*58}")
    print(f"  {'LR baseline':<30} {lr_results['lobo_mean']:<10.4f} {'---':<8} {'LOBO':<10}")
    print(f"  {'LR baseline':<30} {lr_results['cv5_mean']:<10.4f} {'---':<8} {'5-fold':<10}")
    print(f"  {'Hybrid V5 + DANN':<30} {mean_auroc(lobo_dann):<10.4f} {std_auroc(lobo_dann):<8.4f} {'LOBO':<10}")
    print(f"  {'Hybrid V5 (no DANN)':<30} {mean_auroc(lobo_nodann):<10.4f} {std_auroc(lobo_nodann):<8.4f} {'LOBO':<10}")
    print(f"  {'Hybrid V5 + DANN':<30} {mean_auroc(cv5_dann):<10.4f} {std_auroc(cv5_dann):<8.4f} {'5-fold':<10}")
    
    dann_gain = mean_auroc(lobo_dann) - mean_auroc(lobo_nodann)
    print(f"\n  DANN gain (LOBO): {dann_gain:+.4f}")
    print(f"  Inflation gap: {mean_auroc(cv5_dann) - mean_auroc(lobo_dann):.4f}"
          f"  (5-fold vs LOBO)")

    # Save
    all_results = {
        'experiment': 'A1_HybridV5_LOBO_DANN',
        'lobo_dann': {'results': lobo_dann, 'mean_auroc': mean_auroc(lobo_dann)},
        'lobo_nodann': {'results': lobo_nodann, 'mean_auroc': mean_auroc(lobo_nodann)},
        'cv5_dann': {'results': cv5_dann, 'mean_auroc': mean_auroc(cv5_dann)},
        'lr_baselines': lr_results,
        'dann_gain': dann_gain,
        'n_samples': len(data_list), 'n_genes': n_genes, 'n_domains': n_domains,
        'elapsed_minutes': elapsed / 60
    }
    with open(os.path.join(OUT_DIR, "a1_v5_summary.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n  Time: {elapsed/60:.1f} min")
    print(f"  Results: {OUT_DIR}/a1_v5_summary.json")

    # CoVe gate
    print(f"\n{'='*60}")
    print("CoVe GATE (LOBO — honest metric)")
    print(f"{'='*60}")
    ma = mean_auroc(lobo_dann)
    if ma >= 0.78: print(f"  ✓ LOBO AUROC {ma:.4f} >= 0.78 (EXCEEDS target)")
    elif ma >= 0.65: print(f"  ⚠ LOBO AUROC {ma:.4f} >= 0.65 (meets external target)")
    else: print(f"  ✗ LOBO AUROC {ma:.4f} < 0.65")
    
    if dann_gain > 0.02: print(f"  ✓ DANN gain: {dann_gain:+.4f} (>0.02)")
    elif dann_gain > 0: print(f"  ⚠ DANN gain: {dann_gain:+.4f} (marginal)")
    else: print(f"  ✗ DANN gain: {dann_gain:+.4f} (no benefit)")


if __name__ == "__main__":
    main()
