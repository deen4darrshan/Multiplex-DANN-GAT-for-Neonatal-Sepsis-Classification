"""
A1 HGCN V7: Fixed Cross-Validation (StratifiedGroupKFold / LOBO)
=================================================================
Implements Solution A from the CV Inflation Diagnosis:
  - Uses StratifiedGroupKFold where groups = Batch labels
  - This ensures NO batch leakage between Train and Validation
  - With 4 batches, we get 4 folds (Leave-One-Batch-Out)

Key Features:
  - Per-epoch metrics: Loss, Accuracy, AUROC, F1 printed EVERY epoch
  - Model saving: Best model saved per fold + overall best
  - Full results table at the end
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, roc_curve)
from scipy.stats import median_abs_deviation

# ============================================================================
# CONFIGURATION
# ============================================================================
OUT_DIR   = os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "results")
MODEL_DIR = os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "models")
PROC_DIR  = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
TOP_K      = 2000
H_DIM      = 64
DROPOUT    = 0.3
BS         = 16
EPOCHS     = 150
LR         = 3e-4
WD         = 5e-4
PATIENCE   = 30
STRING_THR = 700
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================================
# MODEL
# ============================================================================
class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1)
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
        self.ln1   = nn.LayerNorm(h_dim)
        self.conv2 = HypergraphConv(h_dim, h_dim)
        self.ln2   = nn.LayerNorm(h_dim)
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
        if global_feat is not None:
            mlp_out = self.mlp_branch(global_feat)
        else:
            mlp_out = global_mean_pool(x, batch)
            mlp_out = mlp_out.expand(-1, gnn_out.size(1))
        fused = torch.cat([gnn_out, mlp_out], dim=1)
        return self.classifier(fused)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    expr = pd.read_csv(os.path.join(OUT_DIR, "expression_combat_v2.csv"), index_col=0)
    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata_v2.csv"))
    mad  = expr.apply(median_abs_deviation, axis=1)
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
            pf = ppi[(ppi['source'].isin(gene_set)) &
                      (ppi['target'].isin(gene_set)) &
                      (ppi['score'] >= STRING_THR)]
            se = list(zip(pf['source'].tolist(), pf['target'].tolist()))
    return pw, se


def make_data_list(expr_f, meta, gene_list, pw, se):
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
        sid, cond = row['SampleID'], row['Condition']
        if cond not in label_map or sid not in expr_f.columns:
            continue
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
# AUGMENTATION
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


# ============================================================================
# EVALUATION HELPERS
# ============================================================================
def compute_metrics(probs, labels, threshold=0.5):
    """Compute all metrics given probabilities and labels."""
    if len(set(labels)) < 2:
        return {'auroc': float('nan'), 'acc': float('nan'), 'f1': float('nan'),
                'prec': float('nan'), 'rec': float('nan'), 'threshold': threshold}
    auroc = roc_auc_score(labels, probs)
    preds = [1 if p >= threshold else 0 for p in probs]
    return {
        'auroc': auroc,
        'acc': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, zero_division=0),
        'prec': precision_score(labels, preds, zero_division=0),
        'rec': recall_score(labels, preds, zero_division=0),
        'threshold': threshold,
    }


def opt_threshold(probs, labels):
    fpr, tpr, thr = roc_curve(labels, probs)
    return thr[np.argmax(tpr - fpr)]


def evaluate_model(model, loader, criterion, device):
    """Evaluate model on a DataLoader. Returns loss, probs, labels."""
    model.eval()
    all_probs, all_labels = [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.hyperedge_index, data.batch, data.global_feat)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.y.size(0)
            n += data.y.size(0)
            all_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    return total_loss / max(n, 1), all_probs, all_labels


# ============================================================================
# SINGLE FOLD TRAINING
# ============================================================================
def train_fold(fold_name, train_data, val_data, n_genes, device):
    """Train one fold. Prints metrics EVERY epoch. Returns best model + metrics."""

    n_train, n_val = len(train_data), len(val_data)
    ty = [d.y.item() for d in train_data]
    vy = [d.y.item() for d in val_data]

    print(f"\n{'='*80}")
    print(f"  FOLD: {fold_name}")
    print(f"  Train: {n_train} samples (C={ty.count(0)}, S={ty.count(1)})")
    print(f"  Val:   {n_val} samples (C={vy.count(0)}, S={vy.count(1)})")
    print(f"{'='*80}")

    # Check if val set has both classes
    if len(set(vy)) < 2:
        print(f"  ⚠ Validation has only class {set(vy)} — skipping fold")
        return None, None

    train_loader = DataLoader(train_data, batch_size=BS, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=BS, shuffle=False)

    model     = HybridHGCN(n_genes, H_DIM, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_auroc = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    # Header
    print(f"  {'Ep':>4s} | {'TrLoss':>7s} {'TrAcc':>7s} {'TrAUC':>7s} | "
          f"{'VaLoss':>7s} {'VaAcc':>7s} {'VaAUC':>7s} {'VaF1':>6s} | {'Best':>7s} {'Pat':>3s}")
    print(f"  {'-'*4}-+-{'-'*7}-{'-'*7}-{'-'*7}-+-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*6}-+-{'-'*7}-{'-'*3}")

    for ep in range(1, EPOCHS + 1):
        # --- TRAIN ---
        model.train()
        train_loss, train_n = 0.0, 0
        train_probs, train_labels = [], []

        for data in train_loader:
            data = augment(data).to(device)
            optimizer.zero_grad()
            out = model(data.x, data.hyperedge_index, data.batch, data.global_feat)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * data.y.size(0)
            train_n += data.y.size(0)
            with torch.no_grad():
                train_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
                train_labels.extend(data.y.cpu().numpy())

        scheduler.step()

        # Train metrics
        tr_loss = train_loss / max(train_n, 1)
        tr_m = compute_metrics(train_probs, train_labels)

        # --- VALIDATE ---
        va_loss, va_probs, va_labels = evaluate_model(model, val_loader, criterion, device)
        va_m = compute_metrics(va_probs, va_labels)

        # Track best
        improved = ""
        if va_m['auroc'] > best_auroc:
            best_auroc = va_m['auroc']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            patience_counter = 0
            improved = " ★"
        else:
            patience_counter += 1

        # Print EVERY epoch
        print(f"  {ep:4d} | {tr_loss:7.4f} {tr_m['acc']:7.4f} {tr_m['auroc']:7.4f} | "
              f"{va_loss:7.4f} {va_m['acc']:7.4f} {va_m['auroc']:7.4f} {va_m['f1']:6.3f} | "
              f"{best_auroc:7.4f} {patience_counter:3d}{improved}")

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"  → Early stopping at epoch {ep} (best was epoch {best_epoch})")
            break

    # Load best
    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation with optimal threshold
    _, final_probs, final_labels = evaluate_model(model, val_loader, criterion, device)
    thr = opt_threshold(final_probs, final_labels)
    final_m = compute_metrics(final_probs, final_labels, threshold=thr)

    print(f"\n  ── Best Epoch {best_epoch} (Optimal Threshold = {thr:.4f}) ──")
    print(f"  AUROC:     {final_m['auroc']:.4f}")
    print(f"  Accuracy:  {final_m['acc']:.4f}")
    print(f"  F1 Score:  {final_m['f1']:.4f}")
    print(f"  Precision: {final_m['prec']:.4f}")
    print(f"  Recall:    {final_m['rec']:.4f}")

    return model, final_m


# ============================================================================
# MAIN: StratifiedGroupKFold (LOBO)
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("  A1 HGCN V7: Fixed Cross-Validation (StratifiedGroupKFold / LOBO)")
    print(f"  Device: {DEVICE}  |  Seed: {SEED}  |  Epochs: {EPOCHS}")
    print("=" * 80)

    # ── Load Data ──
    print("\n[1/4] Loading Data...")
    expr_f, meta, gene_list = load_data()
    n_genes = len(gene_list)
    print(f"  Genes: {n_genes}, Samples: {len(meta)}")
    print(f"  Conditions: {meta['Condition'].value_counts().to_dict()}")
    print(f"  Batches:    {meta['Batch'].value_counts().to_dict()}")

    # ── Build Hypergraph ──
    print("\n[2/4] Building Hypergraph...")
    pw, se = build_hyperedges(gene_list)
    print(f"  KEGG Pathways: {len(pw)}, STRING Edges: {len(se)}")

    # ── Build Graph Data ──
    print("\n[3/4] Preparing Patient Graphs...")
    data_list = make_data_list(expr_f, meta, gene_list, pw, se)
    labels  = np.array([d.y.item() for d in data_list])
    batches = np.array([d.batch_label for d in data_list])
    print(f"  Total graphs: {len(data_list)}")
    print(f"  Label distribution: C={sum(labels==0)}, S={sum(labels==1)}")

    # ── StratifiedGroupKFold ──
    print("\n[4/4] Running StratifiedGroupKFold (Groups = Batch)...")
    unique_batches = sorted(set(batches))
    n_groups = len(unique_batches)
    print(f"  Unique batches (groups): {unique_batches}")
    print(f"  Number of folds: {n_groups} (one per batch = LOBO)")

    sgkf = StratifiedGroupKFold(n_splits=n_groups, shuffle=True, random_state=SEED)

    all_results = []
    best_overall_auroc = 0.0
    best_overall_state = None
    best_overall_fold  = ""

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(range(len(data_list)), labels, batches)):
        train_data = [data_list[i] for i in train_idx]
        val_data   = [data_list[i] for i in val_idx]

        # Identify held-out batch(es)
        held_out = sorted(set(batches[val_idx]))
        fold_name = f"Fold {fold_idx+1}/{n_groups} — Test on {', '.join(held_out)}"

        model, metrics = train_fold(fold_name, train_data, val_data, n_genes, DEVICE)

        if model is None or metrics is None:
            print(f"  ⚠ Fold {fold_idx+1} skipped (insufficient classes in val)")
            all_results.append({
                'fold': fold_idx + 1, 'held_out': ', '.join(held_out),
                'auroc': float('nan'), 'acc': float('nan'), 'f1': float('nan'),
                'prec': float('nan'), 'rec': float('nan'), 'n_val': len(val_data),
                'skipped': True
            })
            continue

        # Save per-fold model
        fold_path = os.path.join(MODEL_DIR, f"v7_sgkf_fold{fold_idx+1}.pt")
        torch.save(model.state_dict(), fold_path)
        print(f"  Model saved → {fold_path}")

        # Track overall best
        if metrics['auroc'] > best_overall_auroc:
            best_overall_auroc = metrics['auroc']
            best_overall_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_overall_fold  = fold_name

        all_results.append({
            'fold': fold_idx + 1,
            'held_out': ', '.join(held_out),
            'auroc': metrics['auroc'],
            'acc': metrics['acc'],
            'f1': metrics['f1'],
            'prec': metrics['prec'],
            'rec': metrics['rec'],
            'n_val': len(val_data),
            'skipped': False
        })

    # ── Save overall best model ──
    if best_overall_state:
        best_path = os.path.join(MODEL_DIR, "v7_sgkf_best.pt")
        torch.save(best_overall_state, best_path)
        print(f"\n  ★ Overall best model saved → {best_path}")
        print(f"    From: {best_overall_fold} (AUROC={best_overall_auroc:.4f})")

    # ══════════════════════════════════════════
    # FINAL SUMMARY TABLE
    # ══════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  FINAL RESULTS: StratifiedGroupKFold / LOBO")
    print(f"{'='*80}")
    print(f"  {'Fold':>4s} | {'Held-Out Batch':<20s} | {'N':>4s} | {'AUROC':>7s} | {'Acc':>7s} | {'F1':>6s} | {'Prec':>6s} | {'Rec':>6s}")
    print(f"  {'-'*4}-+-{'-'*20}-+-{'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")

    valid_aurocs, valid_accs, valid_f1s = [], [], []
    for r in all_results:
        if r['skipped']:
            print(f"  {r['fold']:4d} | {r['held_out']:<20s} | {r['n_val']:4d} | {'SKIP':>7s} | {'SKIP':>7s} | {'SKIP':>6s} | {'SKIP':>6s} | {'SKIP':>6s}")
        else:
            print(f"  {r['fold']:4d} | {r['held_out']:<20s} | {r['n_val']:4d} | {r['auroc']:7.4f} | {r['acc']:7.4f} | {r['f1']:6.3f} | {r['prec']:6.3f} | {r['rec']:6.3f}")
            if not np.isnan(r['auroc']):
                valid_aurocs.append(r['auroc'])
                valid_accs.append(r['acc'])
                valid_f1s.append(r['f1'])

    if valid_aurocs:
        print(f"  {'-'*4}-+-{'-'*20}-+-{'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
        print(f"  {'Mean':>4s} | {'':20s} | {'':4s} | {np.mean(valid_aurocs):7.4f} | {np.mean(valid_accs):7.4f} | {np.mean(valid_f1s):6.3f} |")
        print(f"  {'Std':>4s}  | {'':20s} | {'':4s} | {np.std(valid_aurocs):7.4f} | {np.std(valid_accs):7.4f} | {np.std(valid_f1s):6.3f} |")

    # Save results to JSON
    summary = {
        'method': 'StratifiedGroupKFold (LOBO)',
        'n_folds': n_groups,
        'seed': SEED,
        'epochs': EPOCHS,
        'folds': all_results,
        'mean_auroc': float(np.mean(valid_aurocs)) if valid_aurocs else None,
        'std_auroc': float(np.std(valid_aurocs)) if valid_aurocs else None,
        'mean_acc': float(np.mean(valid_accs)) if valid_accs else None,
        'std_acc': float(np.std(valid_accs)) if valid_accs else None,
        'best_fold': best_overall_fold,
        'best_auroc': best_overall_auroc,
        'total_time_min': (time.time() - t0) / 60,
    }

    json_path = os.path.join(OUT_DIR, "v7_sgkf_results.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    elapsed = (time.time() - t0) / 60
    print(f"\n  Total execution time: {elapsed:.1f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()
