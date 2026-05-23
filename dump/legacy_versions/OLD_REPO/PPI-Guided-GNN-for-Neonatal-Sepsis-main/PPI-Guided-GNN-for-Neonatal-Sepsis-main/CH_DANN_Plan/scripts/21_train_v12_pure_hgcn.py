"""
V12 Pure HGCN: Multiplex HypergraphConv + DANN (No MLP)
=====================================================================
Three relation layers (KEGG pathways, STRING PPI, co-expression)
on the same 2,000 gene nodes. Relation-aware attention aggregates them.
Instead of an MLP, a global mean pool reduces the graph to a single 
embedding which is fed to the Sepsis Classifier and Domain Discriminator.
"""

import os, sys, time, json, warnings
warnings.filterwarnings('ignore')

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
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score)
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

TOP_K      = 2000
H_DIM      = 64
DROPOUT    = 0.3
BS         = 16
EPOCHS     = 150
LR         = 3e-4
WD         = 5e-4
PATIENCE   = 30
STRING_THR = 700
COEXPR_THR = 0.7   
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================================
# GRADIENT REVERSAL LAYER
# ============================================================================
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

# ============================================================================
# MODEL: Pure Multiplex HGCN + DANN (No MLP)
# ============================================================================
class PureMultiplexHGCNDANN(nn.Module):
    """
    3 parallel HypergraphConv branches (KEGG, STRING, CoExpr).
    Relation-aware attention aggregates them.
    Global mean pool reduces graph to single vector.
    """
    def __init__(self, n_genes, h_dim=64, dropout=0.3, n_relations=3):
        super().__init__()
        self.n_genes = n_genes
        self.n_relations = n_relations

        # Shared gene embedding
        self.gene_embed = nn.Sequential(
            nn.Linear(1, h_dim), nn.LayerNorm(h_dim), nn.GELU(),
        )

        # Per-relation HypergraphConv branches
        self.convs1 = nn.ModuleList([HypergraphConv(h_dim, h_dim) for _ in range(n_relations)])
        self.lns1   = nn.ModuleList([nn.LayerNorm(h_dim) for _ in range(n_relations)])
        self.convs2 = nn.ModuleList([HypergraphConv(h_dim, h_dim) for _ in range(n_relations)])
        self.lns2   = nn.ModuleList([nn.LayerNorm(h_dim) for _ in range(n_relations)])

        # Relation-aware attention: learns which relation to trust per gene
        self.relation_attn = nn.Sequential(
            nn.Linear(h_dim * n_relations, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, n_relations),
        )

        # Classifier (Sepsis) operates directly on Graph Embedding
        self.classifier = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, 2)
        )
        
        # Domain Discriminator (DANN) operates directly on Graph Embedding
        self.domain_discriminator = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, 4) # Allow up to 4 batches depending on metadata
        )
        self.dropout = dropout

    def forward(self, x, hedge_indices, batch, global_feat=None, alpha=1.0):
        """
        Args:
            x: (N_total, 1) node features
            hedge_indices: list of 3 hyperedge_index tensors [kegg, string, coexpr]
            batch: batch assignment
            global_feat: IGNORED in V12 pure graph model
        """
        n_nodes_per_graph = self.n_genes
        batch_size = batch.max().item() + 1

        # Shared embedding
        g = self.gene_embed(x)  # (N_total, h_dim)

        # Per-relation message passing
        rel_outputs = []
        for i in range(self.n_relations):
            hei = hedge_indices[i]
            if hei is not None and hei.size(1) > 0:
                h = self.convs1[i](g, hei)
                h = self.lns1[i](h); h = F.gelu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                r = g + h  # residual
                h = self.convs2[i](r, hei)
                h = self.lns2[i](h); h = F.gelu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                r = r + h  # residual
                rel_outputs.append(r)
            else:
                rel_outputs.append(g)

        # Relation-aware attention aggregation
        stacked = torch.stack(rel_outputs, dim=1)  # (N_total, n_rel, h_dim)
        concat = torch.cat(rel_outputs, dim=1)     # (N_total, n_rel * h_dim)
        attn_logits = self.relation_attn(concat)    # (N_total, n_rel)
        attn_weights = F.softmax(attn_logits, dim=1)  # (N_total, n_rel)

        # Weighted sum across relations (Node-level embeddings)
        h_multi = (stacked * attn_weights.unsqueeze(2)).sum(dim=1)  # (N_total, h_dim)

        # Graph-level pooling Instead of MLP Feature Selection
        graph_emb = global_mean_pool(h_multi, batch) # (Batch Size, h_dim)

        # Sepsis Classifier
        out = self.classifier(graph_emb)

        # Domain classification branch (reverses gradient)
        reversed_features = GradientReversalFunction.apply(graph_emb, alpha)
        domain_logits = self.domain_discriminator(reversed_features)

        return out, domain_logits, attn_weights

# ============================================================================
# DATA LOADING (Identical to V11)
# ============================================================================
def load_data():
    expr = pd.read_csv(os.path.join(OUT_DIR, "expression_combat_v2.csv"), index_col=0)
    meta = pd.read_csv(os.path.join(OUT_DIR, "metadata_v2.csv"))
    mad  = expr.apply(median_abs_deviation, axis=1)
    top_genes = mad.sort_values(ascending=False).head(TOP_K).index.tolist()
    expr_f = expr.loc[top_genes]
    return expr_f, meta, top_genes

def build_kegg_hyperedges(gene_list):
    gene_set = set(gene_list)
    g2i = {g: i for i, g in enumerate(gene_list)}
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
    ni, hi, hid = [], [], 0
    for genes in pw.values():
        for g in genes:
            if g in g2i:
                ni.append(g2i[g]); hi.append(hid)
        hid += 1
    if ni:
        return torch.tensor([ni, hi], dtype=torch.long), len(pw)
    return torch.zeros(2, 0, dtype=torch.long), 0

def build_string_hyperedges(gene_list):
    gene_set = set(gene_list)
    g2i = {g: i for i, g in enumerate(gene_list)}
    ppi_path = os.path.join(PROC_DIR, "ppi_network.csv")
    ni, hi, hid = [], [], 0
    if os.path.exists(ppi_path):
        ppi = pd.read_csv(ppi_path)
        pf = ppi[(ppi['source'].isin(gene_set)) &
                  (ppi['target'].isin(gene_set)) &
                  (ppi['score'] >= STRING_THR)]
        for _, row in pf.iterrows():
            s, t = row['source'], row['target']
            if s in g2i and t in g2i:
                ni.append(g2i[s]); hi.append(hid)
                ni.append(g2i[t]); hi.append(hid)
                hid += 1
    if ni:
        return torch.tensor([ni, hi], dtype=torch.long), hid
    return torch.zeros(2, 0, dtype=torch.long), 0

def build_coexpr_hyperedges(expr_f, gene_list, sample_ids):
    g2i = {g: i for i, g in enumerate(gene_list)}
    sub_expr = expr_f[sample_ids]
    vals = sub_expr.values
    from scipy.stats import rankdata
    ranked = np.apply_along_axis(rankdata, 1, vals) 
    ranked = (ranked - ranked.mean(axis=1, keepdims=True)) / (ranked.std(axis=1, keepdims=True) + 1e-8)
    corr = ranked @ ranked.T / ranked.shape[1]
    np.fill_diagonal(corr, 0)
    pairs = np.argwhere(np.abs(corr) > COEXPR_THR)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    ni, hi, hid = [], [], 0
    for (i, j) in pairs:
        ni.append(i); hi.append(hid)
        ni.append(j); hi.append(hid)
        hid += 1
    if ni:
        return torch.tensor([ni, hi], dtype=torch.long), hid
    return torch.zeros(2, 0, dtype=torch.long), 0

def make_data_list(expr_f, meta, gene_list, kegg_hei, string_hei):
    label_map = {'Control': 0, 'Sepsis': 1}
    unique_batches = sorted(meta['Batch'].unique())
    batch_map = {b: i for i, b in enumerate(unique_batches)}
    data_list = []
    for _, row in meta.iterrows():
        sid, cond = row['SampleID'], row['Condition']
        if cond not in label_map or sid not in expr_f.columns:
            continue
        x = torch.tensor(expr_f[sid].values, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(label_map[cond], dtype=torch.long)
        d = Data(x=x, y=y)
        d.kegg_hei = kegg_hei.clone()
        d.string_hei = string_hei.clone()
        d.num_nodes = len(gene_list)
        d.sample_id = sid
        d.batch_label = row['Batch']
        d.domain_y = torch.tensor(batch_map[row['Batch']], dtype=torch.long)
        data_list.append(d)
    return data_list

class MultiplexBatch:
    def __init__(self, x, y, domain_y, batch, hedge_indices):
        self.x = x
        self.y = y
        self.domain_y = domain_y
        self.batch = batch
        self.hedge_indices = hedge_indices

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.domain_y = self.domain_y.to(device)
        self.batch = self.batch.to(device)
        self.hedge_indices = [h.to(device) if h is not None else None for h in self.hedge_indices]
        return self

def collate_multiplex(data_list, coexpr_hei):
    xs, ys, domain_ys, batches = [], [], [], []
    kegg_ni, kegg_hi = [], []
    str_ni, str_hi = [], []
    coexpr_ni_all, coexpr_hi_all = [], []
    kegg_hid_offset, str_hid_offset, coexpr_hid_offset, node_offset = 0, 0, 0, 0
    n_genes = data_list[0].num_nodes
    for i, d in enumerate(data_list):
        xs.append(d.x)
        ys.append(d.y)
        domain_ys.append(d.domain_y)
        batches.append(torch.full((d.num_nodes,), i, dtype=torch.long))
        if d.kegg_hei.size(1) > 0:
            kegg_ni.append(d.kegg_hei[0] + node_offset)
            max_he = d.kegg_hei[1].max().item() + 1
            kegg_hi.append(d.kegg_hei[1] + kegg_hid_offset)
            kegg_hid_offset += max_he
        if d.string_hei.size(1) > 0:
            str_ni.append(d.string_hei[0] + node_offset)
            max_he = d.string_hei[1].max().item() + 1
            str_hi.append(d.string_hei[1] + str_hid_offset)
            str_hid_offset += max_he
        if coexpr_hei is not None and coexpr_hei.size(1) > 0:
            coexpr_ni_all.append(coexpr_hei[0] + node_offset)
            max_he = coexpr_hei[1].max().item() + 1
            coexpr_hi_all.append(coexpr_hei[1] + coexpr_hid_offset)
            coexpr_hid_offset += max_he
        node_offset += n_genes
    x = torch.cat(xs)
    y = torch.stack(ys)
    domain_y = torch.stack(domain_ys)
    batch = torch.cat(batches)
    def merge(ni_list, hi_list):
        if ni_list:
            return torch.stack([torch.cat(ni_list), torch.cat(hi_list)])
        return torch.zeros(2, 0, dtype=torch.long)
    kegg_h = merge(kegg_ni, kegg_hi)
    str_h  = merge(str_ni, str_hi)
    coexpr_h = merge(coexpr_ni_all, coexpr_hi_all)
    return MultiplexBatch(x, y, domain_y, batch, [kegg_h, str_h, coexpr_h])

def augment_multiplex(mbatch, hedge_drop=0.05, noise_std=0.02):
    if noise_std > 0:
        mbatch.x = mbatch.x + torch.randn_like(mbatch.x) * noise_std
    if hedge_drop > 0:
        new_hedges = []
        for hei in mbatch.hedge_indices:
            if hei is not None and hei.size(1) > 0:
                uh = hei[1].unique()
                keep = torch.rand(uh.max().item()+1) > hedge_drop
                mask = keep[hei[1]]
                new_hedges.append(hei[:, mask])
            else:
                new_hedges.append(hei)
        mbatch.hedge_indices = new_hedges
    return mbatch

def compute_metrics(probs, labels, threshold=0.5):
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
    best_acc, best_thr = 0, 0.5
    for thr in np.arange(0.1, 0.95, 0.01):
        preds = [1 if p >= thr else 0 for p in probs]
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr

# ============================================================================
# SINGLE FOLD TRAINING
# ============================================================================
def train_fold(fold_name, train_data, val_data, n_genes, coexpr_hei, device):
    n_train, n_val = len(train_data), len(val_data)
    ty = [d.y.item() for d in train_data]
    vy = [d.y.item() for d in val_data]
    n_coexpr = coexpr_hei.size(1) // 2 if coexpr_hei is not None else 0

    print(f"\n{'='*80}")
    print(f"  FOLD: {fold_name}")
    print(f"  Train: {n_train} samples (C={ty.count(0)}, S={ty.count(1)})")
    print(f"  Val:   {n_val} samples (C={vy.count(0)}, S={vy.count(1)})")
    print(f"  Co-expression edges (this fold): {n_coexpr}")
    print(f"{'='*80}")

    def make_loader(data, shuffle):
        batches = []
        idxs = list(range(len(data)))
        if shuffle:
            np.random.shuffle(idxs)
        for start in range(0, len(idxs), BS):
            batch_data = [data[i] for i in idxs[start:start+BS]]
            batches.append(collate_multiplex(batch_data, coexpr_hei))
        return batches

    model     = PureMultiplexHGCNDANN(n_genes, H_DIM, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    best_acc, best_epoch, patience_counter = 0.0, 0, 0
    best_state = None
    all_attn_weights = []

    print(f"  {'Ep':>4s} | {'TrLoss':>7s} {'TrAcc':>7s} {'TrAUC':>7s} | "
          f"{'VaLoss':>7s} {'VaAcc':>7s} {'VaAUC':>7s} {'VaF1':>6s} | "
          f"{'BstAcc':>7s} {'Pat':>3s} | {'αK':>5s} {'αS':>5s} {'αC':>5s}")
    print(f"  {'-'*4}-+-{'-'*7}-{'-'*7}-{'-'*7}-+-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*6}-+-{'-'*7}-{'-'*3}-+-{'-'*5}-{'-'*5}-{'-'*5}")

    for ep in range(1, EPOCHS + 1):
        p = float(ep) / 50.0
        alpha = 2. / (1. + np.exp(-10 * p)) - 1 if ep <= 50 else 1.0
        
        model.train()
        train_loss, train_domain_loss, train_n = 0.0, 0.0, 0
        train_probs, train_labels, train_dom_probs, train_dom_labels = [], [], [], []

        for mbatch in make_loader(train_data, shuffle=True):
            mbatch = augment_multiplex(mbatch).to(device)
            optimizer.zero_grad()
            out, domain_out, attn = model(mbatch.x, mbatch.hedge_indices, mbatch.batch, alpha=alpha)
            loss_sepsis = criterion(out, mbatch.y)
            loss_domain = domain_criterion(domain_out, mbatch.domain_y)
            loss = loss_sepsis + loss_domain
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss_sepsis.item() * mbatch.y.size(0)
            train_domain_loss += loss_domain.item() * mbatch.y.size(0)
            train_n += mbatch.y.size(0)
            with torch.no_grad():
                train_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
                train_labels.extend(mbatch.y.cpu().numpy())
                train_dom_probs.extend(torch.argmax(domain_out, dim=1).cpu().numpy())
                train_dom_labels.extend(mbatch.domain_y.cpu().numpy())

        scheduler.step()
        tr_loss = train_loss / max(train_n, 1)
        tr_m = compute_metrics(train_probs, train_labels)

        model.eval()
        val_probs, val_labels, val_dom_probs, val_dom_labels = [], [], [], []
        val_loss_total, val_dom_loss_total, val_n = 0.0, 0.0, 0
        epoch_attn = []
        with torch.no_grad():
            for mbatch in make_loader(val_data, shuffle=False):
                mbatch = mbatch.to(device)
                out, domain_out, attn = model(mbatch.x, mbatch.hedge_indices, mbatch.batch)
                loss_sepsis = criterion(out, mbatch.y)
                loss_domain = domain_criterion(domain_out, mbatch.domain_y)
                val_loss_total += loss_sepsis.item() * mbatch.y.size(0)
                val_dom_loss_total += loss_domain.item() * mbatch.y.size(0)
                val_n += mbatch.y.size(0)
                val_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
                val_labels.extend(mbatch.y.cpu().numpy())
                val_dom_probs.extend(torch.argmax(domain_out, dim=1).cpu().numpy())
                val_dom_labels.extend(mbatch.domain_y.cpu().numpy())
                epoch_attn.append(attn.mean(dim=0).cpu().numpy())

        va_loss = val_loss_total / max(val_n, 1)
        ep_thr = opt_threshold(val_probs, val_labels)
        va_m = compute_metrics(val_probs, val_labels, threshold=ep_thr)
        mean_attn = np.mean(epoch_attn, axis=0)

        improved = ""
        if va_m['acc'] > best_acc:
            best_acc = va_m['acc']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            patience_counter = 0
            improved = " ★"
            all_attn_weights.append(mean_attn)
        else:
            patience_counter += 1

        print(f"  {ep:4d} | {tr_loss:7.4f} {tr_m['acc']:7.4f} {tr_m['auroc']:7.4f} | "
              f"{va_loss:7.4f} {va_m['acc']:7.4f} {va_m['auroc']:7.4f} {va_m['f1']:6.3f} | "
              f"{best_acc:7.4f} {patience_counter:3d} | {mean_attn[0]:5.3f} {mean_attn[1]:5.3f} {mean_attn[2]:5.3f}{improved}")

        if patience_counter >= PATIENCE:
            print(f"  → Early stopping at epoch {ep} (best accuracy at epoch {best_epoch})")
            break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    final_probs, final_labels, final_attn_all = [], [], []
    with torch.no_grad():
        for mbatch in make_loader(val_data, shuffle=False):
            mbatch = mbatch.to(device)
            out, domain_out, attn = model(mbatch.x, mbatch.hedge_indices, mbatch.batch)
            final_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            final_labels.extend(mbatch.y.cpu().numpy())
            final_attn_all.append(attn.mean(dim=0).cpu().numpy())

    thr = opt_threshold(final_probs, final_labels)
    final_m = compute_metrics(final_probs, final_labels, threshold=thr)
    final_attn = np.mean(final_attn_all, axis=0)

    print(f"\n  ── Best Epoch {best_epoch} (Threshold = {thr:.4f}) ──")
    print(f"  Accuracy:  {final_m['acc']:.4f}  ← PRIMARY")
    print(f"  AUROC:     {final_m['auroc']:.4f}")
    return model, final_m, final_attn.tolist()

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("  V12 Pure HGCN: Multiplex HypergraphConv + DANN (No MLP)")
    print("=" * 80)

    expr_f, meta, gene_list = load_data()
    n_genes = len(gene_list)

    kegg_hei, _ = build_kegg_hyperedges(gene_list)
    string_hei, _ = build_string_hyperedges(gene_list)
    data_list = make_data_list(expr_f, meta, gene_list, kegg_hei, string_hei)

    combined_stratify = [f"{c}_{b}" for c, b in zip(meta['Condition'], meta['Batch'])]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    all_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(data_list)), combined_stratify)):
        train_data = [data_list[i] for i in train_idx]
        val_data   = [data_list[i] for i in val_idx]
        fold_name = f"Fold {fold_idx+1}/5"

        train_sids = [data_list[i].sample_id for i in train_idx]
        coexpr_hei, n_coexpr = build_coexpr_hyperedges(expr_f, gene_list, train_sids)

        model, metrics, attn = train_fold(fold_name, train_data, val_data, n_genes, coexpr_hei, DEVICE)

        if model is None:
            continue

        fold_path = os.path.join(MODEL_DIR, f"v12_pure_hgcn_fold{fold_idx+1}.pt")
        torch.save(model.state_dict(), fold_path)

        all_results.append({
            'fold': fold_idx + 1, 'acc': metrics['acc'], 'auroc': metrics['auroc']
        })

    valid_accs = [r['acc'] for r in all_results]
    valid_aurocs = [r['auroc'] for r in all_results]
    print(f"\nFINAL V12 Mean Accuracy: {np.mean(valid_accs):.4f} ± {np.std(valid_accs):.4f}")
    print(f"FINAL V12 Mean AUROC:    {np.mean(valid_aurocs):.4f} ± {np.std(valid_aurocs):.4f}")
    
    with open(os.path.join(OUT_DIR, "v12_pure_hgcn_results.json"), "w") as f:
        json.dump({'mean_acc': np.mean(valid_accs), 'mean_auroc': np.mean(valid_aurocs)}, f)

if __name__ == "__main__":
    main()
