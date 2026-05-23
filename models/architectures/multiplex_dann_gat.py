"""
A1 HGCN V10: Multiplex HypergraphConv + GNN-Guided Feature Selection
=====================================================================
Three relation layers (KEGG pathways, STRING PPI, co-expression)
on the same 2,000 gene nodes. Relation-aware attention aggregation
feeds GNN-guided feature selection (V8's architecture).

Key innovation: Co-expression edges computed PER FOLD on training
set only (Spearman |ρ| > 0.7) to avoid data leakage.

Fusion: GNN-Guided (V8-style) — multiplex GNN produces per-gene
attention weights that mask the MLP input.

V11 ADDITION:
Domain-Adversarial Neural Network (DANN) added to prevent the model
from learning batch-specific scanner noise from standard StratifiedKFold.
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
                             precision_score, recall_score, roc_curve)

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
COEXPR_THR = 0.7   # Spearman |ρ| threshold for co-expression edges
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
# MODEL: Multiplex GNN-Guided Feature Selection + DANN
# ============================================================================
class MultiplexGNNGuidedDANN(nn.Module):
    """
    3 parallel HypergraphConv branches (KEGG, STRING, CoExpr),
    each producing per-gene representations. Relation-aware attention
    aggregates them, then a scoring head produces per-gene importance
    weights that mask the MLP input.
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

        # Per-gene scoring head (sigmoid → [0,1] attention weights)
        self.gene_scorer = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.Tanh(),
            nn.Linear(h_dim // 2, 1),
        )

        # MLP processes attention-weighted expression
        self.mlp = nn.Sequential(
            nn.Linear(n_genes, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
        )

        # Classifier (Sepsis)
        self.classifier = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, 2)
        )
        
        # Domain Discriminator (DANN)
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
            global_feat: (BS, n_genes) expression vectors
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
                # No edges for this relation → use identity
                rel_outputs.append(g)

        # Relation-aware attention aggregation
        stacked = torch.stack(rel_outputs, dim=1)  # (N_total, n_rel, h_dim)
        concat = torch.cat(rel_outputs, dim=1)     # (N_total, n_rel * h_dim)
        attn_logits = self.relation_attn(concat)    # (N_total, n_rel)
        attn_weights = F.softmax(attn_logits, dim=1)  # (N_total, n_rel)

        # Weighted sum across relations
        h_multi = (stacked * attn_weights.unsqueeze(2)).sum(dim=1)  # (N_total, h_dim)

        # Per-gene importance scores
        gene_scores = torch.sigmoid(self.gene_scorer(h_multi))  # (N_total, 1)

        # Reshape and apply to expression
        if global_feat is not None:
            scores_per_graph = gene_scores.view(batch_size, n_nodes_per_graph)
            weighted_expr = global_feat * scores_per_graph
        else:
            weighted_expr = x.squeeze(1) * gene_scores.squeeze(1)
            weighted_expr = weighted_expr.view(batch_size, n_nodes_per_graph)

        # MLP on weighted expression
        mlp_out = self.mlp(weighted_expr)

        # Domain classification branch (reverses gradient)
        reversed_features = GradientReversalFunction.apply(mlp_out, alpha)
        domain_logits = self.domain_discriminator(reversed_features)

        return self.classifier(mlp_out), domain_logits, attn_weights


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
    """
    Compute co-expression edges from Spearman correlation.
    Only uses samples in sample_ids (training set) to avoid leakage.
    Returns hyperedge_index where each hyperedge connects a correlated pair.
    """
    g2i = {g: i for i, g in enumerate(gene_list)}
    sub_expr = expr_f[sample_ids]

    # Compute Spearman correlation matrix
    # For speed, compute on numpy directly
    vals = sub_expr.values  # (n_genes, n_samples)
    n = vals.shape[0]

    # Rank-based: compute pairwise Spearman for all genes
    # For 2000 genes this is manageable
    from scipy.stats import rankdata
    ranked = np.apply_along_axis(rankdata, 1, vals)  # rank per gene across samples
    # Standardize
    ranked = (ranked - ranked.mean(axis=1, keepdims=True)) / (ranked.std(axis=1, keepdims=True) + 1e-8)
    # Correlation = dot product of standardized ranks / n_samples
    corr = ranked @ ranked.T / ranked.shape[1]

    # Extract pairs above threshold
    np.fill_diagonal(corr, 0)
    pairs = np.argwhere(np.abs(corr) > COEXPR_THR)
    # Keep only upper triangle to avoid duplicates
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
    """Build data list without co-expression (added per fold)."""
    label_map = {'Control': 0, 'Sepsis': 1}
    
    # Dynamically map batches to integers for DANN
    unique_batches = sorted(meta['Batch'].unique())
    batch_map = {b: i for i, b in enumerate(unique_batches)}
    
    data_list = []
    for _, row in meta.iterrows():
        sid, cond = row['SampleID'], row['Condition']
        if cond not in label_map or sid not in expr_f.columns:
            continue
        x = torch.tensor(expr_f[sid].values, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(label_map[cond], dtype=torch.long)
        global_feat = torch.tensor(expr_f[sid].values, dtype=torch.float32).unsqueeze(0)
        d = Data(x=x, y=y)
        d.kegg_hei = kegg_hei.clone()
        d.string_hei = string_hei.clone()
        d.num_nodes = len(gene_list)
        d.global_feat = global_feat
        d.sample_id = sid
        d.batch_label = row['Batch']
        
        # Add Domain Label for DANN
        d.domain_y = torch.tensor(batch_map[row['Batch']], dtype=torch.long)

        data_list.append(d)
    return data_list


# ============================================================================
# CUSTOM COLLATE — batches 3 hyperedge sets
# ============================================================================
class MultiplexBatch:
    """Holds a batched set of graphs with 3 hyperedge types."""
    def __init__(self, x, y, domain_y, batch, global_feat, hedge_indices):
        self.x = x
        self.y = y
        self.domain_y = domain_y
        self.batch = batch
        self.global_feat = global_feat
        self.hedge_indices = hedge_indices  # list of 3 tensors

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.domain_y = self.domain_y.to(device)
        self.batch = self.batch.to(device)
        self.global_feat = self.global_feat.to(device)
        self.hedge_indices = [h.to(device) if h is not None else None for h in self.hedge_indices]
        return self


def collate_multiplex(data_list, coexpr_hei):
    """Custom collate that offsets all 3 hyperedge sets per graph."""
    xs, ys, domain_ys, batches, gf = [], [], [], [], []
    kegg_ni, kegg_hi = [], []
    str_ni, str_hi = [], []
    coexpr_ni_all, coexpr_hi_all = [], []

    kegg_hid_offset = 0
    str_hid_offset = 0
    coexpr_hid_offset = 0
    node_offset = 0
    n_genes = data_list[0].num_nodes

    for i, d in enumerate(data_list):
        xs.append(d.x)
        ys.append(d.y)
        domain_ys.append(d.domain_y)
        batches.append(torch.full((d.num_nodes,), i, dtype=torch.long))
        gf.append(d.global_feat)

        # KEGG hyperedges
        if d.kegg_hei.size(1) > 0:
            kegg_ni.append(d.kegg_hei[0] + node_offset)
            max_he = d.kegg_hei[1].max().item() + 1
            kegg_hi.append(d.kegg_hei[1] + kegg_hid_offset)
            kegg_hid_offset += max_he

        # STRING hyperedges
        if d.string_hei.size(1) > 0:
            str_ni.append(d.string_hei[0] + node_offset)
            max_he = d.string_hei[1].max().item() + 1
            str_hi.append(d.string_hei[1] + str_hid_offset)
            str_hid_offset += max_he

        # CoExpr hyperedges (shared across all graphs, so offset nodes only)
        if coexpr_hei is not None and coexpr_hei.size(1) > 0:
            coexpr_ni_all.append(coexpr_hei[0] + node_offset)
            max_he = coexpr_hei[1].max().item() + 1
            coexpr_hi_all.append(coexpr_hei[1] + coexpr_hid_offset)
            coexpr_hid_offset += max_he

        node_offset += n_genes

    # Stack
    x = torch.cat(xs)
    y = torch.stack(ys)
    domain_y = torch.stack(domain_ys)
    batch = torch.cat(batches)
    global_feat = torch.cat(gf)

    def merge(ni_list, hi_list):
        if ni_list:
            return torch.stack([torch.cat(ni_list), torch.cat(hi_list)])
        return torch.zeros(2, 0, dtype=torch.long)

    kegg_h = merge(kegg_ni, kegg_hi)
    str_h  = merge(str_ni, str_hi)
    coexpr_h = merge(coexpr_ni_all, coexpr_hi_all)

    return MultiplexBatch(x, y, domain_y, batch, global_feat, [kegg_h, str_h, coexpr_h])


# ============================================================================
# AUGMENTATION & EVALUATION
# ============================================================================
def augment_multiplex(mbatch, hedge_drop=0.05, noise_std=0.02):
    """Augment a MultiplexBatch."""
    if noise_std > 0:
        mbatch.x = mbatch.x + torch.randn_like(mbatch.x) * noise_std
        mbatch.global_feat = mbatch.global_feat + torch.randn_like(mbatch.global_feat) * noise_std

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
    """Find threshold that maximizes accuracy."""
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

    if len(set(vy)) < 2:
        print(f"  ⚠ Validation has only class {set(vy)} — skipping fold")
        return None, None, None

    # Create data loaders via manual batching
    def make_loader(data, shuffle):
        batches = []
        idxs = list(range(len(data)))
        if shuffle:
            np.random.shuffle(idxs)
        for start in range(0, len(idxs), BS):
            batch_data = [data[i] for i in idxs[start:start+BS]]
            batches.append(collate_multiplex(batch_data, coexpr_hei))
        return batches

    model     = MultiplexGNNGuidedDANN(n_genes, H_DIM, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    best_acc   = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0
    all_attn_weights = []  # Track relation attention over epochs

    print(f"  {'Ep':>4s} | {'TrLoss':>7s} {'TrAcc':>7s} {'TrAUC':>7s} | "
          f"{'VaLoss':>7s} {'VaAcc':>7s} {'VaAUC':>7s} {'VaF1':>6s} | "
          f"{'BstAcc':>7s} {'Pat':>3s} | {'αK':>5s} {'αS':>5s} {'αC':>5s}")
    print(f"  {'-'*4}-+-{'-'*7}-{'-'*7}-{'-'*7}-+-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*6}-+-{'-'*7}-{'-'*3}-+-{'-'*5}-{'-'*5}-{'-'*5}")

    for ep in range(1, EPOCHS + 1):
        # DANN Lambda Schedule (0 to 1 over first 50 epochs)
        p = float(ep) / 50.0
        alpha = 2. / (1. + np.exp(-10 * p)) - 1 if ep <= 50 else 1.0
        
        model.train()
        train_loss, train_domain_loss, train_n = 0.0, 0.0, 0
        train_probs, train_labels = [], []
        train_dom_probs, train_dom_labels = [], []

        train_loader = make_loader(train_data, shuffle=True)
        for mbatch in train_loader:
            mbatch = augment_multiplex(mbatch).to(device)
            optimizer.zero_grad()
            out, domain_out, attn = model(mbatch.x, mbatch.hedge_indices, mbatch.batch, mbatch.global_feat, alpha=alpha)
            
            # Sepsis Loss
            loss_sepsis = criterion(out, mbatch.y)
            # Domain Loss
            loss_domain = domain_criterion(domain_out, mbatch.domain_y)
            
            # Total Loss (GRL handles the negative gradient for discriminator)
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

        # Validate
        model.eval()
        val_probs, val_labels = [], []
        val_dom_probs, val_dom_labels = [], []
        val_loss_total, val_dom_loss_total, val_n = 0.0, 0.0, 0
        epoch_attn = []
        val_loader = make_loader(val_data, shuffle=False)
        with torch.no_grad():
            for mbatch in val_loader:
                mbatch = mbatch.to(device)
                out, domain_out, attn = model(mbatch.x, mbatch.hedge_indices, mbatch.batch, mbatch.global_feat)
                
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

        # Mean attention across all val nodes
        mean_attn = np.mean(epoch_attn, axis=0)

        improved = ""
        if va_m['acc'] > best_acc:
            best_acc   = va_m['acc']
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
        if ep % 5 == 0:
            tr_dom_acc = accuracy_score(train_dom_labels, train_dom_probs)
            va_dom_acc = accuracy_score(val_dom_labels, val_dom_probs)
            print(f"       → Domain Acc: Train={tr_dom_acc:.3f}, Val={va_dom_acc:.3f}, α={alpha:.3f}")

        if patience_counter >= PATIENCE:
            print(f"  → Early stopping at epoch {ep} (best accuracy at epoch {best_epoch})")
            break

    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    final_probs, final_labels = [], []
    final_attn_all = []
    val_loader = make_loader(val_data, shuffle=False)
    with torch.no_grad():
        for mbatch in val_loader:
            mbatch = mbatch.to(device)
            out, domain_out, attn = model(mbatch.x, mbatch.hedge_indices, mbatch.batch, mbatch.global_feat)
            final_probs.extend(F.softmax(out, dim=1)[:, 1].cpu().numpy())
            final_labels.extend(mbatch.y.cpu().numpy())
            final_attn_all.append(attn.mean(dim=0).cpu().numpy())

    thr = opt_threshold(final_probs, final_labels)
    final_m = compute_metrics(final_probs, final_labels, threshold=thr)
    final_attn = np.mean(final_attn_all, axis=0)

    print(f"\n  ── Best Epoch {best_epoch} (Threshold = {thr:.4f}) ──")
    print(f"  Accuracy:  {final_m['acc']:.4f}  ← PRIMARY")
    print(f"  AUROC:     {final_m['auroc']:.4f}")
    print(f"  F1 Score:  {final_m['f1']:.4f}")
    print(f"  Precision: {final_m['prec']:.4f}")
    print(f"  Recall:    {final_m['rec']:.4f}")
    print(f"  Relation Attention: KEGG={final_attn[0]:.3f}  STRING={final_attn[1]:.3f}  CoExpr={final_attn[2]:.3f}")

    return model, final_m, final_attn.tolist()


# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 80)
    print("  A1 HGCN V10: Multiplex HypergraphConv + GNN-Guided Feature Selection")
    print(f"  Device: {DEVICE}  |  Seed: {SEED}  |  Epochs: {EPOCHS}")
    print(f"  Relations: KEGG + STRING + Co-Expression (|ρ| > {COEXPR_THR})")
    print("  Early stopping: ACCURACY  |  Fusion: GNN-Guided (V8-style)")
    print("=" * 80)

    print("\n[1/4] Loading Data...")
    expr_f, meta, gene_list = load_data()
    n_genes = len(gene_list)
    print(f"  Genes: {n_genes}, Samples: {len(meta)}")

    print("\n[2/4] Building Static Hyperedges (KEGG + STRING)...")
    kegg_hei, n_kegg = build_kegg_hyperedges(gene_list)
    string_hei, n_string = build_string_hyperedges(gene_list)
    print(f"  KEGG: {n_kegg} pathways, {kegg_hei.size(1)} node-hyperedge pairs")
    print(f"  STRING: {n_string} PPI edges, {string_hei.size(1)} node-hyperedge pairs")

    print("\n[3/4] Preparing Patient Graphs...")
    data_list = make_data_list(expr_f, meta, gene_list, kegg_hei, string_hei)
    labels  = np.array([d.y.item() for d in data_list])
    batches = np.array([d.batch_label for d in data_list])
    print(f"  Total graphs: {len(data_list)}")

    print("\n[4/4] Running StratifiedKFold (n_splits=5)...")
    
    # Stratify by a combination of Condition and Batch to ensure even splits
    combined_stratify = [f"{c}_{b}" for c, b in zip(meta['Condition'], meta['Batch'])]
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    all_results = []
    best_overall_acc   = 0.0
    best_overall_state = None
    best_overall_fold  = ""

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(data_list)), combined_stratify)):
        train_data = [data_list[i] for i in train_idx]
        val_data   = [data_list[i] for i in val_idx]
        fold_name = f"Fold {fold_idx+1}/5"

        # Compute co-expression edges from TRAINING samples only
        train_sids = [data_list[i].sample_id for i in train_idx]
        print(f"\n  Computing co-expression edges from {len(train_sids)} training samples...")
        coexpr_hei, n_coexpr = build_coexpr_hyperedges(expr_f, gene_list, train_sids)
        print(f"  Co-expression edges: {n_coexpr} pairs (|ρ| > {COEXPR_THR})")

        model, metrics, attn = train_fold(fold_name, train_data, val_data, n_genes, coexpr_hei, DEVICE)

        if model is None:
            all_results.append({
                'fold': fold_idx + 1, 'acc': float('nan'), 'auroc': float('nan'), 'f1': float('nan'),
                'prec': float('nan'), 'rec': float('nan'),
                'n_val': len(val_data), 'n_coexpr': n_coexpr,
                'attn_kegg': float('nan'), 'attn_string': float('nan'),
                'attn_coexpr': float('nan'), 'skipped': True
            })
            continue

        fold_path = os.path.join(MODEL_DIR, f"v11_multiplex_dann_fold{fold_idx+1}.pt")
        torch.save(model.state_dict(), fold_path)
        print(f"  Model saved → {fold_path}")

        if metrics['acc'] > best_overall_acc:
            best_overall_acc   = metrics['acc']
            best_overall_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_overall_fold  = fold_name

        all_results.append({
            'fold': fold_idx + 1, 'acc': metrics['acc'], 'auroc': metrics['auroc'], 'f1': metrics['f1'],
            'prec': metrics['prec'], 'rec': metrics['rec'],
            'n_val': len(val_data), 'n_coexpr': n_coexpr,
            'attn_kegg': attn[0], 'attn_string': attn[1], 'attn_coexpr': attn[2],
            'skipped': False
        })

    if best_overall_state:
        best_path = os.path.join(MODEL_DIR, "v11_multiplex_dann_best.pt")
        torch.save(best_overall_state, best_path)
        print(f"\n  ★ Overall best model saved → {best_path}")

    # ══════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  FINAL RESULTS: V11 Multiplex DANN (KEGG+STRING+CoExpr) — Ranked by Accuracy")
    print(f"{'='*80}")
    print(f"  {'Fold':>4s} | {'N':>4s} | {'Acc':>7s} | {'AUC':>7s} | {'F1':>6s} | {'αK':>5s} {'αS':>5s} {'αC':>5s}")
    print(f"  {'-'*4}-+-{'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*5}-{'-'*5}-{'-'*5}")

    valid_accs, valid_aurocs, valid_f1s = [], [], []
    for r in all_results:
        if r['skipped']:
            print(f"  {r['fold']:4d} | {r['n_val']:4d} | {'SKIP':>7s} |")
        else:
            print(f"  {r['fold']:4d} | {r['n_val']:4d} | {r['acc']:7.4f} | {r['auroc']:7.4f} | {r['f1']:6.3f} | "
                  f"{r['attn_kegg']:5.3f} {r['attn_string']:5.3f} {r['attn_coexpr']:5.3f}")
            if not np.isnan(r['acc']):
                valid_accs.append(r['acc'])
                valid_aurocs.append(r['auroc'])
                valid_f1s.append(r['f1'])

    if valid_accs:
        print(f"  {'-'*4}-+-{'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}-+-{'-'*5}-{'-'*5}-{'-'*5}")
        print(f"  {'Mean':>4s} | {'':4s} | {np.mean(valid_accs):7.4f} | {np.mean(valid_aurocs):7.4f} | {np.mean(valid_f1s):6.3f} |")
        print(f"  {'Std':>4s}  | {'':4s} | {np.std(valid_accs):7.4f} | {np.std(valid_aurocs):7.4f} | {np.std(valid_f1s):6.3f} |")

    # 4-way comparison
    print(f"\n  ══ Comparison ══")
    comparisons = []
    for label, fname in [("V7 Late Fusion", "v7_sgkf_results.json"),
                          ("V8 GNN-Guided", "v8_guided_results.json"),
                          ("V9 Residual", "v9_residual_results.json")]:
        path = os.path.join(OUT_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            acc = data.get('mean_acc')
            auc = data.get('mean_auroc')
            gse = next((f['acc'] for f in data.get('folds', []) if 'GSE69686' in f.get('held_out', '')), None)
            comparisons.append((label, acc, auc, gse))

    gse_v10 = next((r['acc'] for r in all_results if 'GSE69686' in r.get('held_out', '')), None)
    comparisons.append(("V11 Multiplex DANN", np.mean(valid_accs), np.mean(valid_aurocs), "N/A"))

    print(f"  {'Model':<20s} | {'Mean Acc':>8s} | {'Mean AUC':>8s} | {'GSE69686 Acc':>12s}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}")
    for label, acc, auc, gse in comparisons:
        gse_str = f"{gse:.4f}" if isinstance(gse, float) else "N/A"
        acc_str = f"{acc:.4f}" if acc else "N/A"
        auc_str = f"{auc:.4f}" if auc else "N/A"
        print(f"  {label:<20s} | {acc_str:>8s} | {auc_str:>8s} | {gse_str:>12s}")

    summary = {
        'method': 'Multiplex HypergraphConv + GNN-Guided DANN (V11)',
        'architecture': '3 relations (KEGG+STRING+CoExpr) → attention → gene scoring → MLP + DANN',
        'early_stopping': 'accuracy',
        'coexpr_threshold': COEXPR_THR,
        'n_folds': 5,
        'seed': SEED,
        'folds': all_results,
        'mean_acc': float(np.mean(valid_accs)) if valid_accs else None,
        'std_acc': float(np.std(valid_accs)) if valid_accs else None,
        'mean_auroc': float(np.mean(valid_aurocs)) if valid_aurocs else None,
        'std_auroc': float(np.std(valid_aurocs)) if valid_aurocs else None,
        'best_fold': best_overall_fold,
        'best_acc': best_overall_acc,
        'total_time_min': (time.time() - t0) / 60,
    }

    json_path = os.path.join(OUT_DIR, "v11_multiplex_dann_results.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    elapsed = (time.time() - t0) / 60
    print(f"\n  Total execution time: {elapsed:.1f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()
