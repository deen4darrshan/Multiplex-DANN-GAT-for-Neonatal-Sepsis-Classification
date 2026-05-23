import os
import json
import math
import gzip
import time
import copy
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import HypergraphConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = r"C:\Users\terry\Downloads\Projects\ISEF"
ACSEF_DIR = os.path.join(ROOT, "ACSEF_Final_Submission")
ACSEF_DATA = os.path.join(ACSEF_DIR, "data")
ACSEF_MODELS = os.path.join(ACSEF_DIR, "models")
ACSEF_RESULTS = os.path.join(ACSEF_DIR, "results")
ACSEF_FIGURES = os.path.join(ACSEF_DIR, "figures")
ACSEF_LOGS = os.path.join(ACSEF_DIR, "logs")

CH_RESULTS = os.path.join(ROOT, "CH_DANN_Plan", "results")
RAW_DIR = os.path.join(ROOT, "data", "raw")

EXPR_PATH = os.path.join(CH_RESULTS, "expression_combat_v2.csv")
META_PATH = os.path.join(CH_RESULTS, "metadata_v2.csv")
STRING_GZ = os.path.join(RAW_DIR, "9606.protein.links.v12.0.txt.gz")

for p in [ACSEF_DATA, ACSEF_MODELS, ACSEF_RESULTS, ACSEF_FIGURES, ACSEF_LOGS]:
    os.makedirs(p, exist_ok=True)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SEED = 42
TOP_K = 2000
STRING_THRESHOLD = 700
COEXPR_THRESHOLD = 0.7
H_DIM = 64
DROPOUT = 0.3
LR = 3e-4
WD = 5e-4
EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 12
IG_STEPS = 24
MAX_IG_SAMPLES_PER_CLASS = 100

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(os.path.join(ACSEF_LOGS, f"xai_pipeline_{datetime.now().strftime('%Y-%m-%d')}.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def np_mad(matrix_2d):
    med = np.median(matrix_2d, axis=1, keepdims=True)
    abs_dev = np.abs(matrix_2d - med)
    return np.median(abs_dev, axis=1)


def rank_transform_row(row):
    # Pure NumPy rank approximation (no scipy.rankdata).
    order = np.argsort(row, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(1, len(row) + 1, dtype=np.float32)
    return ranks


def compute_coexpr_hyperedges(expr_top):
    vals = expr_top.values.astype(np.float32)  # genes x samples
    ranks = np.apply_along_axis(rank_transform_row, 1, vals)
    ranks = (ranks - ranks.mean(axis=1, keepdims=True)) / (ranks.std(axis=1, keepdims=True) + 1e-8)
    corr = (ranks @ ranks.T) / ranks.shape[1]
    np.fill_diagonal(corr, 0.0)

    pairs = np.argwhere(np.triu(np.abs(corr) > COEXPR_THRESHOLD, k=1))
    n_pairs = pairs.shape[0]
    if n_pairs == 0:
        return torch.zeros((2, 0), dtype=torch.long), 0

    node_idx = np.empty(n_pairs * 2, dtype=np.int64)
    hedge_idx = np.empty(n_pairs * 2, dtype=np.int64)
    for h, (i, j) in enumerate(pairs):
        node_idx[2 * h] = i
        node_idx[2 * h + 1] = j
        hedge_idx[2 * h] = h
        hedge_idx[2 * h + 1] = h

    return torch.tensor(np.vstack([node_idx, hedge_idx]), dtype=torch.long), int(n_pairs)


def build_kegg_hyperedges(gene_list):
    gset = set(gene_list)
    g2i = {g: i for i, g in enumerate(gene_list)}
    pathways = {}
    try:
        import gseapy as gp
        kegg = gp.get_library("KEGG_2021_Human")
        for pname, genes in kegg.items():
            overlap = list(set(genes) & gset)
            if len(overlap) >= 3:
                pathways[pname] = overlap
    except Exception:
        # Compact fallback focused on immune biology.
        fallback = {
            "Toll-like receptor signaling": ["TLR1", "TLR2", "TLR4", "MYD88", "IRAK1", "TRAF6", "NFKB1", "RELA"],
            "Cytokine signaling": ["IL1B", "IL6", "TNF", "CXCL8", "STAT1", "STAT3", "JAK1", "JAK2"],
            "Neutrophil activation": ["MPO", "MMP9", "S100A8", "S100A9", "FCGR1A", "CEACAM8", "ELANE"],
            "Complement cascade": ["C1QA", "C1QB", "C3", "CFB", "CFD", "CFH"],
        }
        for pname, genes in fallback.items():
            overlap = list(set(genes) & gset)
            if len(overlap) >= 3:
                pathways[pname] = overlap

    node_i = []
    hedge_i = []
    hid = 0
    for genes in pathways.values():
        for g in genes:
            node_i.append(g2i[g])
            hedge_i.append(hid)
        hid += 1

    if not node_i:
        return torch.zeros((2, 0), dtype=torch.long), 0
    hei = torch.tensor([node_i, hedge_i], dtype=torch.long)
    return hei, len(pathways)


def _collect_ensembl_proteins(record):
    out = []
    ens = record.get("ensembl", None)
    if ens is None:
        return out
    if isinstance(ens, dict):
        ens = [ens]
    for item in ens:
        if not isinstance(item, dict):
            continue
        p = item.get("protein")
        if p:
            out.append(str(p))
    return out


def build_string_hyperedges_chunked(gene_list):
    log("Building STRING relation with chunked scan + top-gene protein mapping")
    import mygene

    mg = mygene.MyGeneInfo()
    q = mg.querymany(
        gene_list,
        scopes="symbol",
        fields="ensembl.protein",
        species="human",
        as_dataframe=False,
        returnall=False,
        verbose=False,
    )

    protein_to_gene = {}
    for rec in q:
        if "notfound" in rec and rec["notfound"]:
            continue
        g = str(rec.get("query", "")).upper().strip()
        for p in _collect_ensembl_proteins(rec):
            protein_to_gene[p] = g

    edge_set = set()
    n_lines = 0
    n_kept = 0
    with gzip.open(STRING_GZ, "rt") as f:
        _ = f.readline()
        for line in f:
            n_lines += 1
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            p1 = parts[0].replace("9606.", "")
            p2 = parts[1].replace("9606.", "")
            try:
                score = int(parts[2])
            except Exception:
                continue
            if score < STRING_THRESHOLD:
                continue
            g1 = protein_to_gene.get(p1)
            g2 = protein_to_gene.get(p2)
            if not g1 or not g2 or g1 == g2:
                continue
            if g1 not in protein_to_gene.values() or g2 not in protein_to_gene.values():
                continue
            edge_set.add(tuple(sorted((g1, g2))))
            n_kept += 1

    g2i = {g: i for i, g in enumerate(gene_list)}
    node_i = []
    hedge_i = []
    hid = 0
    for g1, g2 in edge_set:
        if g1 in g2i and g2 in g2i:
            node_i.extend([g2i[g1], g2i[g2]])
            hedge_i.extend([hid, hid])
            hid += 1

    if not node_i:
        return torch.zeros((2, 0), dtype=torch.long), 0, {"string_lines_scanned": n_lines, "string_pairs_kept_raw": n_kept, "hyperedges": 0}

    hei = torch.tensor([node_i, hedge_i], dtype=torch.long)
    meta = {"string_lines_scanned": n_lines, "string_pairs_kept_raw": n_kept, "hyperedges": hid}
    return hei, hid, meta


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class MultiplexHyperDANNMLP(nn.Module):
    def __init__(self, n_genes, h_dim=64, dropout=0.3, n_relations=3, n_domains=4):
        super().__init__()
        self.n_genes = n_genes
        self.n_relations = n_relations
        self.dropout = dropout

        self.gene_embed = nn.Sequential(
            nn.Linear(1, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
        )

        self.convs1 = nn.ModuleList([HypergraphConv(h_dim, h_dim) for _ in range(n_relations)])
        self.lns1 = nn.ModuleList([nn.LayerNorm(h_dim) for _ in range(n_relations)])
        self.convs2 = nn.ModuleList([HypergraphConv(h_dim, h_dim) for _ in range(n_relations)])
        self.lns2 = nn.ModuleList([nn.LayerNorm(h_dim) for _ in range(n_relations)])

        self.relation_attn = nn.Sequential(
            nn.Linear(h_dim * n_relations, h_dim),
            nn.Tanh(),
            nn.Linear(h_dim, n_relations),
        )

        self.gene_scorer = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.Tanh(),
            nn.Linear(h_dim // 2, 1),
        )

        self.mlp = nn.Sequential(
            nn.Linear(n_genes, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, 2),
        )

        self.domain_discriminator = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, n_domains),
        )

    @staticmethod
    def _grl(x, alpha):
        return x + 0.0 * alpha

    def forward(self, x, hedge_indices, batch, global_feat, grl_alpha=1.0, return_gene_scores=False):
        bs = int(batch.max().item()) + 1
        g = self.gene_embed(x)

        rel_outputs = []
        for i in range(self.n_relations):
            hei = hedge_indices[i]
            if hei is not None and hei.numel() > 0 and hei.shape[1] > 0:
                h = self.convs1[i](g, hei)
                h = F.gelu(self.lns1[i](h))
                h = F.dropout(h, p=self.dropout, training=self.training)
                r = g + h
                h2 = self.convs2[i](r, hei)
                h2 = F.gelu(self.lns2[i](h2))
                h2 = F.dropout(h2, p=self.dropout, training=self.training)
                rel_outputs.append(r + h2)
            else:
                rel_outputs.append(g)

        stacked = torch.stack(rel_outputs, dim=1)
        concat = torch.cat(rel_outputs, dim=1)
        attn_logits = self.relation_attn(concat)
        attn_weights = F.softmax(attn_logits, dim=1)
        h_multi = (stacked * attn_weights.unsqueeze(2)).sum(dim=1)
        gene_scores = torch.sigmoid(self.gene_scorer(h_multi))

        scores_per_graph = gene_scores.view(bs, self.n_genes)
        weighted_expr = global_feat * scores_per_graph
        z = self.mlp(weighted_expr)

        cls_logits = self.classifier(z)
        dom_logits = self.domain_discriminator(self._grl(z, grl_alpha))

        if return_gene_scores:
            return cls_logits, dom_logits, attn_weights, gene_scores
        return cls_logits, dom_logits, attn_weights


# -----------------------------------------------------------------------------
# Data batching
# -----------------------------------------------------------------------------
class BatchPack:
    def __init__(self, x, y, domain_y, batch, global_feat, hedges):
        self.x = x
        self.y = y
        self.domain_y = domain_y
        self.batch = batch
        self.global_feat = global_feat
        self.hedges = hedges

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.domain_y = self.domain_y.to(device)
        self.batch = self.batch.to(device)
        self.global_feat = self.global_feat.to(device)
        self.hedges = [h.to(device) if h is not None else None for h in self.hedges]
        return self


def collate_multiplex(items, n_genes):
    xs, ys, ds, bs, gf = [], [], [], [], []
    k_ni, k_hi = [], []
    s_ni, s_hi = [], []
    c_ni, c_hi = [], []
    k_off = 0
    s_off = 0
    c_off = 0
    n_off = 0

    for i, d in enumerate(items):
        xs.append(d.x)
        ys.append(d.y)
        ds.append(d.domain_y)
        bs.append(torch.full((n_genes,), i, dtype=torch.long))
        gf.append(d.global_feat)

        def append_relation(hei, ni_list, hi_list, h_off):
            if hei is None or hei.numel() == 0 or hei.shape[1] == 0:
                return h_off
            ni_list.append(hei[0] + n_off)
            max_h = int(hei[1].max().item()) + 1
            hi_list.append(hei[1] + h_off)
            return h_off + max_h

        k_off = append_relation(d.kegg_hei, k_ni, k_hi, k_off)
        s_off = append_relation(d.string_hei, s_ni, s_hi, s_off)
        c_off = append_relation(d.coexpr_hei, c_ni, c_hi, c_off)
        n_off += n_genes

    def merge(ni, hi):
        if not ni:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.stack([torch.cat(ni), torch.cat(hi)])

    batch = BatchPack(
        x=torch.cat(xs),
        y=torch.stack(ys),
        domain_y=torch.stack(ds),
        batch=torch.cat(bs),
        global_feat=torch.cat(gf),
        hedges=[merge(k_ni, k_hi), merge(s_ni, s_hi), merge(c_ni, c_hi)],
    )
    return batch


def make_batches(data, n_genes, batch_size=12, shuffle=True):
    idx = np.arange(len(data))
    if shuffle:
        np.random.shuffle(idx)
    out = []
    for i in range(0, len(idx), batch_size):
        sel = [data[j] for j in idx[i:i + batch_size]]
        out.append(collate_multiplex(sel, n_genes))
    return out


# -----------------------------------------------------------------------------
# Train + eval
# -----------------------------------------------------------------------------
def eval_model(model, batches):
    model.eval()
    all_prob = []
    all_y = []
    with torch.no_grad():
        for b in batches:
            b = b.to(DEVICE)
            out, _, _ = model(b.x, b.hedges, b.batch, b.global_feat, grl_alpha=0.0)
            prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_prob.extend(prob.tolist())
            all_y.extend(b.y.cpu().numpy().tolist())
    y = np.array(all_y, dtype=np.int64)
    p = np.array(all_prob, dtype=np.float32)
    pred = (p >= 0.5).astype(np.int64)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5
    acc = float(accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, zero_division=0))
    return {"auc": auc, "acc": acc, "f1": f1, "y": y, "p": p}


def train_model(train_data, val_data, n_genes, n_domains):
    model = MultiplexHyperDANNMLP(n_genes=n_genes, h_dim=H_DIM, dropout=DROPOUT, n_domains=n_domains).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    cls_loss = nn.CrossEntropyLoss()
    dom_loss = nn.CrossEntropyLoss()

    best = None
    best_auc = -1.0
    wait = 0
    hist = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        p = min(1.0, epoch / 25.0)
        grl_alpha = float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

        batches = make_batches(train_data, n_genes, BATCH_SIZE, shuffle=True)
        for b in batches:
            b = b.to(DEVICE)
            opt.zero_grad()
            out, dom_out, _ = model(b.x, b.hedges, b.batch, b.global_feat, grl_alpha=grl_alpha)
            l1 = cls_loss(out, b.y)
            l2 = dom_loss(dom_out, b.domain_y)
            loss = l1 + l2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        train_metrics = eval_model(model, make_batches(train_data, n_genes, BATCH_SIZE, shuffle=False))
        val_metrics = eval_model(model, make_batches(val_data, n_genes, BATCH_SIZE, shuffle=False))
        hist.append({"epoch": epoch, "train_auc": train_metrics["auc"], "val_auc": val_metrics["auc"], "train_acc": train_metrics["acc"], "val_acc": val_metrics["acc"]})

        log(f"Epoch {epoch:03d} | train_auc={train_metrics['auc']:.4f} val_auc={val_metrics['auc']:.4f} train_acc={train_metrics['acc']:.4f} val_acc={val_metrics['acc']:.4f}")
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                log(f"Early stopping at epoch {epoch} (best val_auc={best_auc:.4f})")
                break

    if best is not None:
        model.load_state_dict(best)
    return model, hist


# -----------------------------------------------------------------------------
# XAI
# -----------------------------------------------------------------------------
def single_forward(model, d):
    model.eval()
    x = d.x.to(DEVICE)
    gf = d.global_feat.to(DEVICE)
    b = torch.zeros(x.shape[0], dtype=torch.long, device=DEVICE)
    hedges = [d.kegg_hei.to(DEVICE), d.string_hei.to(DEVICE), d.coexpr_hei.to(DEVICE)]
    out, dom, attn, gene_scores = model(x, hedges, b, gf, grl_alpha=0.0, return_gene_scores=True)
    return out, attn, gene_scores.view(-1)


def integrated_gradients(model, d, target_class, steps=24):
    model.eval()
    x0 = torch.zeros_like(d.x, device=DEVICE)
    gf0 = torch.zeros_like(d.global_feat, device=DEVICE)
    x = d.x.to(DEVICE)
    gf = d.global_feat.to(DEVICE)
    b = torch.zeros(x.shape[0], dtype=torch.long, device=DEVICE)
    hedges = [d.kegg_hei.to(DEVICE), d.string_hei.to(DEVICE), d.coexpr_hei.to(DEVICE)]

    total_grad = torch.zeros_like(gf)
    for i in range(1, steps + 1):
        alpha = float(i) / float(steps)
        xs = (x0 + alpha * (x - x0)).detach().requires_grad_(True)
        gfs = (gf0 + alpha * (gf - gf0)).detach().requires_grad_(True)
        out, _, _, _ = model(xs, hedges, b, gfs, grl_alpha=0.0, return_gene_scores=True)
        target = out[0, target_class]
        grad = torch.autograd.grad(target, gfs, retain_graph=False, create_graph=False)[0]
        total_grad += grad

    avg_grad = total_grad / float(steps)
    ig = (gf - gf0) * avg_grad
    return ig.detach().cpu().numpy().reshape(-1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    t0 = time.time()
    log("Robust XAI pipeline started")
    log(f"Device: {DEVICE}")

    if not os.path.exists(EXPR_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("Missing expression_combat_v2.csv or metadata_v2.csv")

    expr = pd.read_csv(EXPR_PATH, index_col=0)
    meta = pd.read_csv(META_PATH)
    meta = meta[meta["SampleID"].isin(expr.columns)].copy()
    expr = expr[meta["SampleID"].tolist()]
    meta["Label"] = (meta["Condition"].astype(str).str.lower() == "sepsis").astype(int)

    # Pure NumPy MAD for top-gene selection.
    mad = np_mad(expr.values.astype(np.float32))
    top_idx = np.argsort(-mad)[:TOP_K]
    gene_list = expr.index[top_idx].tolist()
    expr_top = expr.loc[gene_list]
    expr_top = expr_top.astype(np.float32)

    # Save a data snapshot for reproducibility.
    expr_top.to_csv(os.path.join(ACSEF_DATA, "expression_top2000.csv"))
    meta.to_csv(os.path.join(ACSEF_DATA, "metadata_aligned.csv"), index=False)
    with open(os.path.join(ACSEF_DATA, "top2000_gene_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(gene_list))

    log(f"Loaded aligned matrix: genes={expr.shape[0]}, samples={expr.shape[1]}")
    log(f"Selected top genes by NumPy MAD: {len(gene_list)}")

    kegg_hei, n_kegg = build_kegg_hyperedges(gene_list)
    coexpr_hei, n_coexpr = compute_coexpr_hyperedges(expr_top)
    string_hei, n_string, string_meta = build_string_hyperedges_chunked(gene_list)
    log(f"Hyperedges: KEGG={n_kegg}, STRING={n_string}, CoExpr={n_coexpr}")
    log(f"STRING scan stats: {string_meta}")

    # Build per-sample graphs.
    batch_names = sorted(meta["Batch"].astype(str).unique().tolist())
    batch_map = {b: i for i, b in enumerate(batch_names)}
    n_domains = max(2, len(batch_map))
    n_genes = len(gene_list)

    data_list = []
    for _, row in meta.iterrows():
        sid = row["SampleID"]
        y = int(row["Label"])
        dom = batch_map[str(row["Batch"])]
        x = torch.tensor(expr_top[sid].values.reshape(-1, 1), dtype=torch.float32)
        g = torch.tensor(expr_top[sid].values.reshape(1, -1), dtype=torch.float32)
        d = Data(x=x, y=torch.tensor(y, dtype=torch.long))
        d.domain_y = torch.tensor(dom, dtype=torch.long)
        d.global_feat = g
        d.kegg_hei = kegg_hei
        d.string_hei = string_hei
        d.coexpr_hei = coexpr_hei
        d.sample_id = sid
        d.condition = str(row["Condition"])
        data_list.append(d)

    y_all = np.array([int(d.y.item()) for d in data_list], dtype=np.int64)
    idx = np.arange(len(data_list))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y_all)
    train_data = [data_list[i] for i in tr_idx]
    val_data = [data_list[i] for i in va_idx]
    log(f"Train/Val split: train={len(train_data)}, val={len(val_data)}")

    model, history = train_model(train_data, val_data, n_genes, n_domains)
    val_metrics = eval_model(model, make_batches(val_data, n_genes, BATCH_SIZE, shuffle=False))
    log(f"Final validation metrics: AUC={val_metrics['auc']:.4f}, ACC={val_metrics['acc']:.4f}, F1={val_metrics['f1']:.4f}")

    torch.save(model.state_dict(), os.path.join(ACSEF_MODELS, "multiplex_hyper_dann_mlp_acsef.pt"))
    with open(os.path.join(ACSEF_RESULTS, "xai_training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": "Multiplex-Hypergraph-DANN-MLP",
                "device": str(DEVICE),
                "train_size": len(train_data),
                "val_size": len(val_data),
                "kegg_hyperedges": int(n_kegg),
                "string_hyperedges": int(n_string),
                "coexpr_hyperedges": int(n_coexpr),
                "string_scan": string_meta,
                "validation": {"auc": val_metrics["auc"], "accuracy": val_metrics["acc"], "f1": val_metrics["f1"]},
                "history": history,
            },
            f,
            indent=2,
        )

    # Compute IG + gene scores.
    sepsis_samples = [d for d in data_list if int(d.y.item()) == 1]
    control_samples = [d for d in data_list if int(d.y.item()) == 0]
    if len(sepsis_samples) > MAX_IG_SAMPLES_PER_CLASS:
        rng = np.random.default_rng(SEED)
        sepsis_samples = [sepsis_samples[i] for i in rng.choice(len(sepsis_samples), MAX_IG_SAMPLES_PER_CLASS, replace=False)]
    if len(control_samples) > MAX_IG_SAMPLES_PER_CLASS:
        rng = np.random.default_rng(SEED + 1)
        control_samples = [control_samples[i] for i in rng.choice(len(control_samples), MAX_IG_SAMPLES_PER_CLASS, replace=False)]

    log(f"IG sample counts: sepsis={len(sepsis_samples)}, control={len(control_samples)}")

    ig_sepsis = []
    ig_control = []
    gs_sepsis = []
    gs_control = []
    attn_rows = []

    for d in sepsis_samples:
        out, attn, gs = single_forward(model, d)
        ig = integrated_gradients(model, d, target_class=1, steps=IG_STEPS)
        ig_sepsis.append(ig)
        gs_sepsis.append(gs.detach().cpu().numpy())
        attn_rows.append(attn.detach().cpu().numpy().mean(axis=0))

    for d in control_samples:
        out, attn, gs = single_forward(model, d)
        ig = integrated_gradients(model, d, target_class=0, steps=IG_STEPS)
        ig_control.append(ig)
        gs_control.append(gs.detach().cpu().numpy())
        attn_rows.append(attn.detach().cpu().numpy().mean(axis=0))

    ig_sepsis = np.array(ig_sepsis)
    ig_control = np.array(ig_control)
    gs_sepsis = np.array(gs_sepsis)
    gs_control = np.array(gs_control)
    attn_rows = np.array(attn_rows)

    mean_ig_sepsis = ig_sepsis.mean(axis=0)
    mean_ig_control = ig_control.mean(axis=0)
    mean_gs = np.vstack([gs_sepsis, gs_control]).mean(axis=0)
    score = (mean_ig_sepsis - mean_ig_control) * mean_gs

    res = pd.DataFrame(
        {
            "gene": gene_list,
            "attribution_score": score,
            "abs_attribution_score": np.abs(score),
            "mean_ig_sepsis": mean_ig_sepsis,
            "mean_ig_control": mean_ig_control,
            "mean_gene_score": mean_gs,
            "direction": np.where(score >= 0, "Sepsis_up", "Control_up"),
        }
    ).sort_values("abs_attribution_score", ascending=False)
    res.insert(0, "rank", np.arange(1, len(res) + 1))
    top100 = res.head(100).copy()
    top100.to_csv(os.path.join(ACSEF_RESULTS, "top_100_biomarkers.csv"), index=False)
    res.to_csv(os.path.join(ACSEF_RESULTS, "all_gene_attributions.csv"), index=False)
    log("Saved biomarker tables: top_100_biomarkers.csv and all_gene_attributions.csv")

    # Save relation attention summary.
    attn_df = pd.DataFrame(attn_rows, columns=["KEGG", "STRING", "CoExpr"])
    attn_df.to_csv(os.path.join(ACSEF_RESULTS, "relation_attention_distribution.csv"), index=False)

    # -----------------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Diverging bar (required).
    top20 = res.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top20["attribution_score"]]
    ax.barh(top20["gene"], top20["attribution_score"], color=colors)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_title("Top 20 Biomarkers by Integrated Attribution (Diverging)")
    ax.set_xlabel("Attribution Score (IG × GeneScore)")
    plt.tight_layout()
    plt.savefig(os.path.join(ACSEF_FIGURES, "fig_biomarker_attributions.png"), dpi=220)
    plt.close()

    # PCA of top biomarkers.
    top_biomarker_genes = res.head(20)["gene"].tolist()
    X = expr_top.loc[top_biomarker_genes].T.values
    y = meta.set_index("SampleID").loc[expr_top.columns, "Condition"].values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=SEED)
    pcs = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 5))
    for c, color in [("Control", "#1f77b4"), ("Sepsis", "#d62728")]:
        mask = (y == c)
        ax.scatter(pcs[mask, 0], pcs[mask, 1], s=24, alpha=0.8, c=color, label=c)
    ax.set_title("PCA Projection Using Top 20 Biomarkers")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ACSEF_FIGURES, "fig_top_biomarker_pca.png"), dpi=220)
    plt.close()

    # Normalization distributions.
    raw_vals = expr_top.loc[top_biomarker_genes].values.flatten()
    z_vals = (raw_vals - raw_vals.mean()) / (raw_vals.std() + 1e-8)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(raw_vals, bins=80, alpha=0.5, density=True, label="Raw expression")
    ax.hist(z_vals, bins=80, alpha=0.5, density=True, label="Z-score normalized")
    ax.set_title("Distribution Curves: Raw vs Normalized Top Biomarker Expression")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ACSEF_FIGURES, "fig_normalization_distributions.png"), dpi=220)
    plt.close()

    # Correlation heatmap.
    corr = pd.DataFrame(expr_top.loc[top_biomarker_genes].T).corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(top_biomarker_genes)))
    ax.set_yticks(np.arange(len(top_biomarker_genes)))
    ax.set_xticklabels(top_biomarker_genes, rotation=90, fontsize=7)
    ax.set_yticklabels(top_biomarker_genes, fontsize=7)
    ax.set_title("Top 20 Biomarker Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(ACSEF_FIGURES, "fig_biomarker_correlation_heatmap.png"), dpi=220)
    plt.close()

    # Attention distribution heatmap.
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn_df.values, aspect="auto", cmap="viridis")
    ax.set_title("Relation Attention Distribution Across XAI Samples")
    ax.set_xlabel("Relation")
    ax.set_ylabel("Sample")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["KEGG", "STRING", "CoExpr"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(ACSEF_FIGURES, "fig_relation_attention_heatmap.png"), dpi=220)
    plt.close()

    total_min = (time.time() - t0) / 60.0
    log(f"Robust XAI pipeline complete in {total_min:.2f} min")


if __name__ == "__main__":
    main()

