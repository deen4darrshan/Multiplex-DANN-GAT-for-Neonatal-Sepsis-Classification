#!/usr/bin/env python3
"""
General_Sepsis_V11 - Step 03
Train V11-style MultiplexGNNGuidedDANN with StratifiedGroupKFold.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.nn import HypergraphConv


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    today = datetime.now().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Train General_Sepsis_V11 model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(root / "results"))
    parser.add_argument("--model-dir", default=str(root / "models"))
    parser.add_argument("--log-file", default=str(root / "logs" / f"{today}_03_train_v11_general_sepsis.log"))
    parser.add_argument("--expression-path", default=str(root / "results" / "expression_combat.csv"))
    parser.add_argument("--metadata-path", default=str(root / "results" / "metadata.csv"))
    parser.add_argument("--pathway-info-path", default=str(root / "results" / "pathway_info.json"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lambda-dann", type=float, default=0.1)
    parser.add_argument("--coexpr-threshold", type=float, default=0.7)
    parser.add_argument("--max-coexpr-edges", type=int, default=60000)
    parser.add_argument("--max-string-edges-train", type=int, default=15000)
    parser.add_argument(
        "--cv-mode",
        type=str,
        default="lodo",
        choices=["lodo", "sgkf"],
        help="Validation protocol: leave-one-dataset-out (lodo) or StratifiedGroupKFold by patient_id (sgkf).",
    )
    parser.add_argument(
        "--feature-select-top-k",
        type=int,
        default=800,
        help="Fold-internal MAD feature selection size.",
    )
    parser.add_argument("--smoke", action="store_true", help="Fast smoke run (reduced epochs/splits).")
    return parser.parse_args()


def init_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("general_sepsis_v11_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class MultiplexGNNGuidedDANN(nn.Module):
    def __init__(
        self,
        n_genes: int,
        node_feat_dim: int,
        n_classes: int,
        n_domains: int,
        h_dim: int = 128,
        dropout: float = 0.5,
        n_relations: int = 3,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_domains = n_domains
        self.n_relations = n_relations

        self.gene_embed = nn.Sequential(
            nn.Linear(node_feat_dim, h_dim),
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

        # Mandatory MLP branch per locked decision.
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
            nn.Linear(h_dim, n_classes),
        )

        if n_domains > 1:
            self.domain_discriminator = nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h_dim, n_domains),
            )
        else:
            self.domain_discriminator = None

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        hedge_indices: Sequence[torch.Tensor],
        batch: torch.Tensor,
        global_feat: torch.Tensor,
        alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        bs = int(batch.max().item()) + 1
        g = self.gene_embed(x)

        rel_outputs = []
        for i in range(self.n_relations):
            hei = hedge_indices[i]
            if hei is not None and hei.numel() > 0 and hei.size(1) > 0:
                h = self.convs1[i](g, hei)
                h = self.lns1[i](h)
                h = F.gelu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                r = g + h
                h = self.convs2[i](r, hei)
                h = self.lns2[i](h)
                h = F.gelu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                rel_outputs.append(r + h)
            else:
                rel_outputs.append(g)

        stacked = torch.stack(rel_outputs, dim=1)  # [Nnodes, n_rel, h]
        concat = torch.cat(rel_outputs, dim=1)  # [Nnodes, n_rel*h]
        attn_logits = self.relation_attn(concat)
        attn_weights = F.softmax(attn_logits, dim=1)
        h_multi = (stacked * attn_weights.unsqueeze(2)).sum(dim=1)

        gene_scores = torch.sigmoid(self.gene_scorer(h_multi))  # [Nnodes, 1]
        scores_per_graph = gene_scores.view(bs, self.n_genes)
        weighted_expr = global_feat * scores_per_graph
        mlp_out = self.mlp(weighted_expr)
        class_logits = self.classifier(mlp_out)

        if self.domain_discriminator is not None:
            rev = GradientReversalFunction.apply(mlp_out, alpha)
            domain_logits = self.domain_discriminator(rev)
        else:
            domain_logits = None
        return class_logits, domain_logits, attn_weights


@dataclass
class SampleItem:
    sample_id: str
    y: int
    domain_y: int
    expr: torch.Tensor  # [n_genes]


@dataclass
class MultiplexBatch:
    x: torch.Tensor
    y: torch.Tensor
    domain_y: torch.Tensor
    batch: torch.Tensor
    global_feat: torch.Tensor
    hedge_indices: List[torch.Tensor]
    sample_ids: List[str]

    def to(self, device: torch.device) -> "MultiplexBatch":
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.domain_y = self.domain_y.to(device)
        self.batch = self.batch.to(device)
        self.global_feat = self.global_feat.to(device)
        self.hedge_indices = [h.to(device) for h in self.hedge_indices]
        return self


def empty_hyperedge() -> torch.Tensor:
    return torch.zeros((2, 0), dtype=torch.long)


def pathways_to_hyperedge(pathways: Sequence[Dict[str, object]]) -> torch.Tensor:
    node_idx = []
    hedge_idx = []
    for h_id, p in enumerate(pathways):
        gidx = p.get("gene_indices", [])
        for g in gidx:
            node_idx.append(int(g))
            hedge_idx.append(h_id)
    if not node_idx:
        return empty_hyperedge()
    return torch.tensor([node_idx, hedge_idx], dtype=torch.long)


def string_edges_to_hyperedge(edges: Sequence[Sequence[object]]) -> torch.Tensor:
    node_idx = []
    hedge_idx = []
    for h_id, e in enumerate(edges):
        i, j = int(e[0]), int(e[1])
        if i == j:
            continue
        node_idx.extend([i, j])
        hedge_idx.extend([h_id, h_id])
    if not node_idx:
        return empty_hyperedge()
    return torch.tensor([node_idx, hedge_idx], dtype=torch.long)


def tile_hyperedge(base_hei: torch.Tensor, batch_size: int, n_genes: int) -> torch.Tensor:
    if base_hei is None or base_hei.numel() == 0 or base_hei.size(1) == 0:
        return empty_hyperedge()
    n_h = int(base_hei[1].max().item()) + 1
    nodes = []
    hedges = []
    for b in range(batch_size):
        nodes.append(base_hei[0] + b * n_genes)
        hedges.append(base_hei[1] + b * n_h)
    return torch.stack([torch.cat(nodes), torch.cat(hedges)], dim=0)


def collate_multiplex(
    items: Sequence[SampleItem],
    n_genes: int,
    kegg_hei: torch.Tensor,
    string_hei: torch.Tensor,
    coexpr_hei: torch.Tensor,
) -> MultiplexBatch:
    xs = []
    ys = []
    domains = []
    batch_vec = []
    global_feat = []
    sample_ids = []
    for b, it in enumerate(items):
        xs.append(it.expr.view(-1, 1))
        ys.append(it.y)
        domains.append(it.domain_y)
        batch_vec.append(torch.full((n_genes,), b, dtype=torch.long))
        global_feat.append(it.expr.view(1, -1))
        sample_ids.append(it.sample_id)

    bs = len(items)
    return MultiplexBatch(
        x=torch.cat(xs, dim=0),
        y=torch.tensor(ys, dtype=torch.long),
        domain_y=torch.tensor(domains, dtype=torch.long),
        batch=torch.cat(batch_vec, dim=0),
        global_feat=torch.cat(global_feat, dim=0),
        hedge_indices=[
            tile_hyperedge(kegg_hei, bs, n_genes),
            tile_hyperedge(string_hei, bs, n_genes),
            tile_hyperedge(coexpr_hei, bs, n_genes),
        ],
        sample_ids=sample_ids,
    )


def iter_batches(indices: np.ndarray, batch_size: int, shuffle: bool) -> List[np.ndarray]:
    idx = np.array(indices, dtype=np.int64)
    if shuffle:
        np.random.shuffle(idx)
    chunks = []
    for s in range(0, len(idx), batch_size):
        chunks.append(idx[s : s + batch_size])
    return chunks


def compute_metrics(y_true: np.ndarray, y_prob_sepsis: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob_sepsis >= 0.5).astype(np.int64)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    auc = float("nan")
    if len(np.unique(y_true)) >= 2:
        try:
            auc = float(roc_auc_score(y_true, y_prob_sepsis))
        except Exception:
            auc = float("nan")
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "auroc": auc}


def class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y.astype(np.int64), minlength=2).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    w = counts.sum() / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def build_coexpr_hyperedge(
    expr_genes_x_samples: pd.DataFrame,
    train_sample_ids: Sequence[str],
    threshold: float,
    max_edges: int,
) -> Tuple[torch.Tensor, int]:
    train_expr = expr_genes_x_samples.loc[:, list(train_sample_ids)]
    ranked = train_expr.rank(axis=1, method="average")
    arr = ranked.values.astype(np.float32)
    corr = np.corrcoef(arr)
    tri_i, tri_j = np.triu_indices(corr.shape[0], k=1)
    tri_v = np.abs(corr[tri_i, tri_j])
    keep = tri_v >= threshold
    i = tri_i[keep]
    j = tri_j[keep]
    v = tri_v[keep]
    if max_edges > 0 and i.shape[0] > max_edges:
        top = np.argpartition(-v, max_edges - 1)[:max_edges]
        i = i[top]
        j = j[top]
    n = int(i.shape[0])
    if n == 0:
        return empty_hyperedge(), 0
    hid = np.arange(n, dtype=np.int64)
    node_idx = np.concatenate([i.astype(np.int64), j.astype(np.int64)])
    hedge_idx = np.concatenate([hid, hid])
    hei = torch.tensor(np.stack([node_idx, hedge_idx], axis=0), dtype=torch.long)
    return hei, n


def maybe_number(x):
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def compute_mad(df: pd.DataFrame) -> pd.Series:
    med = df.median(axis=1)
    return (df.sub(med, axis=0).abs()).median(axis=1)


def select_fold_genes(expr_genes_x_samples: pd.DataFrame, train_sample_ids: Sequence[str], top_k: int) -> List[int]:
    if top_k <= 0:
        return list(range(expr_genes_x_samples.shape[0]))
    train_block = expr_genes_x_samples.loc[:, list(train_sample_ids)]
    mad = compute_mad(train_block)
    k = min(top_k, mad.shape[0])
    return mad.sort_values(ascending=False).head(k).index.to_series().map(lambda g: expr_genes_x_samples.index.get_loc(g)).tolist()


def filter_kegg_pathways(
    pathways: Sequence[Dict[str, object]],
    keep_indices: Sequence[int],
) -> List[Dict[str, object]]:
    keep_set = set(int(i) for i in keep_indices)
    remap = {int(old): new for new, old in enumerate(keep_indices)}
    out: List[Dict[str, object]] = []
    for p in pathways:
        gids = p.get("gene_indices", [])
        new_g = [remap[int(g)] for g in gids if int(g) in keep_set]
        if len(new_g) < 2:
            continue
        out.append({"name": p.get("name", "pathway"), "gene_indices": new_g})
    return out


def filter_string_edges(
    edges: Sequence[Sequence[object]],
    keep_indices: Sequence[int],
    max_edges: int,
) -> List[List[int]]:
    keep_set = set(int(i) for i in keep_indices)
    remap = {int(old): new for new, old in enumerate(keep_indices)}
    filt: List[List[int]] = []
    for e in edges:
        i, j = int(e[0]), int(e[1])
        if i == j:
            continue
        if i in keep_set and j in keep_set:
            filt.append([remap[i], remap[j]])
    if max_edges > 0 and len(filt) > max_edges:
        filt = filt[:max_edges]
    return filt


def build_splits(
    args: argparse.Namespace,
    train_meta: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, object]]]:
    n = len(train_meta)
    idx = np.arange(n, dtype=np.int64)
    out: List[Tuple[np.ndarray, np.ndarray, Dict[str, object]]] = []
    if args.cv_mode == "lodo":
        datasets = train_meta["dataset"].astype(str).values
        unique_datasets = sorted(pd.Series(datasets).unique().tolist())
        for d in unique_datasets:
            va_idx = idx[datasets == d]
            tr_idx = idx[datasets != d]
            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue
            out.append((tr_idx, va_idx, {"val_dataset": d}))
    else:
        n_splits = min(args.n_splits, int(np.bincount(y).min()), len(np.unique(groups)))
        if n_splits < 2:
            raise RuntimeError(f"Cannot run SGKF with current data. n_splits={n_splits}")
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        for tr_idx, va_idx in sgkf.split(idx, y, groups):
            out.append((np.array(tr_idx, dtype=np.int64), np.array(va_idx, dtype=np.int64), {}))
    if len(out) < 2:
        raise RuntimeError(f"Insufficient folds generated for cv_mode={args.cv_mode}.")
    return out


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 4)
        args.patience = min(args.patience, 2)
        args.n_splits = min(args.n_splits, 3)
        args.feature_select_top_k = min(args.feature_select_top_k, 300)

    set_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    logger = init_logger(Path(args.log_file).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=== General_Sepsis_V11 Step 03: train_v11_general_sepsis ===")
    logger.info(
        "device=%s seed=%d smoke=%s cv_mode=%s feature_select_top_k=%d",
        device,
        args.seed,
        args.smoke,
        args.cv_mode,
        args.feature_select_top_k,
    )

    expr = pd.read_csv(args.expression_path, index_col=0)
    meta = pd.read_csv(args.metadata_path)
    if "sample_id" not in meta.columns and "index" in meta.columns:
        meta = meta.rename(columns={"index": "sample_id"})
    with open(args.pathway_info_path, "r", encoding="utf-8") as f:
        pathway = json.load(f)

    genes = pathway["genes"]
    expr = expr.reindex(genes)
    expr = expr.loc[:, meta["sample_id"].tolist()]
    n_genes_total = len(genes)
    logger.info("Loaded expression=%s metadata=%d genes=%d", expr.shape, len(meta), n_genes_total)

    train_meta = meta.loc[meta["split_role"] == "train"].copy()
    holdout_meta = meta.loc[meta["split_role"] == "holdout"].copy()
    train_ids = train_meta["sample_id"].tolist()
    holdout_ids = holdout_meta["sample_id"].tolist()

    y_map = {"control": 0, "sepsis": 1}
    y = train_meta["condition"].str.lower().map(y_map).values.astype(np.int64)
    groups = train_meta["patient_id"].astype(str).values
    if np.any(pd.isna(y)):
        bad = train_meta.loc[pd.isna(train_meta["condition"].str.lower().map(y_map)), "condition"].unique().tolist()
        raise RuntimeError(f"Unknown condition labels in train metadata: {bad}")

    # Domain labels from batch.
    domain_values = sorted(train_meta["batch"].astype(str).unique().tolist())
    domain_map = {b: i for i, b in enumerate(domain_values)}
    train_domains = train_meta["batch"].astype(str).map(domain_map).values.astype(np.int64)
    n_domains = len(domain_values)
    logger.info("Train samples=%d holdout=%d classes=%s domains=%s", len(train_ids), len(holdout_ids), dict(pd.Series(y).value_counts()), domain_map)

    expr_train = expr.loc[:, train_ids]
    base_kegg = pathway.get("kegg", {}).get("pathways", [])
    base_string_edges = pathway.get("string", {}).get("edges", [])
    split_plan = build_splits(args, train_meta, y, groups)
    logger.info("Prepared %d folds with mode=%s", len(split_plan), args.cv_mode)

    fold_results = []
    oof_records = []
    relation_attention_traces = {}
    best_overall_auc = -1.0
    best_overall_fold = -1
    best_overall_state = None
    best_overall_fold_meta: Dict[str, object] = {}

    for fold_idx, (tr_idx, va_idx, fold_meta) in enumerate(split_plan, start=1):
        fold_name = f"fold{fold_idx}"
        tr_sample_ids = [train_ids[i] for i in tr_idx]
        va_sample_ids = [train_ids[i] for i in va_idx]
        val_dataset = fold_meta.get("val_dataset")
        logger.info(
            "[%s] train=%d val=%d%s",
            fold_name,
            len(tr_idx),
            len(va_idx),
            "" if val_dataset is None else f" val_dataset={val_dataset}",
        )

        selected_gene_indices = select_fold_genes(
            expr_genes_x_samples=expr_train,
            train_sample_ids=tr_sample_ids,
            top_k=args.feature_select_top_k,
        )
        selected_genes = [genes[int(i)] for i in selected_gene_indices]
        n_genes = len(selected_genes)
        expr_train_sel = expr_train.iloc[selected_gene_indices, :]
        fold_mean = expr_train_sel.loc[:, tr_sample_ids].mean(axis=1)
        fold_std = expr_train_sel.loc[:, tr_sample_ids].std(axis=1).replace(0, 1.0).fillna(1.0)
        expr_train_sel = expr_train_sel.sub(fold_mean, axis=0).div(fold_std, axis=0)
        logger.info("[%s] selected_genes=%d", fold_name, n_genes)

        fold_kegg = filter_kegg_pathways(base_kegg, selected_gene_indices)
        fold_string_edges = filter_string_edges(base_string_edges, selected_gene_indices, args.max_string_edges_train)
        kegg_hei = pathways_to_hyperedge(fold_kegg)
        string_hei = string_edges_to_hyperedge(fold_string_edges)
        logger.info(
            "[%s] static_edges kegg=%d string=%d",
            fold_name,
            int(kegg_hei.size(1)),
            int(string_hei.size(1)),
        )

        coexpr_hei, n_coexpr = build_coexpr_hyperedge(
            expr_genes_x_samples=expr_train_sel,
            train_sample_ids=tr_sample_ids,
            threshold=args.coexpr_threshold,
            max_edges=args.max_coexpr_edges,
        )
        logger.info("[%s] coexpr_edges=%d", fold_name, n_coexpr)

        sample_items: List[SampleItem] = []
        for i, sid in enumerate(train_ids):
            item = SampleItem(
                sample_id=sid,
                y=int(y[i]),
                domain_y=int(train_domains[i]),
                expr=torch.tensor(expr_train_sel[sid].values.astype(np.float32), dtype=torch.float32),
            )
            sample_items.append(item)

        model = MultiplexGNNGuidedDANN(
            n_genes=n_genes,
            node_feat_dim=1,
            n_classes=2,
            n_domains=n_domains,
            h_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)

        y_tr = y[tr_idx]
        cls_loss = nn.CrossEntropyLoss(weight=class_weights(y_tr, device))
        dom_loss = nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state = None
        best_metric = -1.0
        patience = 0
        epoch_logs = []

        for ep in range(1, args.epochs + 1):
            model.train()
            p = float(ep) / max(1.0, min(50.0, float(args.epochs)))
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

            tr_probs = []
            tr_true = []
            tr_loss_sum = 0.0
            tr_n = 0

            for chunk in iter_batches(tr_idx, args.batch_size, shuffle=True):
                items = [sample_items[int(i)] for i in chunk]
                mb = collate_multiplex(items, n_genes, kegg_hei, string_hei, coexpr_hei).to(device)
                opt.zero_grad()
                logits, dom_logits, _ = model(mb.x, mb.hedge_indices, mb.batch, mb.global_feat, alpha=alpha)
                loss_cls = cls_loss(logits, mb.y)
                if dom_logits is not None and n_domains > 1:
                    loss_dom = dom_loss(dom_logits, mb.domain_y)
                    loss = loss_cls + args.lambda_dann * loss_dom
                else:
                    loss = loss_cls
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                y_true = mb.y.detach().cpu().numpy()
                tr_probs.append(probs)
                tr_true.append(y_true)
                tr_loss_sum += float(loss.item()) * len(y_true)
                tr_n += len(y_true)

            tr_probs_np = np.concatenate(tr_probs)
            tr_true_np = np.concatenate(tr_true)
            tr_metrics = compute_metrics(tr_true_np, tr_probs_np)

            model.eval()
            va_probs = []
            va_true = []
            va_loss_sum = 0.0
            va_n = 0
            attn_epoch = []
            with torch.no_grad():
                for chunk in iter_batches(va_idx, args.batch_size, shuffle=False):
                    items = [sample_items[int(i)] for i in chunk]
                    mb = collate_multiplex(items, n_genes, kegg_hei, string_hei, coexpr_hei).to(device)
                    logits, dom_logits, attn = model(mb.x, mb.hedge_indices, mb.batch, mb.global_feat, alpha=0.0)
                    loss_cls = cls_loss(logits, mb.y)
                    # Validation loss uses task loss only; domain targets may be unseen in LODO folds.
                    loss = loss_cls
                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    y_true = mb.y.cpu().numpy()
                    va_probs.append(probs)
                    va_true.append(y_true)
                    va_loss_sum += float(loss.item()) * len(y_true)
                    va_n += len(y_true)
                    # Mean relation attention over nodes.
                    attn_epoch.append(attn.mean(dim=0).cpu().numpy().tolist())

            va_probs_np = np.concatenate(va_probs)
            va_true_np = np.concatenate(va_true)
            va_metrics = compute_metrics(va_true_np, va_probs_np)
            attn_mean = np.mean(np.array(attn_epoch), axis=0).tolist() if attn_epoch else [0.0, 0.0, 0.0]

            score = va_metrics["auroc"] if not math.isnan(va_metrics["auroc"]) else va_metrics["accuracy"]
            if score > best_metric:
                best_metric = score
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1

            epoch_log = {
                "epoch": ep,
                "alpha_grl": float(alpha),
                "train_loss": float(tr_loss_sum / max(tr_n, 1)),
                "val_loss": float(va_loss_sum / max(va_n, 1)),
                "train_metrics": tr_metrics,
                "val_metrics": va_metrics,
                "relation_attention_mean": {
                    "kegg": maybe_number(attn_mean[0]),
                    "string": maybe_number(attn_mean[1]),
                    "coexpr": maybe_number(attn_mean[2]),
                },
            }
            epoch_logs.append(epoch_log)
            logger.info(
                "[%s] ep=%03d train_loss=%.4f val_loss=%.4f val_auc=%s val_acc=%.4f attn=(%.3f,%.3f,%.3f)",
                fold_name,
                ep,
                epoch_log["train_loss"],
                epoch_log["val_loss"],
                "nan" if math.isnan(va_metrics["auroc"]) else f"{va_metrics['auroc']:.4f}",
                va_metrics["accuracy"],
                attn_mean[0],
                attn_mean[1],
                attn_mean[2],
            )
            if patience >= args.patience:
                logger.info("[%s] early stop at epoch %d", fold_name, ep)
                break

        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)

        # Final val predictions with best checkpoint.
        model.eval()
        fold_probs = []
        fold_true = []
        fold_sid = []
        with torch.no_grad():
            for chunk in iter_batches(va_idx, args.batch_size, shuffle=False):
                items = [sample_items[int(i)] for i in chunk]
                mb = collate_multiplex(items, n_genes, kegg_hei, string_hei, coexpr_hei).to(device)
                logits, _, attn = model(mb.x, mb.hedge_indices, mb.batch, mb.global_feat, alpha=0.0)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                y_true = mb.y.cpu().numpy()
                fold_probs.extend(probs.tolist())
                fold_true.extend(y_true.tolist())
                fold_sid.extend(mb.sample_ids)

        fold_probs_np = np.array(fold_probs, dtype=float)
        fold_true_np = np.array(fold_true, dtype=np.int64)
        fold_metrics = compute_metrics(fold_true_np, fold_probs_np)

        ckpt_path = model_dir / f"general_sepsis_v11_fold{fold_idx}.pt"
        torch.save(
            {
                "state_dict": best_state,
                "model_config": {
                    "n_genes": n_genes,
                    "node_feat_dim": 1,
                    "n_classes": 2,
                    "n_domains": n_domains,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                },
                "fold": fold_idx,
                "domain_map": domain_map,
                "coexpr_threshold": args.coexpr_threshold,
                "max_coexpr_edges": args.max_coexpr_edges,
                "max_string_edges_train": args.max_string_edges_train,
                "coexpr_n_edges": int(n_coexpr),
                "coexpr_train_sample_ids": tr_sample_ids,
                "selected_gene_indices": [int(i) for i in selected_gene_indices],
                "selected_genes": selected_genes,
                "fold_norm_mean": fold_mean.astype(float).tolist(),
                "fold_norm_std": fold_std.astype(float).tolist(),
                "cv_mode": args.cv_mode,
            },
            ckpt_path,
        )
        logger.info("[%s] saved checkpoint %s", fold_name, ckpt_path)

        fold_record = {
            "fold": fold_idx,
            "n_train": len(tr_idx),
            "n_val": len(va_idx),
            "train_sample_ids": tr_sample_ids,
            "val_sample_ids": va_sample_ids,
            "coexpr_train_sample_ids": tr_sample_ids,
            "coexpr_n_edges": int(n_coexpr),
            "selected_gene_indices": [int(i) for i in selected_gene_indices],
            "selected_genes": selected_genes,
            "fold_norm_mean": fold_mean.astype(float).tolist(),
            "fold_norm_std": fold_std.astype(float).tolist(),
            "val_dataset": val_dataset,
            "metrics": fold_metrics,
            "checkpoint_path": str(ckpt_path),
            "epoch_metrics": epoch_logs,
        }
        fold_results.append(fold_record)
        relation_attention_traces[fold_name] = [
            e["relation_attention_mean"] for e in epoch_logs if "relation_attention_mean" in e
        ]

        for sid, yt, yp in zip(fold_sid, fold_true, fold_probs):
            oof_records.append(
                {
                    "sample_id": sid,
                    "y_true": int(yt),
                    "y_prob_sepsis": float(yp),
                    "fold": fold_idx,
                }
            )

        fold_auc = fold_metrics["auroc"]
        score = fold_auc if not math.isnan(fold_auc) else fold_metrics["accuracy"]
        if score > best_overall_auc:
            best_overall_auc = score
            best_overall_fold = fold_idx
            best_overall_state = copy.deepcopy(best_state)
            best_overall_fold_meta = {
                "selected_gene_indices": [int(i) for i in selected_gene_indices],
                "selected_genes": selected_genes,
                "fold_norm_mean": fold_mean.astype(float).tolist(),
                "fold_norm_std": fold_std.astype(float).tolist(),
                "val_dataset": val_dataset,
            }

    if best_overall_state is None:
        raise RuntimeError("Training produced no checkpoint state.")

    best_path = model_dir / "general_sepsis_v11_best.pt"
    torch.save(
        {
            "state_dict": best_overall_state,
            "model_config": {
                "n_genes": n_genes,
                "node_feat_dim": 1,
                "n_classes": 2,
                "n_domains": n_domains,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            },
            "best_fold": int(best_overall_fold),
            "domain_map": domain_map,
            "coexpr_threshold": args.coexpr_threshold,
            "max_coexpr_edges": args.max_coexpr_edges,
            "max_string_edges_train": args.max_string_edges_train,
            "train_sample_ids": train_ids,
            "holdout_sample_ids": holdout_ids,
            "selected_gene_indices": best_overall_fold_meta.get("selected_gene_indices", []),
            "selected_genes": best_overall_fold_meta.get("selected_genes", []),
            "fold_norm_mean": best_overall_fold_meta.get("fold_norm_mean", []),
            "fold_norm_std": best_overall_fold_meta.get("fold_norm_std", []),
            "cv_mode": args.cv_mode,
        },
        best_path,
    )
    logger.info("Saved best model: %s (fold=%d)", best_path, best_overall_fold)

    oof_df = pd.DataFrame(oof_records).sort_values("sample_id").reset_index(drop=True)
    pooled_metrics = compute_metrics(
        oof_df["y_true"].values.astype(np.int64),
        oof_df["y_prob_sepsis"].values.astype(float),
    )
    cv_summary = {
        "mean_accuracy": float(np.mean([f["metrics"]["accuracy"] for f in fold_results])),
        "mean_f1": float(np.mean([f["metrics"]["f1"] for f in fold_results])),
        "mean_precision": float(np.mean([f["metrics"]["precision"] for f in fold_results])),
        "mean_recall": float(np.mean([f["metrics"]["recall"] for f in fold_results])),
        "mean_auroc": float(np.nanmean([f["metrics"]["auroc"] for f in fold_results])),
        "pooled_oof": pooled_metrics,
    }

    out_payload = {
        "generated_at": datetime.now().isoformat(),
        "seed": args.seed,
        "device": str(device),
        "cv_mode": args.cv_mode,
        "config": {
            "n_splits": len(split_plan),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lambda_dann": args.lambda_dann,
            "coexpr_threshold": args.coexpr_threshold,
            "max_coexpr_edges": args.max_coexpr_edges,
            "max_string_edges_train": args.max_string_edges_train,
            "feature_select_top_k": args.feature_select_top_k,
            "smoke": args.smoke,
        },
        "domain_map": domain_map,
        "train_sample_count": len(train_ids),
        "holdout_sample_count": len(holdout_ids),
        "folds": fold_results,
        "cv_summary": cv_summary,
        "oof_predictions": oof_records,
        "relation_attention_traces": relation_attention_traces,
        "best_model_path": str(best_path),
    }

    out_path = output_dir / "cv_metrics_raw.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    logger.info("Saved CV metrics: %s", out_path)


if __name__ == "__main__":
    main()
