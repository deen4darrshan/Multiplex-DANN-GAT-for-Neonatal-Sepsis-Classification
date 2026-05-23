"""
Transfer experiment: apply the V11 multiplex HGCN+MLP+DANN idea to Alzheimer data.

This script adapts the V11 architecture lineage from CH_DANN_Plan to ADNI-style
PyG datasets saved as lists of torch_geometric.data.Data objects.

Input defaults:
  ALZHEIMERS_STRATEGIC_PATHWAY/data/adni/processed/dataset_real_ad_2000.pt

Output:
  CH_DANN_Plan/models/v11_alz_transfer_*.pt
  CH_DANN_Plan/results/v11_alz_transfer_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.nn import HypergraphConv

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(PROJECT_ROOT)

OUT_DIR = os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "results")
MODEL_DIR = os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


class MultiplexGNNGuidedDANN(nn.Module):
    def __init__(
        self,
        n_genes: int,
        node_feat_dim: int,
        n_classes: int,
        n_domains: int,
        h_dim: int = 64,
        dropout: float = 0.3,
        n_relations: int = 3,
    ) -> None:
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
        batch_size = int(batch.max().item()) + 1
        g = self.gene_embed(x)

        rel_outputs: List[torch.Tensor] = []
        for i in range(self.n_relations):
            hei = hedge_indices[i]
            if hei is not None and hei.size(1) > 0:
                h = self.convs1[i](g, hei)
                h = self.lns1[i](h)
                h = F.gelu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                r = g + h
                h = self.convs2[i](r, hei)
                h = self.lns2[i](h)
                h = F.gelu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                r = r + h
                rel_outputs.append(r)
            else:
                rel_outputs.append(g)

        stacked = torch.stack(rel_outputs, dim=1)
        concat = torch.cat(rel_outputs, dim=1)
        attn_logits = self.relation_attn(concat)
        attn_weights = F.softmax(attn_logits, dim=1)
        h_multi = (stacked * attn_weights.unsqueeze(2)).sum(dim=1)

        gene_scores = torch.sigmoid(self.gene_scorer(h_multi))
        scores_per_graph = gene_scores.view(batch_size, self.n_genes)
        weighted_expr = global_feat * scores_per_graph
        mlp_out = self.mlp(weighted_expr)
        class_logits = self.classifier(mlp_out)

        if self.domain_discriminator is not None:
            rev = GradientReversalFunction.apply(mlp_out, alpha)
            domain_logits = self.domain_discriminator(rev)
        else:
            domain_logits = None

        return class_logits, domain_logits, attn_weights


class MultiplexBatch:
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        domain_y: torch.Tensor,
        batch: torch.Tensor,
        global_feat: torch.Tensor,
        hedge_indices: Sequence[torch.Tensor],
    ) -> None:
        self.x = x
        self.y = y
        self.domain_y = domain_y
        self.batch = batch
        self.global_feat = global_feat
        self.hedge_indices = list(hedge_indices)

    def to(self, device: torch.device) -> "MultiplexBatch":
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.domain_y = self.domain_y.to(device)
        self.batch = self.batch.to(device)
        self.global_feat = self.global_feat.to(device)
        self.hedge_indices = [h.to(device) if h is not None else None for h in self.hedge_indices]
        return self


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V11 transfer run on Alzheimer dataset")
    parser.add_argument(
        "--dataset-path",
        default=os.path.join(
            "ALZHEIMERS_STRATEGIC_PATHWAY", "data", "adni", "processed", "dataset_real_ad_2000.pt"
        ),
    )
    parser.add_argument("--gene-list-path", default="")
    parser.add_argument("--h-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coexpr-threshold", type=float, default=0.70)
    parser.add_argument("--max-coexpr-edges", type=int, default=60000)
    parser.add_argument("--domain-count", type=int, default=3)
    parser.add_argument("--domain-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--force-pseudo-domains",
        action="store_true",
        help="Ignore dataset-provided domain labels and infer pseudo domains from expression.",
    )
    parser.add_argument("--disable-kegg", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_empty_hyperedge() -> torch.Tensor:
    return torch.zeros((2, 0), dtype=torch.long)


def load_gene_list(dataset_path: str, n_genes: int, explicit_path: str = "") -> List[str]:
    if explicit_path:
        path = explicit_path
    else:
        processed_dir = os.path.dirname(dataset_path)
        path = os.path.join(processed_dir, f"gene_list_{n_genes}.txt")

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            genes = [line.strip() for line in f if line.strip()]
        if len(genes) == n_genes:
            return genes
    return [f"gene_{i}" for i in range(n_genes)]


def edge_index_to_hyperedge(edge_index: torch.Tensor) -> Tuple[torch.Tensor, int]:
    if edge_index is None or edge_index.numel() == 0:
        return safe_empty_hyperedge(), 0

    src = edge_index[0].long()
    dst = edge_index[1].long()
    undirected = torch.stack([torch.minimum(src, dst), torch.maximum(src, dst)], dim=1)
    undirected = undirected[undirected[:, 0] != undirected[:, 1]]
    if undirected.numel() == 0:
        return safe_empty_hyperedge(), 0
    undirected = torch.unique(undirected, dim=0)

    hid = torch.arange(undirected.size(0), dtype=torch.long)
    node_idx = torch.cat([undirected[:, 0], undirected[:, 1]], dim=0)
    hedge_idx = torch.cat([hid, hid], dim=0)
    return torch.stack([node_idx, hedge_idx], dim=0), int(undirected.size(0))


def build_kegg_hyperedges(gene_list: Sequence[str]) -> Tuple[torch.Tensor, int]:
    gene_set = set(gene_list)
    g2i = {g: i for i, g in enumerate(gene_list)}
    ni: List[int] = []
    hi: List[int] = []
    hid = 0

    try:
        import gseapy as gp

        kegg = gp.get_library("KEGG_2021_Human")
        for genes in kegg.values():
            overlap = [g for g in set(genes) if g in gene_set]
            if len(overlap) < 3:
                continue
            for g in overlap:
                ni.append(g2i[g])
                hi.append(hid)
            hid += 1
    except Exception:
        return safe_empty_hyperedge(), 0

    if not ni:
        return safe_empty_hyperedge(), 0
    return torch.tensor([ni, hi], dtype=torch.long), hid


def assign_pseudo_domains(expr_matrix: np.ndarray, n_domains: int) -> np.ndarray:
    if n_domains <= 1:
        return np.zeros(expr_matrix.shape[0], dtype=np.int64)

    centered = expr_matrix - expr_matrix.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        pc1 = centered @ vt[0]
    except np.linalg.LinAlgError:
        pc1 = centered.mean(axis=1)

    bins = np.quantile(pc1, np.linspace(0.0, 1.0, n_domains + 1))
    bins[0] -= 1e-8
    bins[-1] += 1e-8
    return np.digitize(pc1, bins[1:-1], right=False).astype(np.int64)


def load_alz_samples(
    dataset_path: str,
    gene_list_path: str,
    n_domains: int,
    disable_kegg: bool,
    force_pseudo_domains: bool,
) -> Tuple[List[Data], Dict[str, int], List[str], torch.Tensor, int, int]:
    raw = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Dataset at {dataset_path} is empty or not a list.")

    samples: List[Data] = []
    y_raw: List[int] = []
    for i, d in enumerate(raw):
        if not hasattr(d, "x") or not hasattr(d, "y"):
            continue
        x = d.x.float()
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.size(0) == 0:
            continue
        y = int(d.y.view(-1)[0].item())
        y_raw.append(y)

        g = Data(x=x, y=torch.tensor(y, dtype=torch.long))
        g.num_nodes = int(x.size(0))
        g.sample_id = getattr(d, "sample_id", f"AD_{i:04d}")
        g.edge_index = (
            d.edge_index.clone().long()
            if hasattr(d, "edge_index") and d.edge_index is not None
            else safe_empty_hyperedge()
        )
        if hasattr(d, "global_feat") and d.global_feat is not None:
            gf = d.global_feat.float()
            if gf.dim() == 1:
                gf = gf.unsqueeze(0)
            g.global_feat = gf
        else:
            g.global_feat = x[:, 0].clone().unsqueeze(0)
        if hasattr(d, "domain_y") and d.domain_y is not None:
            g.domain_y = torch.tensor(int(d.domain_y.view(-1)[0].item()), dtype=torch.long)
        if hasattr(d, "batch_label") and d.batch_label is not None:
            g.batch_label = str(d.batch_label)
        samples.append(g)

    if not samples:
        raise ValueError("No usable samples were found in dataset.")

    unique_labels = sorted(set(y_raw))
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    for s in samples:
        s.y = torch.tensor(label_map[int(s.y.item())], dtype=torch.long)

    n_genes = int(samples[0].num_nodes)
    node_feat_dim = int(samples[0].x.size(1))
    gene_list = load_gene_list(dataset_path, n_genes, explicit_path=gene_list_path)

    expr_matrix = np.stack([s.global_feat.squeeze(0).numpy() for s in samples], axis=0)
    use_existing_domain = (not force_pseudo_domains) and all(hasattr(s, "domain_y") for s in samples)
    if use_existing_domain:
        raw_domains = [int(s.domain_y.item()) for s in samples]
        unique_domains = sorted(set(raw_domains))
        dom_map = {d: i for i, d in enumerate(unique_domains)}
        for s in samples:
            s.domain_y = torch.tensor(dom_map[int(s.domain_y.item())], dtype=torch.long)
    elif (not force_pseudo_domains) and all(hasattr(s, "batch_label") for s in samples):
        batch_names = [str(s.batch_label) for s in samples]
        unique_batches = sorted(set(batch_names))
        batch_map = {b: i for i, b in enumerate(unique_batches)}
        for s in samples:
            s.domain_y = torch.tensor(batch_map[str(s.batch_label)], dtype=torch.long)
    else:
        domain_labels = assign_pseudo_domains(expr_matrix, n_domains=n_domains)
        for i, s in enumerate(samples):
            s.domain_y = torch.tensor(int(domain_labels[i]), dtype=torch.long)

    string_hei, n_string = edge_index_to_hyperedge(samples[0].edge_index)
    if disable_kegg:
        kegg_hei, n_kegg = safe_empty_hyperedge(), 0
    else:
        kegg_hei, n_kegg = build_kegg_hyperedges(gene_list)

    for s in samples:
        s.kegg_hei = kegg_hei.clone()
        s.string_hei = string_hei.clone()

    return samples, label_map, gene_list, expr_matrix, n_kegg, n_string


def build_coexpr_hyperedges(
    expr_matrix: np.ndarray,
    train_indices: Sequence[int],
    threshold: float,
    max_edges: int,
) -> Tuple[torch.Tensor, int]:
    vals = expr_matrix[np.array(train_indices, dtype=np.int64)].T  # [n_genes, n_train]
    if vals.shape[1] < 3:
        return safe_empty_hyperedge(), 0

    ranked = np.apply_along_axis(rankdata, 1, vals)
    ranked = (ranked - ranked.mean(axis=1, keepdims=True)) / (ranked.std(axis=1, keepdims=True) + 1e-8)
    corr = ranked @ ranked.T / ranked.shape[1]
    np.fill_diagonal(corr, 0.0)

    tri_i, tri_j = np.triu_indices(corr.shape[0], k=1)
    tri_vals = np.abs(corr[tri_i, tri_j])
    keep_mask = tri_vals >= threshold
    if not np.any(keep_mask):
        return safe_empty_hyperedge(), 0

    keep_i = tri_i[keep_mask]
    keep_j = tri_j[keep_mask]
    keep_v = tri_vals[keep_mask]

    if max_edges > 0 and keep_i.shape[0] > max_edges:
        top_idx = np.argpartition(-keep_v, max_edges - 1)[:max_edges]
        keep_i = keep_i[top_idx]
        keep_j = keep_j[top_idx]

    n_edges = keep_i.shape[0]
    if n_edges == 0:
        return safe_empty_hyperedge(), 0

    hid = torch.arange(n_edges, dtype=torch.long)
    ni = torch.from_numpy(np.concatenate([keep_i, keep_j]).astype(np.int64))
    hi = torch.cat([hid, hid], dim=0)
    return torch.stack([ni, hi], dim=0), int(n_edges)


def collate_multiplex(data_list: Sequence[Data], coexpr_hei: torch.Tensor) -> MultiplexBatch:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    domain_ys: List[torch.Tensor] = []
    batches: List[torch.Tensor] = []
    global_feats: List[torch.Tensor] = []

    kegg_ni: List[torch.Tensor] = []
    kegg_hi: List[torch.Tensor] = []
    string_ni: List[torch.Tensor] = []
    string_hi: List[torch.Tensor] = []
    co_ni: List[torch.Tensor] = []
    co_hi: List[torch.Tensor] = []

    kegg_hid_offset = 0
    string_hid_offset = 0
    co_hid_offset = 0
    node_offset = 0
    n_genes = int(data_list[0].num_nodes)

    for i, d in enumerate(data_list):
        xs.append(d.x)
        ys.append(d.y)
        domain_ys.append(d.domain_y)
        batches.append(torch.full((d.num_nodes,), i, dtype=torch.long))
        global_feats.append(d.global_feat)

        if d.kegg_hei is not None and d.kegg_hei.size(1) > 0:
            kegg_ni.append(d.kegg_hei[0] + node_offset)
            max_h = int(d.kegg_hei[1].max().item()) + 1
            kegg_hi.append(d.kegg_hei[1] + kegg_hid_offset)
            kegg_hid_offset += max_h

        if d.string_hei is not None and d.string_hei.size(1) > 0:
            string_ni.append(d.string_hei[0] + node_offset)
            max_h = int(d.string_hei[1].max().item()) + 1
            string_hi.append(d.string_hei[1] + string_hid_offset)
            string_hid_offset += max_h

        if coexpr_hei is not None and coexpr_hei.size(1) > 0:
            co_ni.append(coexpr_hei[0] + node_offset)
            max_h = int(coexpr_hei[1].max().item()) + 1
            co_hi.append(coexpr_hei[1] + co_hid_offset)
            co_hid_offset += max_h

        node_offset += n_genes

    def merge(ni_list: Sequence[torch.Tensor], hi_list: Sequence[torch.Tensor]) -> torch.Tensor:
        if ni_list:
            return torch.stack([torch.cat(list(ni_list)), torch.cat(list(hi_list))], dim=0)
        return safe_empty_hyperedge()

    return MultiplexBatch(
        x=torch.cat(xs, dim=0),
        y=torch.stack(ys),
        domain_y=torch.stack(domain_ys),
        batch=torch.cat(batches, dim=0),
        global_feat=torch.cat(global_feats, dim=0),
        hedge_indices=[merge(kegg_ni, kegg_hi), merge(string_ni, string_hi), merge(co_ni, co_hi)],
    )


def make_batches(data: Sequence[Data], bs: int, coexpr_hei: torch.Tensor, shuffle: bool) -> List[MultiplexBatch]:
    idxs = np.arange(len(data))
    if shuffle:
        np.random.shuffle(idxs)
    batches = []
    for s in range(0, len(idxs), bs):
        subset = [data[int(i)] for i in idxs[s : s + bs]]
        batches.append(collate_multiplex(subset, coexpr_hei))
    return batches


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> Dict[str, float]:
    preds = np.argmax(probs, axis=1)
    acc = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, average="macro", zero_division=0))

    auroc = float("nan")
    if len(np.unique(labels)) >= 2:
        try:
            if n_classes == 2:
                auroc = float(roc_auc_score(labels, probs[:, 1]))
            else:
                auroc = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
        except Exception:
            auroc = float("nan")
    return {"acc": acc, "f1_macro": f1, "auroc_macro_ovr": auroc}


def class_weights_from_labels(labels: Sequence[int], n_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=n_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_fold(
    fold_name: str,
    train_data: Sequence[Data],
    val_data: Sequence[Data],
    coexpr_hei: torch.Tensor,
    n_genes: int,
    node_feat_dim: int,
    n_classes: int,
    n_domains: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[MultiplexGNNGuidedDANN, Dict[str, float], List[float]]:
    model = MultiplexGNNGuidedDANN(
        n_genes=n_genes,
        node_feat_dim=node_feat_dim,
        n_classes=n_classes,
        n_domains=n_domains,
        h_dim=args.h_dim,
        dropout=args.dropout,
    ).to(device)

    train_labels = [int(d.y.item()) for d in train_data]
    cls_criterion = nn.CrossEntropyLoss(
        weight=class_weights_from_labels(train_labels, n_classes=n_classes, device=device)
    )
    domain_criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-6)

    best_state = None
    best_acc = -1.0
    best_epoch = 0
    patience = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        tr_probs: List[np.ndarray] = []
        tr_y: List[np.ndarray] = []

        p = float(ep) / max(1.0, min(50.0, float(args.epochs)))
        alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

        for mb in make_batches(train_data, args.batch_size, coexpr_hei, shuffle=True):
            mb = mb.to(device)
            optim.zero_grad()
            out, domain_out, _ = model(mb.x, mb.hedge_indices, mb.batch, mb.global_feat, alpha=alpha)
            loss_cls = cls_criterion(out, mb.y)
            if domain_out is not None and n_domains > 1:
                loss_dom = domain_criterion(domain_out, mb.domain_y)
                loss = loss_cls + args.domain_loss_weight * loss_dom
            else:
                loss = loss_cls
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            n = int(mb.y.size(0))
            tr_loss += float(loss.item()) * n
            tr_n += n
            tr_probs.append(F.softmax(out, dim=1).detach().cpu().numpy())
            tr_y.append(mb.y.detach().cpu().numpy())

        scheduler.step()

        tr_probs_np = np.concatenate(tr_probs, axis=0)
        tr_y_np = np.concatenate(tr_y, axis=0)
        tr_metrics = compute_metrics(tr_probs_np, tr_y_np, n_classes=n_classes)

        model.eval()
        va_loss = 0.0
        va_n = 0
        va_probs: List[np.ndarray] = []
        va_y: List[np.ndarray] = []
        va_attn: List[np.ndarray] = []
        with torch.no_grad():
            for mb in make_batches(val_data, args.batch_size, coexpr_hei, shuffle=False):
                mb = mb.to(device)
                out, domain_out, attn = model(mb.x, mb.hedge_indices, mb.batch, mb.global_feat, alpha=0.0)
                loss_cls = cls_criterion(out, mb.y)
                if domain_out is not None and n_domains > 1:
                    loss_dom = domain_criterion(domain_out, mb.domain_y)
                    loss = loss_cls + args.domain_loss_weight * loss_dom
                else:
                    loss = loss_cls
                n = int(mb.y.size(0))
                va_loss += float(loss.item()) * n
                va_n += n
                va_probs.append(F.softmax(out, dim=1).cpu().numpy())
                va_y.append(mb.y.cpu().numpy())
                va_attn.append(attn.mean(dim=0).cpu().numpy())

        va_probs_np = np.concatenate(va_probs, axis=0)
        va_y_np = np.concatenate(va_y, axis=0)
        va_metrics = compute_metrics(va_probs_np, va_y_np, n_classes=n_classes)
        va_attn_mean = np.mean(np.stack(va_attn, axis=0), axis=0)

        print(
            f"[{fold_name}] ep={ep:03d} "
            f"train_loss={tr_loss/max(tr_n,1):.4f} train_acc={tr_metrics['acc']:.4f} "
            f"val_loss={va_loss/max(va_n,1):.4f} val_acc={va_metrics['acc']:.4f} "
            f"val_auroc={va_metrics['auroc_macro_ovr']:.4f} "
            f"attn=[{va_attn_mean[0]:.3f},{va_attn_mean[1]:.3f},{va_attn_mean[2]:.3f}]"
        )

        if va_metrics["acc"] > best_acc:
            best_acc = va_metrics["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            patience = 0
        else:
            patience += 1

        if patience >= args.patience:
            print(f"[{fold_name}] early stopping at epoch {ep}, best epoch {best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    final_probs: List[np.ndarray] = []
    final_y: List[np.ndarray] = []
    final_attn: List[np.ndarray] = []
    with torch.no_grad():
        for mb in make_batches(val_data, args.batch_size, coexpr_hei, shuffle=False):
            mb = mb.to(device)
            out, _, attn = model(mb.x, mb.hedge_indices, mb.batch, mb.global_feat, alpha=0.0)
            final_probs.append(F.softmax(out, dim=1).cpu().numpy())
            final_y.append(mb.y.cpu().numpy())
            final_attn.append(attn.mean(dim=0).cpu().numpy())

    probs_np = np.concatenate(final_probs, axis=0)
    y_np = np.concatenate(final_y, axis=0)
    metrics = compute_metrics(probs_np, y_np, n_classes=n_classes)
    attn_mean = np.mean(np.stack(final_attn, axis=0), axis=0).tolist()
    return model, metrics, attn_mean


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    dataset_path = os.path.abspath(args.dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print("=" * 88)
    print("V11 transfer run on Alzheimer dataset")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device} | Seed: {args.seed}")
    print("=" * 88)

    samples, label_map, gene_list, expr_matrix, n_kegg, n_string = load_alz_samples(
        dataset_path=dataset_path,
        gene_list_path=args.gene_list_path,
        n_domains=max(1, args.domain_count),
        disable_kegg=args.disable_kegg,
        force_pseudo_domains=args.force_pseudo_domains,
    )

    labels = np.array([int(s.y.item()) for s in samples], dtype=np.int64)
    domain_labels = np.array([int(s.domain_y.item()) for s in samples], dtype=np.int64)

    n_samples = len(samples)
    n_genes = int(samples[0].num_nodes)
    node_feat_dim = int(samples[0].x.size(1))
    n_classes = len(np.unique(labels))
    n_domains = int(max(1, domain_labels.max() + 1))

    print(
        f"Samples={n_samples} Genes={n_genes} NodeFeatDim={node_feat_dim} "
        f"Classes={n_classes} Domains={n_domains}"
    )
    print(f"Static relations: KEGG={n_kegg} STRING(pair-hyperedges)={n_string}")

    class_counts = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    min_class_count = min(class_counts.values())
    n_splits = min(args.folds, min_class_count)
    if n_splits < 2:
        raise ValueError(f"Not enough samples per class for CV. Class counts: {class_counts}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    all_results: List[Dict[str, float]] = []
    best_overall_acc = -1.0
    best_overall_state = None
    best_overall_fold = ""

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.arange(n_samples), labels), start=1):
        fold_name = f"fold_{fold_idx}"
        train_data = [samples[int(i)] for i in tr_idx]
        val_data = [samples[int(i)] for i in va_idx]

        coexpr_hei, n_coexpr = build_coexpr_hyperedges(
            expr_matrix=expr_matrix,
            train_indices=tr_idx,
            threshold=args.coexpr_threshold,
            max_edges=args.max_coexpr_edges,
        )
        print(
            f"\n[{fold_name}] train={len(train_data)} val={len(val_data)} "
            f"coexpr_edges={n_coexpr}"
        )

        model, metrics, attn = train_one_fold(
            fold_name=fold_name,
            train_data=train_data,
            val_data=val_data,
            coexpr_hei=coexpr_hei,
            n_genes=n_genes,
            node_feat_dim=node_feat_dim,
            n_classes=n_classes,
            n_domains=n_domains,
            args=args,
            device=device,
        )

        fold_model_path = os.path.join(MODEL_DIR, f"v11_alz_transfer_{fold_name}.pt")
        torch.save(model.state_dict(), fold_model_path)
        print(f"[{fold_name}] saved model -> {fold_model_path}")

        if metrics["acc"] > best_overall_acc:
            best_overall_acc = metrics["acc"]
            best_overall_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_overall_fold = fold_name

        all_results.append(
            {
                "fold": fold_idx,
                "n_train": len(train_data),
                "n_val": len(val_data),
                "n_coexpr_edges": n_coexpr,
                "acc": float(metrics["acc"]),
                "f1_macro": float(metrics["f1_macro"]),
                "auroc_macro_ovr": float(metrics["auroc_macro_ovr"]),
                "attn_kegg": float(attn[0]),
                "attn_string": float(attn[1]),
                "attn_coexpr": float(attn[2]),
            }
        )

    if best_overall_state is not None:
        best_model_path = os.path.join(MODEL_DIR, "v11_alz_transfer_best.pt")
        torch.save(best_overall_state, best_model_path)
        print(f"\nBest model saved -> {best_model_path} ({best_overall_fold})")

    valid_acc = [r["acc"] for r in all_results if not np.isnan(r["acc"])]
    valid_f1 = [r["f1_macro"] for r in all_results if not np.isnan(r["f1_macro"])]
    valid_auc = [r["auroc_macro_ovr"] for r in all_results if not np.isnan(r["auroc_macro_ovr"])]

    summary = {
        "method": "V11 transfer to Alzheimer (multiplex HGCN + MLP + DANN)",
        "dataset_path": dataset_path,
        "n_samples": n_samples,
        "n_genes": n_genes,
        "node_feature_dim": node_feat_dim,
        "n_classes": n_classes,
        "n_domains": n_domains,
        "class_counts_mapped": class_counts,
        "label_map_original_to_mapped": label_map,
        "gene_count": len(gene_list),
        "kegg_hyperedges": n_kegg,
        "string_pair_hyperedges": n_string,
        "coexpr_threshold": args.coexpr_threshold,
        "max_coexpr_edges": args.max_coexpr_edges,
        "domain_loss_weight": args.domain_loss_weight,
        "n_folds": n_splits,
        "seed": args.seed,
        "folds": all_results,
        "mean_acc": float(np.mean(valid_acc)) if valid_acc else None,
        "std_acc": float(np.std(valid_acc)) if valid_acc else None,
        "mean_f1_macro": float(np.mean(valid_f1)) if valid_f1 else None,
        "std_f1_macro": float(np.std(valid_f1)) if valid_f1 else None,
        "mean_auroc_macro_ovr": float(np.mean(valid_auc)) if valid_auc else None,
        "std_auroc_macro_ovr": float(np.std(valid_auc)) if valid_auc else None,
        "best_fold": best_overall_fold,
        "best_acc": float(best_overall_acc),
        "elapsed_minutes": (time.time() - t0) / 60.0,
    }

    out_path = os.path.join(OUT_DIR, "v11_alz_transfer_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary -> {out_path}")
    print(f"Elapsed minutes: {summary['elapsed_minutes']:.2f}")


if __name__ == "__main__":
    main()
