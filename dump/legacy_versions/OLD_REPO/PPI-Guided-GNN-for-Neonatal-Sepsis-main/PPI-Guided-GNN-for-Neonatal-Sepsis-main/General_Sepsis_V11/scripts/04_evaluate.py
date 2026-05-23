#!/usr/bin/env python3
"""
General_Sepsis_V11 - Step 04
Evaluate model, run baselines, perform statistical tests, and emit audit reports.
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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import HypergraphConv


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    today = datetime.now().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Evaluate General_Sepsis_V11")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(root / "results"))
    parser.add_argument("--log-file", default=str(root / "logs" / f"{today}_04_evaluate.log"))
    parser.add_argument("--expression-path", default=str(root / "results" / "expression_combat.csv"))
    parser.add_argument("--metadata-path", default=str(root / "results" / "metadata.csv"))
    parser.add_argument("--pathway-info-path", default=str(root / "results" / "pathway_info.json"))
    parser.add_argument("--cv-metrics-path", default=str(root / "results" / "cv_metrics_raw.json"))
    parser.add_argument("--best-model-path", default=str(root / "models" / "general_sepsis_v11_best.pt"))
    parser.add_argument("--holdout-dataset", default="GSE26378")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--permutation-n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def init_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("general_sepsis_v11_eval")
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


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= float(threshold)).astype(np.int64)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    auc = float("nan")
    if len(np.unique(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, y_prob))
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "auroc": auc}


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str,
    n_boot: int,
    seed: int,
    threshold: float = 0.5,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = y_prob[idx]
        if metric_name == "auroc":
            if len(np.unique(yb)) < 2:
                continue
            vals.append(roc_auc_score(yb, pb))
        elif metric_name == "accuracy":
            vals.append(accuracy_score(yb, (pb >= float(threshold)).astype(int)))
        elif metric_name == "f1":
            vals.append(f1_score(yb, (pb >= float(threshold)).astype(int), zero_division=0))
    if not vals:
        return {"mean": None, "lower": None, "upper": None, "n_valid": 0}
    arr = np.array(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "lower": float(np.quantile(arr, 0.025)),
        "upper": float(np.quantile(arr, 0.975)),
        "n_valid": int(arr.shape[0]),
    }


def optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    if len(j) == 0:
        return 0.5
    t = float(thr[int(np.argmax(j))])
    if not np.isfinite(t):
        return 0.5
    return float(np.clip(t, 0.01, 0.99))


def paired_permutation_test_auc(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_perm: int,
    seed: int,
) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {"delta_auc": None, "p_value": None}
    rng = np.random.default_rng(seed)
    obs = roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b)
    count = 0
    for _ in range(n_perm):
        mask = rng.integers(0, 2, size=len(y_true)).astype(bool)
        pa = scores_a.copy()
        pb = scores_b.copy()
        pa[mask], pb[mask] = pb[mask], pa[mask].copy()
        delta = roc_auc_score(y_true, pa) - roc_auc_score(y_true, pb)
        if delta >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"delta_auc": float(obs), "p_value": float(p)}


def safe_curve_arrays(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, List[float]]:
    out = {"fpr": [], "tpr": [], "roc_thresholds": [], "precision": [], "recall": [], "pr_thresholds": []}
    if len(np.unique(y_true)) < 2:
        return out
    fpr, tpr, rt = roc_curve(y_true, y_prob)
    prec, rec, prt = precision_recall_curve(y_true, y_prob)
    out["fpr"] = fpr.tolist()
    out["tpr"] = tpr.tolist()
    out["roc_thresholds"] = rt.tolist()
    out["precision"] = prec.tolist()
    out["recall"] = rec.tolist()
    out["pr_thresholds"] = prt.tolist()
    return out


def run_sklearn_baselines(
    expr: pd.DataFrame,
    train_meta: pd.DataFrame,
    folds: Sequence[Dict[str, object]],
    seed: int,
) -> Dict[str, object]:
    y_map = {"control": 0, "sepsis": 1}
    train_ids = train_meta["sample_id"].tolist()
    y_all = train_meta["condition"].str.lower().map(y_map).values.astype(np.int64)
    X_all = expr.loc[:, train_ids].T.values.astype(np.float32)
    sid_to_idx = {sid: i for i, sid in enumerate(train_ids)}

    baseline_defs = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "mlp_only": MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            alpha=1e-4,
            max_iter=500,
            random_state=seed,
        ),
        # Sanity no-MLP ablation: linear log-loss head.
        "v12_no_mlp_ablation": SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=3000,
            tol=1e-4,
            random_state=seed,
        ),
    }

    out: Dict[str, object] = {}
    for name, model in baseline_defs.items():
        fold_metrics = []
        oof = []
        for fold in folds:
            tr_ids = fold["train_sample_ids"]
            va_ids = fold["val_sample_ids"]
            tr_idx = [sid_to_idx[s] for s in tr_ids]
            va_idx = [sid_to_idx[s] for s in va_ids]
            sel = fold.get("selected_gene_indices")
            if sel:
                sel_idx = np.array(sel, dtype=np.int64)
                X_tr = X_all[tr_idx][:, sel_idx]
                X_va = X_all[va_idx][:, sel_idx]
            else:
                X_tr = X_all[tr_idx]
                X_va = X_all[va_idx]
            y_tr, y_va = y_all[tr_idx], y_all[va_idx]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

            m = copy.deepcopy(model)
            m.fit(X_tr, y_tr)
            if hasattr(m, "predict_proba"):
                prob = m.predict_proba(X_va)[:, 1]
            else:
                # SGD decision_function -> sigmoid
                dec = m.decision_function(X_va)
                prob = 1.0 / (1.0 + np.exp(-dec))
            thr = optimal_threshold(y_tr, m.predict_proba(X_tr)[:, 1] if hasattr(m, "predict_proba") else 1.0 / (1.0 + np.exp(-m.decision_function(X_tr))))
            met = compute_metrics(y_va, prob, threshold=thr)
            fold_metrics.append({"fold": int(fold["fold"]), "metrics": met})
            for sid, yt, yp in zip(va_ids, y_va.tolist(), prob.tolist()):
                oof.append({"sample_id": sid, "y_true": int(yt), "y_prob_sepsis": float(yp)})

        oof_df = pd.DataFrame(oof).sort_values("sample_id")
        global_thr = optimal_threshold(
            oof_df["y_true"].values.astype(np.int64),
            oof_df["y_prob_sepsis"].values.astype(float),
        )
        pooled = compute_metrics(
            oof_df["y_true"].values.astype(np.int64),
            oof_df["y_prob_sepsis"].values.astype(float),
            threshold=global_thr,
        )
        out[name] = {
            "fold_metrics": fold_metrics,
            "oof_predictions": oof,
            "pooled_metrics": pooled,
            "optimal_threshold": float(global_thr),
        }
    return out


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
        self.n_relations = n_relations
        self.gene_embed = nn.Sequential(nn.Linear(node_feat_dim, h_dim), nn.LayerNorm(h_dim), nn.GELU())
        self.convs1 = nn.ModuleList([HypergraphConv(h_dim, h_dim) for _ in range(n_relations)])
        self.lns1 = nn.ModuleList([nn.LayerNorm(h_dim) for _ in range(n_relations)])
        self.convs2 = nn.ModuleList([HypergraphConv(h_dim, h_dim) for _ in range(n_relations)])
        self.lns2 = nn.ModuleList([nn.LayerNorm(h_dim) for _ in range(n_relations)])
        self.relation_attn = nn.Sequential(nn.Linear(h_dim * n_relations, h_dim), nn.Tanh(), nn.Linear(h_dim, n_relations))
        self.gene_scorer = nn.Sequential(nn.Linear(h_dim, h_dim // 2), nn.Tanh(), nn.Linear(h_dim // 2, 1))
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

    def forward(self, x, hedge_indices, batch, global_feat, alpha=0.0):
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

        stacked = torch.stack(rel_outputs, dim=1)
        concat = torch.cat(rel_outputs, dim=1)
        attn_logits = self.relation_attn(concat)
        attn_weights = F.softmax(attn_logits, dim=1)
        h_multi = (stacked * attn_weights.unsqueeze(2)).sum(dim=1)
        scores = torch.sigmoid(self.gene_scorer(h_multi)).view(bs, self.n_genes)
        weighted = global_feat * scores
        mlp_out = self.mlp(weighted)
        cls = self.classifier(mlp_out)
        if self.domain_discriminator is not None:
            rev = GradientReversalFunction.apply(mlp_out, alpha)
            dom = self.domain_discriminator(rev)
        else:
            dom = None
        return cls, dom, attn_weights


def empty_hyperedge() -> torch.Tensor:
    return torch.zeros((2, 0), dtype=torch.long)


def pathways_to_hyperedge(pathways: Sequence[Dict[str, object]]) -> torch.Tensor:
    ni, hi = [], []
    for hid, p in enumerate(pathways):
        for g in p.get("gene_indices", []):
            ni.append(int(g))
            hi.append(hid)
    if not ni:
        return empty_hyperedge()
    return torch.tensor([ni, hi], dtype=torch.long)


def string_to_hyperedge(edges: Sequence[Sequence[object]]) -> torch.Tensor:
    ni, hi = [], []
    for hid, e in enumerate(edges):
        i, j = int(e[0]), int(e[1])
        if i == j:
            continue
        ni.extend([i, j])
        hi.extend([hid, hid])
    if not ni:
        return empty_hyperedge()
    return torch.tensor([ni, hi], dtype=torch.long)


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


def tile_hyperedge(base: torch.Tensor, bs: int, n_genes: int) -> torch.Tensor:
    if base is None or base.numel() == 0 or base.size(1) == 0:
        return empty_hyperedge()
    nh = int(base[1].max().item()) + 1
    ns, hs = [], []
    for b in range(bs):
        ns.append(base[0] + b * n_genes)
        hs.append(base[1] + b * nh)
    return torch.stack([torch.cat(ns), torch.cat(hs)], dim=0)


def build_coexpr_hyperedge(expr: pd.DataFrame, train_ids: Sequence[str], threshold: float, max_edges: int) -> torch.Tensor:
    data = expr.loc[:, list(train_ids)]
    ranked = data.rank(axis=1, method="average")
    arr = ranked.values.astype(np.float32)
    corr = np.corrcoef(arr)
    i, j = np.triu_indices(corr.shape[0], k=1)
    v = np.abs(corr[i, j])
    keep = v >= threshold
    i = i[keep]
    j = j[keep]
    v = v[keep]
    if max_edges > 0 and i.shape[0] > max_edges:
        top = np.argpartition(-v, max_edges - 1)[:max_edges]
        i = i[top]
        j = j[top]
    if i.shape[0] == 0:
        return empty_hyperedge()
    hid = np.arange(i.shape[0], dtype=np.int64)
    ni = np.concatenate([i.astype(np.int64), j.astype(np.int64)])
    hi = np.concatenate([hid, hid])
    return torch.tensor(np.stack([ni, hi], axis=0), dtype=torch.long)


@dataclass
class InferenceBatch:
    x: torch.Tensor
    batch: torch.Tensor
    global_feat: torch.Tensor
    hedge_indices: List[torch.Tensor]
    sample_ids: List[str]

    def to(self, device: torch.device) -> "InferenceBatch":
        self.x = self.x.to(device)
        self.batch = self.batch.to(device)
        self.global_feat = self.global_feat.to(device)
        self.hedge_indices = [h.to(device) for h in self.hedge_indices]
        return self


def collate_infer(
    sample_ids: Sequence[str],
    expr: pd.DataFrame,
    n_genes: int,
    kegg: torch.Tensor,
    string: torch.Tensor,
    coexpr: torch.Tensor,
) -> InferenceBatch:
    xs = []
    gfs = []
    bvec = []
    for b, sid in enumerate(sample_ids):
        vec = torch.tensor(expr[sid].values.astype(np.float32), dtype=torch.float32)
        xs.append(vec.view(-1, 1))
        gfs.append(vec.view(1, -1))
        bvec.append(torch.full((n_genes,), b, dtype=torch.long))
    bs = len(sample_ids)
    return InferenceBatch(
        x=torch.cat(xs, dim=0),
        batch=torch.cat(bvec, dim=0),
        global_feat=torch.cat(gfs, dim=0),
        hedge_indices=[tile_hyperedge(kegg, bs, n_genes), tile_hyperedge(string, bs, n_genes), tile_hyperedge(coexpr, bs, n_genes)],
        sample_ids=list(sample_ids),
    )


def run_external_holdout(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    pathway: Dict[str, object],
    ckpt: Dict[str, object],
    batch_size: int,
    device: torch.device,
    threshold: float,
) -> Dict[str, object]:
    model_cfg = ckpt["model_config"]
    model = MultiplexGNNGuidedDANN(
        n_genes=int(model_cfg["n_genes"]),
        node_feat_dim=int(model_cfg["node_feat_dim"]),
        n_classes=int(model_cfg["n_classes"]),
        n_domains=int(model_cfg["n_domains"]),
        h_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    train_ids = ckpt.get("train_sample_ids", meta.loc[meta["split_role"] == "train", "sample_id"].tolist())
    holdout_meta = meta.loc[meta["split_role"] == "holdout"].copy()
    holdout_ids = holdout_meta["sample_id"].tolist()
    y_map = {"control": 0, "sepsis": 1}
    y_true = holdout_meta["condition"].str.lower().map(y_map).values.astype(np.int64)
    n_genes = int(model_cfg["n_genes"])

    selected_gene_indices = ckpt.get("selected_gene_indices", [])
    if selected_gene_indices:
        selected_gene_indices = [int(i) for i in selected_gene_indices]
        sel_genes = [pathway["genes"][i] for i in selected_gene_indices]
        expr_use = expr.reindex(sel_genes)
        kegg_paths = filter_kegg_pathways(pathway.get("kegg", {}).get("pathways", []), selected_gene_indices)
        max_string_edges = int(ckpt.get("max_string_edges_train", 15000))
        string_edges = filter_string_edges(pathway.get("string", {}).get("edges", []), selected_gene_indices, max_string_edges)
        kegg = pathways_to_hyperedge(kegg_paths)
        string = string_to_hyperedge(string_edges)
    else:
        expr_use = expr
        kegg = pathways_to_hyperedge(pathway.get("kegg", {}).get("pathways", []))
        string = string_to_hyperedge(pathway.get("string", {}).get("edges", []))

    norm_mean = ckpt.get("fold_norm_mean", [])
    norm_std = ckpt.get("fold_norm_std", [])
    if norm_mean and norm_std and len(norm_mean) == expr_use.shape[0] and len(norm_std) == expr_use.shape[0]:
        mu = pd.Series(np.array(norm_mean, dtype=float), index=expr_use.index)
        sd = pd.Series(np.array(norm_std, dtype=float), index=expr_use.index).replace(0, 1.0).fillna(1.0)
        expr_use = expr_use.sub(mu, axis=0).div(sd, axis=0)

    coexpr = build_coexpr_hyperedge(
        expr=expr_use,
        train_ids=train_ids,
        threshold=float(ckpt.get("coexpr_threshold", 0.7)),
        max_edges=int(ckpt.get("max_coexpr_edges", 60000)),
    )

    probs = []
    attn_all = []
    for s in range(0, len(holdout_ids), batch_size):
        chunk_ids = holdout_ids[s : s + batch_size]
        mb = collate_infer(chunk_ids, expr_use, n_genes, kegg, string, coexpr).to(device)
        with torch.no_grad():
            logits, _, attn = model(mb.x, mb.hedge_indices, mb.batch, mb.global_feat, alpha=0.0)
            p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            probs.extend(p.tolist())
            attn_all.append(attn.mean(dim=0).detach().cpu().numpy())

    probs_np = np.array(probs, dtype=float)
    metrics = compute_metrics(y_true, probs_np, threshold=threshold)
    y_pred = (probs_np >= float(threshold)).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    attn_mean = np.mean(np.array(attn_all), axis=0).tolist() if attn_all else [0.0, 0.0, 0.0]
    curves = safe_curve_arrays(y_true, probs_np)
    return {
        "sample_ids": holdout_ids,
        "y_true": y_true.tolist(),
        "y_prob_sepsis": probs_np.tolist(),
        "metrics": metrics,
        "decision_threshold": float(threshold),
        "confusion_matrix": {
            "labels": ["control", "sepsis"],
            "matrix": cm.astype(int).tolist(),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "relation_attention_mean": {"kegg": float(attn_mean[0]), "string": float(attn_mean[1]), "coexpr": float(attn_mean[2])},
        "curves": curves,
        "coexpr_n_edges": int(coexpr.size(1)),
    }


def run_leakage_checks(
    meta: pd.DataFrame,
    cv_data: Dict[str, object],
    holdout_dataset: str,
) -> Dict[str, object]:
    train_meta = meta.loc[meta["split_role"] == "train"].copy()
    holdout_meta = meta.loc[meta["split_role"] == "holdout"].copy()
    holdout_ids = set(holdout_meta["sample_id"].tolist())
    sid_to_patient = train_meta.set_index("sample_id")["patient_id"].to_dict()
    sid_to_dataset = meta.set_index("sample_id")["dataset"].to_dict()
    checks = []
    cv_mode = str(cv_data.get("cv_mode", "sgkf")).lower()

    # Fold integrity.
    for fold in cv_data["folds"]:
        tr = set(fold["train_sample_ids"])
        va = set(fold["val_sample_ids"])
        overlap = tr.intersection(va)
        checks.append(
            {
                "name": f"fold_{fold['fold']}_train_val_sample_disjoint",
                "passed": len(overlap) == 0,
                "detail": f"overlap={len(overlap)}",
            }
        )
        # Group leakage.
        tr_pat = {sid_to_patient[s] for s in tr}
        va_pat = {sid_to_patient[s] for s in va}
        grp_overlap = tr_pat.intersection(va_pat)
        checks.append(
            {
                "name": f"fold_{fold['fold']}_group_disjoint_patient_id",
                "passed": len(grp_overlap) == 0,
                "detail": f"patient_overlap={len(grp_overlap)}",
            }
        )
        # Co-expression isolation.
        co_set = set(fold.get("coexpr_train_sample_ids", []))
        checks.append(
            {
                "name": f"fold_{fold['fold']}_coexpr_train_ids_match_fold_train",
                "passed": co_set == tr,
                "detail": f"coexpr_ids={len(co_set)} train_ids={len(tr)}",
            }
        )
        # Holdout leakage into CV.
        checks.append(
            {
                "name": f"fold_{fold['fold']}_no_holdout_in_cv",
                "passed": len((tr.union(va)).intersection(holdout_ids)) == 0,
                "detail": "intersection_with_holdout=0 expected",
            }
        )
        if cv_mode == "lodo":
            tr_ds = {sid_to_dataset[s] for s in tr}
            va_ds = {sid_to_dataset[s] for s in va}
            ds_overlap = tr_ds.intersection(va_ds)
            checks.append(
                {
                    "name": f"fold_{fold['fold']}_dataset_disjoint_lodo",
                    "passed": len(ds_overlap) == 0,
                    "detail": f"train_datasets={sorted(tr_ds)} val_datasets={sorted(va_ds)}",
                }
            )

    # Holdout dataset isolation.
    train_datasets = set(train_meta["dataset"].unique().tolist())
    checks.append(
        {
            "name": "holdout_dataset_not_in_train_split",
            "passed": holdout_dataset not in train_datasets,
            "detail": f"train_datasets={sorted(train_datasets)} holdout={holdout_dataset}",
        }
    )
    checks.append(
        {
            "name": "holdout_split_all_from_target_dataset",
            "passed": set(holdout_meta["dataset"].unique().tolist()) == {holdout_dataset},
            "detail": f"holdout_datasets={sorted(set(holdout_meta['dataset'].unique().tolist()))}",
        }
    )
    passed = all(c["passed"] for c in checks)
    return {"passed": passed, "checks": checks}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    logger = init_logger(Path(args.log_file).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=== General_Sepsis_V11 Step 04: evaluate ===")
    logger.info("device=%s seed=%d", device, args.seed)

    expr = pd.read_csv(args.expression_path, index_col=0)
    meta = pd.read_csv(args.metadata_path)
    if "sample_id" not in meta.columns and "index" in meta.columns:
        meta = meta.rename(columns={"index": "sample_id"})
    with open(args.pathway_info_path, "r", encoding="utf-8") as f:
        pathway = json.load(f)
    with open(args.cv_metrics_path, "r", encoding="utf-8") as f:
        cv_data = json.load(f)
    ckpt = torch.load(args.best_model_path, map_location="cpu")

    # Ensure expression follows pathway gene order and metadata sample order.
    genes = pathway["genes"]
    expr = expr.reindex(genes)
    expr = expr.loc[:, meta["sample_id"].tolist()]

    # Core model CV (from OOF produced in step 03).
    oof_df = pd.DataFrame(cv_data["oof_predictions"]).sort_values("sample_id")
    model_oof_y = oof_df["y_true"].values.astype(np.int64)
    model_oof_p = oof_df["y_prob_sepsis"].values.astype(float)
    model_threshold = optimal_threshold(model_oof_y, model_oof_p)
    model_oof_metrics = compute_metrics(model_oof_y, model_oof_p, threshold=model_threshold)

    # Baselines on identical folds.
    train_meta = meta.loc[meta["split_role"] == "train"].copy()
    baselines = run_sklearn_baselines(expr=expr, train_meta=train_meta, folds=cv_data["folds"], seed=args.seed)
    baseline_summary = {
        name: val["pooled_metrics"] for name, val in baselines.items()
    }

    # Best baseline by pooled AUROC.
    best_baseline_name = None
    best_baseline_auc = -1.0
    best_baseline_probs = None
    for name, val in baselines.items():
        auc = val["pooled_metrics"]["auroc"]
        if auc is not None and not math.isnan(auc) and auc > best_baseline_auc:
            best_baseline_auc = float(auc)
            best_baseline_name = name
            bdf = pd.DataFrame(val["oof_predictions"]).sort_values("sample_id")
            best_baseline_probs = bdf["y_prob_sepsis"].values.astype(float)

    # External holdout.
    external = run_external_holdout(
        expr=expr,
        meta=meta,
        pathway=pathway,
        ckpt=ckpt,
        batch_size=args.batch_size,
        device=device,
        threshold=model_threshold,
    )
    ext_y = np.array(external["y_true"], dtype=np.int64)
    ext_p = np.array(external["y_prob_sepsis"], dtype=float)

    # CIs.
    cv_ci = {
        "auroc": bootstrap_ci(model_oof_y, model_oof_p, "auroc", args.bootstrap_n, args.seed + 101),
        "accuracy": bootstrap_ci(model_oof_y, model_oof_p, "accuracy", args.bootstrap_n, args.seed + 102, threshold=model_threshold),
        "f1": bootstrap_ci(model_oof_y, model_oof_p, "f1", args.bootstrap_n, args.seed + 103, threshold=model_threshold),
    }
    ext_ci = {
        "auroc": bootstrap_ci(ext_y, ext_p, "auroc", args.bootstrap_n, args.seed + 201),
        "accuracy": bootstrap_ci(ext_y, ext_p, "accuracy", args.bootstrap_n, args.seed + 202, threshold=model_threshold),
        "f1": bootstrap_ci(ext_y, ext_p, "f1", args.bootstrap_n, args.seed + 203, threshold=model_threshold),
    }

    # Permutation test: model vs best baseline on same OOF samples.
    if best_baseline_probs is not None:
        perm = paired_permutation_test_auc(
            y_true=model_oof_y,
            scores_a=model_oof_p,
            scores_b=best_baseline_probs,
            n_perm=args.permutation_n,
            seed=args.seed + 301,
        )
    else:
        perm = {"delta_auc": None, "p_value": None}

    # Leakage checks.
    leakage = run_leakage_checks(meta=meta, cv_data=cv_data, holdout_dataset=args.holdout_dataset)

    # Gates.
    cv_fold_mean_auc = float(cv_data["cv_summary"]["mean_auroc"])
    cv_pooled_oof_auc = float(model_oof_metrics["auroc"])
    ext_auc = float(external["metrics"]["auroc"])
    ext_auc_l = ext_ci["auroc"]["lower"]
    delta_auc = perm["delta_auc"] if perm["delta_auc"] is not None else None
    p_val = perm["p_value"] if perm["p_value"] is not None else None
    gate_results = {
        "leakage_checks_pass": bool(leakage["passed"]),
        "cv_mean_auroc_ge_0_75": bool(cv_fold_mean_auc >= 0.75),
        "cv_pooled_oof_auroc_ge_0_75": bool(cv_pooled_oof_auc >= 0.75),
        "external_auroc_ge_0_70": bool(ext_auc >= 0.70),
        "external_auroc_ci_lower_gt_0_60": bool(ext_auc_l is not None and ext_auc_l > 0.60),
        "model_auc_improvement_ge_0_05": bool(delta_auc is not None and delta_auc >= 0.05),
        "permutation_p_lt_0_05": bool(p_val is not None and p_val < 0.05),
    }
    gate_results["all_passed"] = bool(all(gate_results.values()))

    # Main results JSON.
    results = {
        "generated_at": datetime.now().isoformat(),
        "seed": args.seed,
        "holdout_dataset": args.holdout_dataset,
        "cv_fold_mean_auroc": cv_fold_mean_auc,
        "cv_pooled_oof_auroc": cv_pooled_oof_auc,
        "decision_threshold_from_cv_oof": float(model_threshold),
        "model_cv_oof_metrics": model_oof_metrics,
        "model_cv_bootstrap_ci_95": cv_ci,
        "external_holdout": external,
        "external_bootstrap_ci_95": ext_ci,
        "baseline_summary": baseline_summary,
        "best_baseline_for_permutation": {
            "name": best_baseline_name,
            "auroc": None if best_baseline_name is None else baseline_summary[best_baseline_name]["auroc"],
        },
        "permutation_test_model_vs_best_baseline_auroc": perm,
        "hard_pass_gates": gate_results,
    }
    results_path = output_dir / "general_sepsis_v11_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %s", results_path)

    # Baseline detail JSON.
    baseline_payload = {
        "generated_at": datetime.now().isoformat(),
        "seed": args.seed,
        "baselines": baselines,
        "model_oof_metrics": model_oof_metrics,
        "best_baseline": {
            "name": best_baseline_name,
            "metrics": baseline_summary.get(best_baseline_name, {}),
        },
        "model_vs_best_baseline_permutation": perm,
    }
    baseline_path = output_dir / "baseline_comparison.json"
    with baseline_path.open("w", encoding="utf-8") as f:
        json.dump(baseline_payload, f, indent=2)
    logger.info("Saved %s", baseline_path)

    # Validation audit markdown.
    report_lines = []
    report_lines.append("# Validation Audit Report")
    report_lines.append("")
    report_lines.append(f"- Generated: {datetime.now().isoformat()}")
    report_lines.append(f"- Holdout dataset: `{args.holdout_dataset}`")
    report_lines.append("")
    report_lines.append("## Leakage Checks")
    report_lines.append(f"- Overall pass: **{leakage['passed']}**")
    for c in leakage["checks"]:
        mark = "PASS" if c["passed"] else "FAIL"
        report_lines.append(f"- `{mark}` {c['name']}: {c['detail']}")
    report_lines.append("")
    report_lines.append("## Statistical Checks")
    report_lines.append(f"- CV mode: {cv_data.get('cv_mode', 'sgkf')}")
    report_lines.append(f"- Operating threshold (from OOF Youden-J): {model_threshold:.4f}")
    report_lines.append(f"- CV fold-mean AUROC: {cv_fold_mean_auc:.4f}")
    report_lines.append(f"- CV pooled OOF AUROC: {cv_pooled_oof_auc:.4f}")
    report_lines.append(f"- External AUROC: {ext_auc:.4f}")
    report_lines.append(
        f"- External AUROC 95% CI: {ext_ci['auroc']['lower']} to {ext_ci['auroc']['upper']}"
    )
    report_lines.append(f"- Best baseline: {best_baseline_name} (AUROC={best_baseline_auc:.4f})")
    report_lines.append(f"- Model-baseline AUROC delta: {delta_auc}")
    report_lines.append(f"- Permutation p-value: {p_val}")
    report_lines.append("")
    report_lines.append("## Hard Gates")
    for k, v in gate_results.items():
        report_lines.append(f"- `{k}`: **{v}**")
    report_lines.append("")
    report_lines.append("## Residual Risks")
    report_lines.append("- GSE95233 fallback policy may alter adult-cohort composition when strict admission parsing yields no sepsis D00 samples.")
    report_lines.append("- External holdout is pediatric; adult-to-pediatric domain shift remains a known risk despite adversarial training.")
    report_lines.append("- Full multi-seed variance analysis is recommended for final claim hardening.")

    audit_path = output_dir / "validation_audit_report.md"
    audit_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Saved %s", audit_path)


if __name__ == "__main__":
    main()
