#!/usr/bin/env python3
"""
Internal ablation for General_Sepsis_V11:
replace the hybrid model's internal graph propagation blocks with GAT-style layers,
train under the same pipeline, and compare against the current hybrid checkpoint metrics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAIN_SCRIPT = ROOT / "scripts" / "03_train_v11_general_sepsis.py"
BASE_RESULTS_JSON = ROOT / "results" / "cv_metrics_raw.json"
ABLATION_RESULTS_DIR = ROOT / "results_gat_internal_ablation"
ABLATION_MODELS_DIR = ROOT / "models" / "gat_internal_ablation"
SEPSIS_RESULTS_DIR = ROOT.parents[0] / "results" / "sepsis"
TEST_JSON = SEPSIS_RESULTS_DIR / "gat_internal_swap_test.json"
TEST_MD = SEPSIS_RESULTS_DIR / "gat_internal_swap_test.md"
OVERRIDE_JSON = SEPSIS_RESULTS_DIR / "hybrid_metric_override_from_gat_swap.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GAT-internal swap ablation for hybrid model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lambda-dann", type=float, default=0.1)
    parser.add_argument("--feature-select-top-k", type=int, default=800)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_base_train_module():
    spec = importlib.util.spec_from_file_location("v11_base_train", BASE_TRAIN_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base script: {BASE_TRAIN_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def hyperedge_to_gat_edges(hei: torch.Tensor) -> torch.Tensor:
    """
    Convert hyperedge incidence [2, K] into sparse pairwise edges for GAT:
    connect consecutive nodes within each hyperedge (bidirectional).
    """
    if hei is None or hei.numel() == 0 or hei.size(1) < 2:
        return torch.zeros((2, 0), dtype=torch.long, device=hei.device if hei is not None else None)
    node_idx = hei[0]
    hedge_idx = hei[1]
    order = torch.argsort(hedge_idx)
    nodes = node_idx[order]
    hedges = hedge_idx[order]
    same_hyperedge = hedges[1:] == hedges[:-1]
    if not bool(same_hyperedge.any()):
        return torch.zeros((2, 0), dtype=torch.long, device=hei.device)
    src = nodes[:-1][same_hyperedge]
    dst = nodes[1:][same_hyperedge]
    valid = src != dst
    src = src[valid]
    dst = dst[valid]
    if src.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=hei.device)
    e0 = torch.cat([src, dst], dim=0)
    e1 = torch.cat([dst, src], dim=0)
    return torch.stack([e0, e1], dim=0)


def build_gat_swap_class(base_module):
    grl = base_module.GradientReversalFunction

    class MultiplexGATSwapDANN(nn.Module):
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
            self.dropout = dropout

            heads = 4 if h_dim % 4 == 0 else 1
            out_per_head = h_dim // heads if heads > 1 else h_dim

            self.gene_embed = nn.Sequential(
                nn.Linear(node_feat_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            )
            self.convs1 = nn.ModuleList(
                [GATConv(h_dim, out_per_head, heads=heads, concat=True, dropout=dropout, add_self_loops=False) for _ in range(n_relations)]
            )
            self.lns1 = nn.ModuleList([nn.LayerNorm(h_dim) for _ in range(n_relations)])
            self.convs2 = nn.ModuleList(
                [GATConv(h_dim, out_per_head, heads=heads, concat=True, dropout=dropout, add_self_loops=False) for _ in range(n_relations)]
            )
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

        def forward(
            self,
            x: torch.Tensor,
            hedge_indices,
            batch: torch.Tensor,
            global_feat: torch.Tensor,
            alpha: float = 1.0,
        ):
            bs = int(batch.max().item()) + 1
            g = self.gene_embed(x)
            rel_outputs = []
            for i in range(self.n_relations):
                hei = hedge_indices[i]
                edge_index = hyperedge_to_gat_edges(hei)
                if edge_index is not None and edge_index.numel() > 0:
                    h = self.convs1[i](g, edge_index)
                    h = self.lns1[i](h)
                    h = F.gelu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    r = g + h
                    h = self.convs2[i](r, edge_index)
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

            gene_scores = torch.sigmoid(self.gene_scorer(h_multi))
            scores_per_graph = gene_scores.view(bs, self.n_genes)
            weighted_expr = global_feat * scores_per_graph
            mlp_out = self.mlp(weighted_expr)
            class_logits = self.classifier(mlp_out)

            if self.domain_discriminator is not None:
                rev = grl.apply(mlp_out, alpha)
                domain_logits = self.domain_discriminator(rev)
            else:
                domain_logits = None
            return class_logits, domain_logits, attn_weights

    return MultiplexGATSwapDANN


def run_ablation(args: argparse.Namespace) -> Dict[str, float]:
    base = load_base_train_module()
    base.MultiplexGNNGuidedDANN = build_gat_swap_class(base)

    ABLATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = ROOT / "logs" / f"{datetime.now().strftime('%Y-%m-%d')}_03b_train_v11_gat_internal_ablation.log"

    argv_backup = list(sys.argv)
    sys.argv = [
        str(BASE_TRAIN_SCRIPT),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(ABLATION_RESULTS_DIR),
        "--model-dir",
        str(ABLATION_MODELS_DIR),
        "--log-file",
        str(log_file),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--dropout",
        str(args.dropout),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--lambda-dann",
        str(args.lambda_dann),
        "--feature-select-top-k",
        str(args.feature_select_top_k),
    ]
    if args.smoke:
        sys.argv.append("--smoke")

    try:
        base.main()
    finally:
        sys.argv = argv_backup

    ablation_blob = json.loads((ABLATION_RESULTS_DIR / "cv_metrics_raw.json").read_text(encoding="utf-8"))
    pooled = ablation_blob["cv_summary"]["pooled_oof"]
    return {
        "accuracy": float(pooled["accuracy"]),
        "auroc": float(pooled["auroc"]),
        "f1": float(pooled["f1"]),
    }


def load_current_hybrid_metrics() -> Dict[str, float]:
    blob = json.loads(BASE_RESULTS_JSON.read_text(encoding="utf-8"))
    pooled = blob["cv_summary"]["pooled_oof"]
    return {
        "accuracy": float(pooled["accuracy"]),
        "auroc": float(pooled["auroc"]),
        "f1": float(pooled["f1"]),
    }


def write_reports(current: Dict[str, float], swapped: Dict[str, float], config: Dict[str, object]) -> None:
    SEPSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gat_beats_hybrid = bool(swapped["auroc"] > current["auroc"] and swapped["accuracy"] >= current["accuracy"])
    payload = {
        "test": "internal_swap_hybrid_branch_to_gat",
        "decision_rule": "GAT-internal must exceed hybrid AUROC and at least match hybrid accuracy.",
        "current_hybrid_metrics": current,
        "gat_internal_swap_metrics": swapped,
        "gat_internal_beats_hybrid": gat_beats_hybrid,
        "final_decision": "replace_with_gat_internal" if gat_beats_hybrid else "keep_current_hybrid",
        "ablation_training_config": config,
        "note": "This compares the same hybrid+DANN pipeline with internal graph propagation swapped to GAT-style relation layers.",
    }
    TEST_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# GAT Internal Swap vs Current Hybrid",
        "",
        f"- Rule: {payload['decision_rule']}",
        "- Current hybrid pooled OOF:",
        f"  - Accuracy: {current['accuracy']:.4f}",
        f"  - AUROC: {current['auroc']:.4f}",
        f"  - F1: {current['f1']:.4f}",
        "- GAT-internal swap pooled OOF:",
        f"  - Accuracy: {swapped['accuracy']:.4f}",
        f"  - AUROC: {swapped['auroc']:.4f}",
        f"  - F1: {swapped['f1']:.4f}",
        f"- GAT-internal beats current hybrid: {gat_beats_hybrid}",
        f"- Final decision: `{payload['final_decision']}`",
    ]
    TEST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if gat_beats_hybrid:
        OVERRIDE_JSON.write_text(
            json.dumps(
                {
                    "accuracy": swapped["accuracy"],
                    "auroc": swapped["auroc"],
                    "f1": swapped["f1"],
                    "source": str(TEST_JSON),
                    "reason": "GAT-internal swap outperformed current hybrid on configured decision rule.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    elif OVERRIDE_JSON.exists():
        OVERRIDE_JSON.unlink()


def main() -> None:
    args = parse_args()
    swapped = run_ablation(args)
    current = load_current_hybrid_metrics()
    cfg = {
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_dann": args.lambda_dann,
        "feature_select_top_k": args.feature_select_top_k,
        "smoke": bool(args.smoke),
    }
    write_reports(current=current, swapped=swapped, config=cfg)
    print(TEST_JSON)


if __name__ == "__main__":
    main()
