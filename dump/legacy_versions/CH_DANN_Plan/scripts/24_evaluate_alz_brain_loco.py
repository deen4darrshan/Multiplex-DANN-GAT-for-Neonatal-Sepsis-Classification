"""
Leave-one-cohort-out evaluation for Alzheimer brain cohorts using V11 transfer model.

This script reuses the architecture and training logic from:
  CH_DANN_Plan/scripts/22_train_v11_alzheimers_transfer.py
so all metrics stay method-consistent.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(PROJECT_ROOT)

SCRIPT_22 = os.path.join(
    PROJECT_ROOT, "CH_DANN_Plan", "scripts", "22_train_v11_alzheimers_transfer.py"
)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOCO evaluation on Alzheimer brain cohorts")
    parser.add_argument(
        "--dataset-path",
        default=os.path.join("CH_DANN_Plan", "data", "alz", "alz_brain_true_domains_2000.pt"),
    )
    parser.add_argument(
        "--gene-list-path",
        default=os.path.join("CH_DANN_Plan", "data", "alz", "gene_list_2000.txt"),
    )
    parser.add_argument("--h-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coexpr-threshold", type=float, default=0.70)
    parser.add_argument("--max-coexpr-edges", type=int, default=20000)
    parser.add_argument("--domain-count", type=int, default=3)
    parser.add_argument("--domain-loss-weight", type=float, default=0.2)
    parser.add_argument("--force-pseudo-domains", action="store_true")
    parser.add_argument("--disable-kegg", action="store_true")
    parser.add_argument(
        "--out-json",
        default=os.path.join("CH_DANN_Plan", "results", "v11_alz_brain_loco_results.json"),
    )
    return parser.parse_args()


def load_v11_module():
    spec = importlib.util.spec_from_file_location("v11_transfer", SCRIPT_22)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load script: {SCRIPT_22}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def build_train_args(args: argparse.Namespace) -> SimpleNamespace:
    # Only fields used inside train_one_fold are included.
    return SimpleNamespace(
        h_dim=args.h_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        domain_loss_weight=args.domain_loss_weight,
    )


def main() -> None:
    args = parse_args()
    mod = load_v11_module()
    mod.set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = os.path.abspath(args.dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    t0 = time.time()
    print("=" * 88)
    print("LOCO evaluation on Alzheimer brain cohorts (V11 transfer)")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device} | Seed: {args.seed}")
    print("=" * 88)

    samples, label_map, gene_list, expr_matrix, n_kegg, n_string = mod.load_alz_samples(
        dataset_path=dataset_path,
        gene_list_path=args.gene_list_path,
        n_domains=max(1, args.domain_count),
        disable_kegg=args.disable_kegg,
        force_pseudo_domains=args.force_pseudo_domains,
    )

    labels = np.array([int(s.y.item()) for s in samples], dtype=np.int64)
    domains = np.array([int(s.domain_y.item()) for s in samples], dtype=np.int64)
    batch_labels = np.array(
        [
            str(getattr(s, "batch_label", f"domain_{int(s.domain_y.item())}"))
            for s in samples
        ]
    )
    unique_batches = sorted(set(batch_labels.tolist()))
    if len(unique_batches) < 2:
        raise ValueError(f"Need at least two cohorts for LOCO, found: {unique_batches}")

    n_samples = len(samples)
    n_genes = int(samples[0].num_nodes)
    node_feat_dim = int(samples[0].x.size(1))
    n_classes = len(np.unique(labels))
    n_domains = int(max(1, domains.max() + 1))

    print(
        f"Samples={n_samples} Genes={n_genes} NodeFeatDim={node_feat_dim} "
        f"Classes={n_classes} Domains={n_domains}"
    )
    print(f"Static relations: KEGG={n_kegg} STRING(pair-hyperedges)={n_string}")
    print(f"LOCO cohorts: {unique_batches}")

    train_args = build_train_args(args)
    fold_results: List[Dict[str, object]] = []
    best_acc = -1.0
    best_fold = ""
    best_state = None

    for holdout in unique_batches:
        tr_idx = np.where(batch_labels != holdout)[0]
        va_idx = np.where(batch_labels == holdout)[0]
        train_data = [samples[int(i)] for i in tr_idx]
        val_data = [samples[int(i)] for i in va_idx]

        tr_labels = labels[tr_idx]
        va_labels = labels[va_idx]
        if len(np.unique(tr_labels)) < 2:
            raise ValueError(f"Train split has one class for holdout={holdout}")
        if len(np.unique(va_labels)) < 2:
            print(f"[holdout={holdout}] warning: validation split has one class")

        coexpr_hei, n_coexpr = mod.build_coexpr_hyperedges(
            expr_matrix=expr_matrix,
            train_indices=tr_idx,
            threshold=args.coexpr_threshold,
            max_edges=args.max_coexpr_edges,
        )

        fold_name = f"holdout_{holdout}"
        print(
            f"\n[{fold_name}] train={len(train_data)} val={len(val_data)} "
            f"coexpr_edges={n_coexpr}"
        )
        model, metrics, attn = mod.train_one_fold(
            fold_name=fold_name,
            train_data=train_data,
            val_data=val_data,
            coexpr_hei=coexpr_hei,
            n_genes=n_genes,
            node_feat_dim=node_feat_dim,
            n_classes=n_classes,
            n_domains=n_domains,
            args=train_args,
            device=device,
        )

        model_path = os.path.join(
            MODELS_DIR, f"v11_alz_brain_loco_{safe_name(holdout)}.pt"
        )
        torch.save(model.state_dict(), model_path)
        print(f"[{fold_name}] saved model -> {model_path}")

        if float(metrics["acc"]) > best_acc:
            best_acc = float(metrics["acc"])
            best_fold = holdout
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        fold_results.append(
            {
                "holdout_batch": holdout,
                "n_train": int(len(train_data)),
                "n_val": int(len(val_data)),
                "train_class_counts": {
                    int(c): int((tr_labels == c).sum()) for c in np.unique(tr_labels)
                },
                "val_class_counts": {
                    int(c): int((va_labels == c).sum()) for c in np.unique(va_labels)
                },
                "n_coexpr_edges": int(n_coexpr),
                "acc": float(metrics["acc"]),
                "f1_macro": float(metrics["f1_macro"]),
                "auroc_macro_ovr": float(metrics["auroc_macro_ovr"]),
                "attn_kegg": float(attn[0]),
                "attn_string": float(attn[1]),
                "attn_coexpr": float(attn[2]),
            }
        )

    if best_state is not None:
        best_model_path = os.path.join(MODELS_DIR, "v11_alz_brain_loco_best.pt")
        torch.save(best_state, best_model_path)
        print(f"\nBest LOCO model saved -> {best_model_path} (holdout={best_fold})")

    accs = np.array([r["acc"] for r in fold_results], dtype=np.float64)
    f1s = np.array([r["f1_macro"] for r in fold_results], dtype=np.float64)
    aucs = np.array([r["auroc_macro_ovr"] for r in fold_results], dtype=np.float64)

    summary = {
        "method": "V11 transfer to Alzheimer brain (LOCO)",
        "dataset_path": dataset_path,
        "n_samples": n_samples,
        "n_genes": n_genes,
        "node_feature_dim": node_feat_dim,
        "n_classes": n_classes,
        "n_domains": n_domains,
        "cohorts": unique_batches,
        "label_map_original_to_mapped": label_map,
        "gene_count": len(gene_list),
        "kegg_hyperedges": n_kegg,
        "string_pair_hyperedges": n_string,
        "coexpr_threshold": args.coexpr_threshold,
        "max_coexpr_edges": args.max_coexpr_edges,
        "domain_loss_weight": args.domain_loss_weight,
        "seed": args.seed,
        "folds": fold_results,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_f1_macro": float(np.mean(f1s)),
        "std_f1_macro": float(np.std(f1s)),
        "mean_auroc_macro_ovr": float(np.nanmean(aucs)),
        "std_auroc_macro_ovr": float(np.nanstd(aucs)),
        "best_holdout_batch": best_fold,
        "best_acc": float(best_acc),
        "elapsed_minutes": (time.time() - t0) / 60.0,
    }

    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved LOCO summary -> {out_path}")
    print(f"Elapsed minutes: {summary['elapsed_minutes']:.2f}")


if __name__ == "__main__":
    main()

