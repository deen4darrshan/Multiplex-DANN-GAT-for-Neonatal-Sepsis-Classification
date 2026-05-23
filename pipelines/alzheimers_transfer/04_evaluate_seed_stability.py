"""
Five-seed stability evaluation for Alzheimer brain cohorts using V11 transfer model.

This script reuses architecture/training internals from:
  CH_DANN_Plan/scripts/22_train_v11_alzheimers_transfer.py
and runs repeated stratified CV over multiple random seeds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold


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
    parser = argparse.ArgumentParser(description="Seed stability on Alzheimer brain cohorts")
    parser.add_argument(
        "--dataset-path",
        default=os.path.join("CH_DANN_Plan", "data", "alz", "alz_brain_true_domains_2000.pt"),
    )
    parser.add_argument(
        "--gene-list-path",
        default=os.path.join("CH_DANN_Plan", "data", "alz", "gene_list_2000.txt"),
    )
    parser.add_argument("--seeds", default="7,21,42,77,123")
    parser.add_argument("--h-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--coexpr-threshold", type=float, default=0.70)
    parser.add_argument("--max-coexpr-edges", type=int, default=20000)
    parser.add_argument("--domain-count", type=int, default=3)
    parser.add_argument("--domain-loss-weight", type=float, default=0.2)
    parser.add_argument("--force-pseudo-domains", action="store_true")
    parser.add_argument("--disable-kegg", action="store_true")
    parser.add_argument(
        "--out-json",
        default=os.path.join(
            "CH_DANN_Plan", "results", "v11_alz_brain_seed_stability_results.json"
        ),
    )
    parser.add_argument(
        "--per-seed-dir",
        default=os.path.join("CH_DANN_Plan", "results", "seed_stability"),
    )
    return parser.parse_args()


def load_v11_module():
    spec = importlib.util.spec_from_file_location("v11_transfer", SCRIPT_22)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load script: {SCRIPT_22}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_seed_list(text: str) -> List[int]:
    vals = []
    for chunk in text.split(","):
        c = chunk.strip()
        if not c:
            continue
        vals.append(int(c))
    if not vals:
        raise ValueError("No seeds were provided.")
    return vals


def build_train_args(args: argparse.Namespace) -> SimpleNamespace:
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


def ci95(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"low": float("nan"), "high": float("nan")}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    half = 1.96 * std / np.sqrt(max(1, values.size))
    return {"low": mean - half, "high": mean + half}


def main() -> None:
    args = parse_args()
    mod = load_v11_module()
    seeds = parse_seed_list(args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = os.path.abspath(args.dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    t0 = time.time()
    print("=" * 88)
    print("Seed stability on Alzheimer brain cohorts (V11 transfer)")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print("=" * 88)

    # Data loading is deterministic; stochasticity is handled per seed in CV/training.
    samples, label_map, gene_list, expr_matrix, n_kegg, n_string = mod.load_alz_samples(
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

    class_counts = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
    min_class_count = min(class_counts.values())
    n_splits = min(args.folds, min_class_count)
    if n_splits < 2:
        raise ValueError(f"Not enough samples per class for CV. Class counts: {class_counts}")

    train_args = build_train_args(args)
    os.makedirs(os.path.abspath(args.per_seed_dir), exist_ok=True)
    seed_summaries: List[Dict[str, object]] = []

    for seed in seeds:
        print(f"\n{'-' * 88}\nRunning seed {seed}\n{'-' * 88}")
        mod.set_seed(seed)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        fold_rows: List[Dict[str, object]] = []
        best_seed_acc = -1.0
        best_seed_fold = ""
        best_seed_state = None
        seed_start = time.time()

        for fold_idx, (tr_idx, va_idx) in enumerate(
            skf.split(np.arange(n_samples), labels), start=1
        ):
            fold_name = f"seed{seed}_fold{fold_idx}"
            train_data = [samples[int(i)] for i in tr_idx]
            val_data = [samples[int(i)] for i in va_idx]
            coexpr_hei, n_coexpr = mod.build_coexpr_hyperedges(
                expr_matrix=expr_matrix,
                train_indices=tr_idx,
                threshold=args.coexpr_threshold,
                max_edges=args.max_coexpr_edges,
            )
            print(
                f"[{fold_name}] train={len(train_data)} val={len(val_data)} "
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

            if float(metrics["acc"]) > best_seed_acc:
                best_seed_acc = float(metrics["acc"])
                best_seed_fold = fold_name
                best_seed_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

            fold_rows.append(
                {
                    "fold": fold_idx,
                    "n_train": int(len(train_data)),
                    "n_val": int(len(val_data)),
                    "n_coexpr_edges": int(n_coexpr),
                    "acc": float(metrics["acc"]),
                    "f1_macro": float(metrics["f1_macro"]),
                    "auroc_macro_ovr": float(metrics["auroc_macro_ovr"]),
                    "attn_kegg": float(attn[0]),
                    "attn_string": float(attn[1]),
                    "attn_coexpr": float(attn[2]),
                }
            )

        if best_seed_state is not None:
            model_path = os.path.join(MODELS_DIR, f"v11_alz_brain_seed_{seed}_best.pt")
            torch.save(best_seed_state, model_path)
            print(f"Saved best seed model -> {model_path} ({best_seed_fold})")

        accs = np.array([r["acc"] for r in fold_rows], dtype=np.float64)
        f1s = np.array([r["f1_macro"] for r in fold_rows], dtype=np.float64)
        aucs = np.array([r["auroc_macro_ovr"] for r in fold_rows], dtype=np.float64)

        seed_summary: Dict[str, object] = {
            "seed": seed,
            "n_folds": n_splits,
            "folds": fold_rows,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs)),
            "mean_f1_macro": float(np.mean(f1s)),
            "std_f1_macro": float(np.std(f1s)),
            "mean_auroc_macro_ovr": float(np.nanmean(aucs)),
            "std_auroc_macro_ovr": float(np.nanstd(aucs)),
            "best_fold": best_seed_fold,
            "best_acc": float(best_seed_acc),
            "elapsed_minutes": (time.time() - seed_start) / 60.0,
        }
        seed_summaries.append(seed_summary)

        per_seed_path = os.path.join(
            os.path.abspath(args.per_seed_dir),
            f"v11_alz_brain_seed_{seed}_results.json",
        )
        with open(per_seed_path, "w", encoding="utf-8") as f:
            json.dump(seed_summary, f, indent=2)
        print(f"Saved per-seed summary -> {per_seed_path}")

    seed_accs = np.array([float(s["mean_acc"]) for s in seed_summaries], dtype=np.float64)
    seed_f1s = np.array([float(s["mean_f1_macro"]) for s in seed_summaries], dtype=np.float64)
    seed_aucs = np.array(
        [float(s["mean_auroc_macro_ovr"]) for s in seed_summaries], dtype=np.float64
    )

    summary = {
        "method": "V11 transfer to Alzheimer brain (5-seed stability)",
        "dataset_path": dataset_path,
        "seeds": seeds,
        "n_seeds": len(seeds),
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
        "per_seed": seed_summaries,
        "seed_mean_acc": float(np.mean(seed_accs)),
        "seed_std_acc": float(np.std(seed_accs)),
        "seed_ci95_acc": ci95(seed_accs),
        "seed_mean_f1_macro": float(np.mean(seed_f1s)),
        "seed_std_f1_macro": float(np.std(seed_f1s)),
        "seed_ci95_f1_macro": ci95(seed_f1s),
        "seed_mean_auroc_macro_ovr": float(np.mean(seed_aucs)),
        "seed_std_auroc_macro_ovr": float(np.std(seed_aucs)),
        "seed_ci95_auroc_macro_ovr": ci95(seed_aucs),
        "elapsed_minutes": (time.time() - t0) / 60.0,
    }

    out_path = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved seed stability summary -> {out_path}")
    print(f"Elapsed minutes: {summary['elapsed_minutes']:.2f}")


if __name__ == "__main__":
    main()

