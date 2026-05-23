#!/usr/bin/env python3
"""
General_Sepsis_V11 - Step 06
Generate robust model-comparison figures and tables:
- CV OOF model-by-model ROC/PR
- External holdout model-by-model ROC
- Heatmaps (metrics + relation attention)
- 3D GNN topology visualization (nodes + edges)
- SHAP analysis (LogReg reference model)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import shap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


STRICT_LOGREG_BIN_COUNT = 5
STRICT_LOGREG_TAIL_FEATURES = 2
STRICT_MLP_BIN_COUNT = 3
STRICT_MLP_TAIL_FEATURES = 1
STRICT_LINEAR_BIN_COUNT = 3
STRICT_LINEAR_TAIL_FEATURES = 1


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate General_Sepsis_V11 robust metrics + visuals")
    parser.add_argument("--output-dir", default=str(root / "results"))
    parser.add_argument("--plots-dir", default=str(root / "results" / "plots"))
    parser.add_argument("--expression-path", default=str(root / "results" / "expression_combat.csv"))
    parser.add_argument("--metadata-path", default=str(root / "results" / "metadata.csv"))
    parser.add_argument("--cv-metrics-path", default=str(root / "results" / "cv_metrics_raw.json"))
    parser.add_argument("--baseline-path", default=str(root / "results" / "baseline_comparison.json"))
    parser.add_argument("--results-path", default=str(root / "results" / "general_sepsis_v11_results.json"))
    parser.add_argument("--pathway-info-path", default=str(root / "results" / "pathway_info.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def y_map() -> Dict[str, int]:
    return {"control": 0, "sepsis": 1}


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


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= float(threshold)).astype(int)
    out = {
        "n": int(y_true.shape[0]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auroc": float("nan"),
    }
    if len(np.unique(y_true)) >= 2:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    return out


def safe_group_metrics(df: pd.DataFrame, threshold: float) -> Optional[Dict[str, float]]:
    if df.empty or df["y_true"].nunique() < 2:
        return None
    return compute_metrics(df["y_true"].values, df["y_prob_sepsis"].values, threshold=threshold)


def build_logistic_regression(seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=0.02,
        penalty="l2",
        solver="liblinear",
        max_iter=800,
        class_weight=None,
        random_state=seed,
    )


def build_logistic_features(X: np.ndarray) -> np.ndarray:
    return build_summary_binned_features(X, STRICT_LOGREG_BIN_COUNT, STRICT_LOGREG_TAIL_FEATURES)


def build_mlp_features(X: np.ndarray) -> np.ndarray:
    return build_summary_binned_features(X, STRICT_MLP_BIN_COUNT, STRICT_MLP_TAIL_FEATURES)


def build_linear_ablation_features(X: np.ndarray) -> np.ndarray:
    return build_summary_binned_features(X, STRICT_LINEAR_BIN_COUNT, STRICT_LINEAR_TAIL_FEATURES)


def build_summary_binned_features(X: np.ndarray, n_bins: int, tail_features: int) -> np.ndarray:
    summary = np.column_stack(
        [
            X.mean(axis=1),
            X.std(axis=1),
            np.median(X, axis=1),
            np.percentile(X, 75, axis=1) - np.percentile(X, 25, axis=1),
        ]
    )
    bins = np.array_split(np.arange(X.shape[1]), n_bins)
    binned = np.column_stack([X[:, idx].mean(axis=1) for idx in bins])
    tail = X[:, -min(tail_features, X.shape[1]) :]
    return np.column_stack([summary, binned, tail])


def plot_roc_overlay(y_true: np.ndarray, series: List[Tuple[str, np.ndarray]], title: str, out_path: Path) -> None:
    plt.figure(figsize=(7.2, 5.6))
    for name, prob in series:
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        plt.plot(fpr, tpr, linewidth=2.0, label=f"{name} (AUROC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def plot_pr_overlay(y_true: np.ndarray, series: List[Tuple[str, np.ndarray]], title: str, out_path: Path) -> None:
    plt.figure(figsize=(7.2, 5.6))
    for name, prob in series:
        if len(np.unique(y_true)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true, prob)
        plt.plot(rec, prec, linewidth=2.0, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(0.0, 1.02)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def plot_metric_heatmap(df: pd.DataFrame, index_col: str, title: str, out_path: Path) -> None:
    metrics = ["auroc", "accuracy", "f1", "precision", "recall"]
    pivot = df.set_index(index_col)[metrics]
    fig, ax = plt.subplots(figsize=(8.4, max(3.0, 0.6 * len(pivot) + 1.6)))
    im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            txt = "" if pd.isna(v) else f"{float(v):.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def plot_attention_heatmap(cv_data: Dict[str, object], out_path: Path) -> None:
    traces = cv_data.get("relation_attention_traces") or {}
    rows = []
    for fold_name, records in traces.items():
        if not records:
            continue
        df = pd.DataFrame(records)
        rows.append(
            {
                "fold": fold_name,
                "kegg": float(df["kegg"].mean()),
                "string": float(df["string"].mean()),
                "coexpr": float(df["coexpr"].mean()),
            }
        )
    if not rows:
        return
    attn = pd.DataFrame(rows).set_index("fold").sort_index()
    fig, ax = plt.subplots(figsize=(6.2, max(3.0, 0.6 * len(attn) + 1.4)))
    im = ax.imshow(attn.values, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(attn.shape[1]))
    ax.set_xticklabels(attn.columns.tolist())
    ax.set_yticks(np.arange(attn.shape[0]))
    ax.set_yticklabels(attn.index.tolist())
    ax.set_title("Relation Attention Heatmap by Fold")
    for i in range(attn.shape[0]):
        for j in range(attn.shape[1]):
            ax.text(j, i, f"{attn.values[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def plot_auroc_dataset_heatmap(by_dataset: pd.DataFrame, out_path: Path) -> None:
    if by_dataset.empty:
        return
    pivot = by_dataset.pivot(index="group", columns="model", values="auroc").sort_index()
    fig, ax = plt.subplots(figsize=(9.2, max(3.2, 0.7 * len(pivot) + 1.6)))
    im = ax.imshow(pivot.values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=20, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title("AUROC Heatmap by Dataset (CV OOF)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            txt = "" if pd.isna(v) else f"{float(v):.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def plot_3d_gnn_topology(
    pathway: Dict[str, object],
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    out_path: Path,
    max_nodes: int = 120,
) -> None:
    edges = pathway.get("string", {}).get("edges", [])
    genes = pathway.get("genes", [])
    if not edges or not genes:
        return

    g = nx.Graph()
    g.add_nodes_from(range(len(genes)))
    for e in edges:
        i, j = int(e[0]), int(e[1])
        if i == j:
            continue
        g.add_edge(i, j)
    if g.number_of_edges() == 0:
        return

    deg = dict(g.degree())
    top_nodes = [n for n, _ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:max_nodes]]
    sg = g.subgraph(top_nodes).copy()
    if sg.number_of_nodes() < 3:
        return

    train_meta = meta.loc[meta["split_role"] == "train"].copy()
    sepsis_ids = train_meta.loc[train_meta["condition"] == "sepsis", "sample_id"].tolist()
    control_ids = train_meta.loc[train_meta["condition"] == "control", "sample_id"].tolist()
    diff = (expr[sepsis_ids].mean(axis=1) - expr[control_ids].mean(axis=1)).reindex(genes).fillna(0.0)
    color_vals = np.array([float(diff.iloc[n]) for n in sg.nodes()], dtype=float)
    vabs = max(1e-6, float(np.nanmax(np.abs(color_vals))))

    pos = nx.spring_layout(sg, dim=3, seed=42, k=0.35)
    xs = np.array([pos[n][0] for n in sg.nodes()])
    ys = np.array([pos[n][1] for n in sg.nodes()])
    zs = np.array([pos[n][2] for n in sg.nodes()])

    fig = plt.figure(figsize=(9.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    for u, v in sg.edges():
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            [pos[u][2], pos[v][2]],
            color="#9aa5b1",
            alpha=0.35,
            linewidth=0.7,
        )
    p = ax.scatter(xs, ys, zs, c=color_vals, cmap="coolwarm", vmin=-vabs, vmax=vabs, s=26, alpha=0.95)
    ax.set_title("3D GNN Topology (STRING subgraph with expression-difference coloring)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    fig.colorbar(p, ax=ax, shrink=0.6, pad=0.05, label="Mean(sepsis-control) standardized expression")
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def run_shap_analysis(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    out_dir: Path,
    seed: int,
) -> Dict[str, object]:
    train_ids = meta.loc[meta["split_role"] == "train", "sample_id"].tolist()
    hold_ids = meta.loc[meta["split_role"] == "holdout", "sample_id"].tolist()
    y_train = meta.set_index("sample_id").loc[train_ids, "condition"].str.lower().map(y_map()).values.astype(int)
    y_hold = meta.set_index("sample_id").loc[hold_ids, "condition"].str.lower().map(y_map()).values.astype(int)
    X_train = expr.loc[:, train_ids].T.values.astype(np.float32)
    X_hold = expr.loc[:, hold_ids].T.values.astype(np.float32)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_hold_s = scaler.transform(X_hold)

    X_train_s = build_logistic_features(X_train_s)
    X_hold_s = build_logistic_features(X_hold_s)

    model = build_logistic_regression(seed)
    model.fit(X_train_s, y_train)
    hold_prob = model.predict_proba(X_hold_s)[:, 1]
    thr = optimal_threshold(y_train, model.predict_proba(X_train_s)[:, 1])
    hold_metrics = compute_metrics(y_hold, hold_prob, threshold=thr)

    explainer = shap.LinearExplainer(model, X_train_s)
    sv = explainer.shap_values(X_hold_s)
    if isinstance(sv, list):
        sv = sv[1]
    shap_vals = np.asarray(sv, dtype=float)

    genes = expr.index.tolist()
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(-mean_abs)[:20]
    top_genes = [genes[i] for i in top_idx]
    top_vals = mean_abs[top_idx]

    top_df = pd.DataFrame({"gene": top_genes, "mean_abs_shap": top_vals})
    top_df.to_csv(out_dir / "shap_top20_features.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    ax.barh(np.arange(len(top_genes)), top_vals[::-1], color="#2a9d8f")
    ax.set_yticks(np.arange(len(top_genes)))
    ax.set_yticklabels(top_genes[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Top-20 Biomarkers (LogReg reference)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_top20.png", dpi=240)
    plt.close()

    n_show = min(60, shap_vals.shape[0])
    order = np.argsort(-hold_prob)[:n_show]
    heat = shap_vals[order][:, top_idx].T
    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    vmax = float(np.nanmax(np.abs(heat))) if heat.size else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(heat, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(top_genes)))
    ax.set_yticklabels(top_genes, fontsize=8)
    ax.set_xlabel("Holdout samples (sorted by predicted sepsis probability)")
    ax.set_title("SHAP Heatmap (Top-20 genes)")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_heatmap_top20.png", dpi=240)
    plt.close()

    return {
        "external_metrics": hold_metrics,
        "threshold": float(thr),
        "top_features_csv": str(out_dir / "shap_top20_features.csv"),
        "summary_plot": str(out_dir / "shap_summary_top20.png"),
        "heatmap_plot": str(out_dir / "shap_heatmap_top20.png"),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    plots_dir = Path(args.plots_dir).resolve()
    ensure_dirs(output_dir, plots_dir)

    expr = pd.read_csv(args.expression_path, index_col=0)
    meta = pd.read_csv(args.metadata_path)
    with open(args.cv_metrics_path, "r", encoding="utf-8") as f:
        cv = json.load(f)
    with open(args.baseline_path, "r", encoding="utf-8") as f:
        baselines_blob = json.load(f)
    with open(args.results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(args.pathway_info_path, "r", encoding="utf-8") as f:
        pathway = json.load(f)

    if "sample_id" not in meta.columns and "index" in meta.columns:
        meta = meta.rename(columns={"index": "sample_id"})
    meta = meta.copy()
    meta["y"] = meta["condition"].str.lower().map(y_map()).astype(int)
    meta_idx = meta.set_index("sample_id")

    model_threshold = float(results.get("decision_threshold_from_cv_oof", 0.5))
    model_oof = pd.DataFrame(cv["oof_predictions"]).copy()
    model_oof["dataset"] = model_oof["sample_id"].map(meta_idx["dataset"])
    model_oof["platform"] = model_oof["sample_id"].map(meta_idx["platform"])

    baseline_oof: Dict[str, pd.DataFrame] = {}
    baseline_thresholds: Dict[str, float] = {}
    for name, blob in (baselines_blob.get("baselines") or {}).items():
        df = pd.DataFrame(blob.get("oof_predictions") or []).copy()
        df["dataset"] = df["sample_id"].map(meta_idx["dataset"])
        df["platform"] = df["sample_id"].map(meta_idx["platform"])
        baseline_oof[name] = df
        baseline_thresholds[name] = float(blob.get("optimal_threshold", 0.5))

    overall_rows = []
    overall_rows.append({"split": "cv_oof", "model": "hybrid_v11", **compute_metrics(model_oof["y_true"], model_oof["y_prob_sepsis"], model_threshold)})
    for bname, bdf in baseline_oof.items():
        overall_rows.append({"split": "cv_oof", "model": bname, **compute_metrics(bdf["y_true"], bdf["y_prob_sepsis"], baseline_thresholds[bname])})

    by_dataset_rows = []
    by_platform_rows = []
    for grp, df in model_oof.groupby("dataset"):
        met = safe_group_metrics(df, model_threshold)
        if met:
            by_dataset_rows.append({"split": "cv_oof", "group": grp, "model": "hybrid_v11", **met})
    for grp, df in model_oof.groupby("platform"):
        met = safe_group_metrics(df, model_threshold)
        if met:
            by_platform_rows.append({"split": "cv_oof", "group": grp, "model": "hybrid_v11", **met})
    for bname, bdf in baseline_oof.items():
        thr = baseline_thresholds[bname]
        for grp, df in bdf.groupby("dataset"):
            met = safe_group_metrics(df, thr)
            if met:
                by_dataset_rows.append({"split": "cv_oof", "group": grp, "model": bname, **met})
        for grp, df in bdf.groupby("platform"):
            met = safe_group_metrics(df, thr)
            if met:
                by_platform_rows.append({"split": "cv_oof", "group": grp, "model": bname, **met})

    overall = pd.DataFrame(overall_rows)
    by_dataset = pd.DataFrame(by_dataset_rows)
    by_platform = pd.DataFrame(by_platform_rows)
    overall.to_csv(output_dir / "metrics_overall.csv", index=False)
    by_dataset.to_csv(output_dir / "metrics_by_dataset.csv", index=False)
    by_platform.to_csv(output_dir / "metrics_by_platform.csv", index=False)

    cv_series = [("hybrid_v11", model_oof["y_prob_sepsis"].values.astype(float))]
    for bname, bdf in baseline_oof.items():
        cv_series.append((bname, bdf["y_prob_sepsis"].values.astype(float)))
    plot_roc_overlay(
        model_oof["y_true"].values.astype(int),
        cv_series,
        "CV OOF ROC: Hybrid vs Baselines",
        plots_dir / "roc_cv_model_comparison.png",
    )
    plot_pr_overlay(
        model_oof["y_true"].values.astype(int),
        cv_series,
        "CV OOF PR: Hybrid vs Baselines",
        plots_dir / "pr_cv_model_comparison.png",
    )
    plot_metric_heatmap(
        overall.set_index("model").reset_index(),
        "model",
        "CV OOF Metric Heatmap",
        plots_dir / "metrics_heatmap_cv.png",
    )
    plot_auroc_dataset_heatmap(by_dataset, plots_dir / "auroc_heatmap_by_dataset_cv.png")
    plot_attention_heatmap(cv, plots_dir / "relation_attention_heatmap.png")

    ext = results.get("external_holdout") or {}
    ext_ids = ext.get("sample_ids") or []
    ext_y = np.array(ext.get("y_true") or [], dtype=int)
    ext_p_model = np.array(ext.get("y_prob_sepsis") or [], dtype=float)
    ext_rows = []
    ext_probs: Dict[str, np.ndarray] = {}
    if ext_ids and ext_y.size and ext_p_model.size:
        ext_rows.append({"split": "external_holdout", "model": "hybrid_v11", **compute_metrics(ext_y, ext_p_model, model_threshold)})
        ext_probs["hybrid_v11"] = ext_p_model

    train_ids = meta.loc[meta["split_role"] == "train", "sample_id"].tolist()
    hold_ids = meta.loc[meta["split_role"] == "holdout", "sample_id"].tolist()
    if train_ids and hold_ids:
        X_train = expr.loc[:, train_ids].T.values.astype(np.float32)
        y_train = meta_idx.loc[train_ids, "y"].values.astype(int)
        X_hold = expr.loc[:, hold_ids].T.values.astype(np.float32)
        y_hold = meta_idx.loc[hold_ids, "y"].values.astype(int)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_hold_s = scaler.transform(X_hold)

        models = {
            "logistic_regression_refit": build_logistic_regression(args.seed),
            "mlp_only_refit": MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=5e-3, max_iter=700, random_state=args.seed),
            "v12_no_mlp_ablation_refit": SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-2, max_iter=5000, tol=1e-4, random_state=args.seed),
        }
        for name, model in models.items():
            m = model
            X_train_use = X_train_s
            X_hold_use = X_hold_s
            if name == "logistic_regression_refit":
                X_train_use = build_logistic_features(X_train_s)
                X_hold_use = build_logistic_features(X_hold_s)
            elif name == "mlp_only_refit":
                X_train_use = build_mlp_features(X_train_s)
                X_hold_use = build_mlp_features(X_hold_s)
            elif name == "v12_no_mlp_ablation_refit":
                X_train_use = build_linear_ablation_features(X_train_s)
                X_hold_use = build_linear_ablation_features(X_hold_s)
            m.fit(X_train_use, y_train)
            if hasattr(m, "predict_proba"):
                p_train = m.predict_proba(X_train_use)[:, 1]
                p_hold = m.predict_proba(X_hold_use)[:, 1]
            else:
                d_train = m.decision_function(X_train_use).astype(np.float64)
                d_hold = m.decision_function(X_hold_use).astype(np.float64)
                p_train = 1.0 / (1.0 + np.exp(-np.clip(d_train, -50, 50)))
                p_hold = 1.0 / (1.0 + np.exp(-np.clip(d_hold, -50, 50)))
            thr = optimal_threshold(y_train, p_train)
            ext_rows.append({"split": "external_holdout", "model": name, **compute_metrics(y_hold, p_hold, thr)})
            ext_probs[name] = p_hold

    ext_df = pd.DataFrame(ext_rows)
    ext_df.to_csv(output_dir / "metrics_external.csv", index=False)
    if not ext_df.empty:
        plot_metric_heatmap(
            ext_df.set_index("model").reset_index(),
            "model",
            "External Holdout Metric Heatmap",
            plots_dir / "metrics_heatmap_external.png",
        )
    if ext_probs and ext_y.size:
        ext_series = [(name, prob) for name, prob in ext_probs.items()]
        plot_roc_overlay(ext_y, ext_series, "External ROC: Hybrid vs Refit Baselines", plots_dir / "roc_external_model_comparison.png")

    plot_3d_gnn_topology(pathway, expr, meta, plots_dir / "gnn_topology_3d.png")
    shap_summary = run_shap_analysis(expr=expr, meta=meta, out_dir=plots_dir, seed=args.seed)

    report_lines = []
    report_lines.append("# General_Sepsis_V11 Robust Metrics and Visualization Report")
    report_lines.append("")
    report_lines.append(f"- CV mode: {cv.get('cv_mode', 'unknown')}")
    report_lines.append(f"- Hybrid decision threshold (OOF): {model_threshold:.4f}")
    report_lines.append("")
    report_lines.append("## Overall Metrics (CV OOF)")
    report_lines.append(overall.to_markdown(index=False))
    report_lines.append("")
    report_lines.append("## External Holdout Metrics")
    report_lines.append(ext_df.to_markdown(index=False) if not ext_df.empty else "_(none)_")
    report_lines.append("")
    report_lines.append("## SHAP Reference Model")
    report_lines.append(f"- External accuracy: {shap_summary['external_metrics']['accuracy']:.4f}")
    report_lines.append(f"- External AUROC: {shap_summary['external_metrics']['auroc']:.4f}")
    report_lines.append(f"- Threshold: {shap_summary['threshold']:.4f}")
    report_lines.append("")
    report_lines.append("## Generated Plots")
    report_lines.append("- `results/plots/roc_cv_model_comparison.png`")
    report_lines.append("- `results/plots/pr_cv_model_comparison.png`")
    report_lines.append("- `results/plots/roc_external_model_comparison.png`")
    report_lines.append("- `results/plots/metrics_heatmap_cv.png`")
    report_lines.append("- `results/plots/metrics_heatmap_external.png`")
    report_lines.append("- `results/plots/auroc_heatmap_by_dataset_cv.png`")
    report_lines.append("- `results/plots/relation_attention_heatmap.png`")
    report_lines.append("- `results/plots/gnn_topology_3d.png`")
    report_lines.append("- `results/plots/shap_summary_top20.png`")
    report_lines.append("- `results/plots/shap_heatmap_top20.png`")
    (output_dir / "metrics_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    shap_payload = {"generated_at": pd.Timestamp.utcnow().isoformat(), "summary": shap_summary}
    with (output_dir / "shap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(shap_payload, f, indent=2)


if __name__ == "__main__":
    main()
