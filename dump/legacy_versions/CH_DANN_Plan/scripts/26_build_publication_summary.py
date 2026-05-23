"""
Build publication-ready tables and figures for Alzheimer transfer experiments.

Expected inputs are JSON result files in CH_DANN_Plan/results.
Outputs are written to CH_DANN_Plan/results/publication_ready.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "CH_DANN_Plan" / "results"
PUB_DIR = RESULTS_DIR / "publication_ready"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create publication-ready tables/figures")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out-dir", default=str(PUB_DIR))
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required result file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_load_json(path: Path) -> Dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(v: float | None, nd: int = 4) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "NA"
    return f"{v:.{nd}f}"


def to_metric_row(name: str, d: Dict, kind: str) -> Dict[str, object]:
    row = {
        "Experiment": name,
        "Type": kind,
        "n_samples": d.get("n_samples"),
        "n_domains": d.get("n_domains"),
        "mean_acc": d.get("mean_acc", d.get("seed_mean_acc")),
        "std_acc": d.get("std_acc", d.get("seed_std_acc")),
        "mean_f1_macro": d.get("mean_f1_macro", d.get("seed_mean_f1_macro")),
        "std_f1_macro": d.get("std_f1_macro", d.get("seed_std_f1_macro")),
        "mean_auroc_macro_ovr": d.get("mean_auroc_macro_ovr", d.get("seed_mean_auroc_macro_ovr")),
        "std_auroc_macro_ovr": d.get("std_auroc_macro_ovr", d.get("seed_std_auroc_macro_ovr")),
        "best_acc": d.get("best_acc"),
        "domain_loss_weight": d.get("domain_loss_weight"),
        "string_pair_hyperedges": d.get("string_pair_hyperedges"),
        "kegg_hyperedges": d.get("kegg_hyperedges"),
    }
    if "seed_ci95_acc" in d:
        row["seed_ci95_acc_low"] = d["seed_ci95_acc"]["low"]
        row["seed_ci95_acc_high"] = d["seed_ci95_acc"]["high"]
    return row


def markdown_table(df: pd.DataFrame) -> str:
    show = df.copy()
    for c in [
        "mean_acc",
        "std_acc",
        "mean_f1_macro",
        "std_f1_macro",
        "mean_auroc_macro_ovr",
        "std_auroc_macro_ovr",
        "best_acc",
    ]:
        if c in show.columns:
            show[c] = show[c].map(lambda x: fmt(x, nd=4) if pd.notna(x) else "NA")
    return show.to_markdown(index=False)


def build_loco_table(name: str, d: Dict) -> pd.DataFrame:
    rows = []
    for fold in d.get("folds", []):
        rows.append(
            {
                "config": name,
                "holdout_batch": fold.get("holdout_batch"),
                "n_train": fold.get("n_train"),
                "n_val": fold.get("n_val"),
                "acc": fold.get("acc"),
                "f1_macro": fold.get("f1_macro"),
                "auroc_macro_ovr": fold.get("auroc_macro_ovr"),
                "attn_kegg": fold.get("attn_kegg"),
                "attn_string": fold.get("attn_string"),
                "attn_coexpr": fold.get("attn_coexpr"),
            }
        )
    return pd.DataFrame(rows)


def build_seed_table(name: str, d: Dict) -> pd.DataFrame:
    rows = []
    for seed_row in d.get("per_seed", []):
        rows.append(
            {
                "config": name,
                "seed": seed_row.get("seed"),
                "mean_acc": seed_row.get("mean_acc"),
                "mean_f1_macro": seed_row.get("mean_f1_macro"),
                "mean_auroc_macro_ovr": seed_row.get("mean_auroc_macro_ovr"),
                "best_acc": seed_row.get("best_acc"),
                "elapsed_minutes": seed_row.get("elapsed_minutes"),
            }
        )
    return pd.DataFrame(rows)


def plot_loco(df_loco: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    cohorts = sorted(df_loco["holdout_batch"].unique().tolist())
    x = np.arange(len(cohorts))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
    for i, metric in enumerate(["acc", "auroc_macro_ovr"]):
        ax = axes[i]
        for j, cfg in enumerate(sorted(df_loco["config"].unique())):
            sub = df_loco[df_loco["config"] == cfg].set_index("holdout_batch").reindex(cohorts)
            vals = sub[metric].values.astype(float)
            offset = (-0.5 + j) * width
            ax.bar(x + offset, vals, width=width, label=cfg)
        ax.set_xticks(x)
        ax.set_xticklabels(cohorts, rotation=15)
        ax.set_ylim(0.5, 1.0)
        ax.set_ylabel("Score")
        title = "LOCO Accuracy by Held-Out Cohort" if metric == "acc" else "LOCO AUROC by Held-Out Cohort"
        ax.set_title(title)
        ax.legend(frameon=True, fontsize=8)

    fig.tight_layout()
    png = out_dir / "figure_loco_cohort_performance.png"
    pdf = out_dir / "figure_loco_cohort_performance.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_seed_stability(df_seed: pd.DataFrame, out_dir: Path) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    long_df = df_seed.melt(
        id_vars=["config", "seed"],
        value_vars=["mean_acc", "mean_auroc_macro_ovr"],
        var_name="metric",
        value_name="value",
    )
    metric_map = {"mean_acc": "Accuracy", "mean_auroc_macro_ovr": "AUROC"}
    long_df["metric"] = long_df["metric"].map(metric_map)

    fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
    sns.boxplot(
        data=long_df,
        x="metric",
        y="value",
        hue="config",
        ax=ax,
        width=0.55,
        fliersize=2,
    )
    sns.stripplot(
        data=long_df,
        x="metric",
        y="value",
        hue="config",
        dodge=True,
        ax=ax,
        size=4,
        alpha=0.6,
        linewidth=0.3,
        edgecolor="black",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], frameon=True, fontsize=8, title="Config")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Five-Seed Stability on Brain Alzheimer Transfer")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    fig.tight_layout()

    png = out_dir / "figure_seed_stability.png"
    pdf = out_dir / "figure_seed_stability.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_lineage(out_dir: Path, results_dir: Path) -> None:
    lineage_files: List[Tuple[str, str]] = [
        ("V7 SGKF", "v7_sgkf_results.json"),
        ("V8 Guided", "v8_guided_results.json"),
        ("V9 Residual", "v9_residual_results.json"),
        ("V10 Multiplex", "v10_multiplex_results.json"),
        ("V11 Multiplex+DANN", "v11_multiplex_dann_results.json"),
    ]
    rows = []
    for label, fname in lineage_files:
        p = results_dir / fname
        if not p.exists():
            continue
        d = load_json(p)
        rows.append(
            {
                "model": label,
                "mean_acc": float(d.get("mean_acc", np.nan)),
                "mean_auroc": float(d.get("mean_auroc_macro_ovr", d.get("mean_auroc", np.nan))),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(9, 4), dpi=300)
    x = np.arange(len(df))
    ax.plot(x, df["mean_acc"], marker="o", linewidth=2, label="Mean Accuracy")
    ax.plot(x, df["mean_auroc"], marker="s", linewidth=2, label="Mean AUROC")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20)
    ax.set_ylim(0.82, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Lineage Performance (Neonatal Sepsis Source Task)")
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()

    png = out_dir / "figure_model_lineage_neonatal.png"
    pdf = out_dir / "figure_model_lineage_neonatal.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    brain_cv_dann = load_json(results_dir / "v11_alz_brain_true_domains_results.json")
    brain_cv_nodann = load_json(results_dir / "v11_alz_brain_true_domains_static_nodann_results.json")
    blood_cv_dann = load_json(results_dir / "v11_alz_blood_true_domains_static_dann_results.json")
    blood_cv_nodann = load_json(results_dir / "v11_alz_blood_true_domains_static_nodann_results.json")
    loco_dann = load_json(results_dir / "v11_alz_brain_loco_dann_results.json")
    loco_nodann = load_json(results_dir / "v11_alz_brain_loco_nodann_results.json")
    seed_dann = load_json(results_dir / "v11_alz_brain_seed_stability_dann_results.json")
    seed_nodann = load_json(results_dir / "v11_alz_brain_seed_stability_nodann_results.json")

    summary_rows = [
        to_metric_row("Brain 5-fold CV (DANN)", brain_cv_dann, "CV"),
        to_metric_row("Brain 5-fold CV (No DANN)", brain_cv_nodann, "CV"),
        to_metric_row("Blood 5-fold CV (DANN)", blood_cv_dann, "CV"),
        to_metric_row("Blood 5-fold CV (No DANN)", blood_cv_nodann, "CV"),
        to_metric_row("Brain LOCO (DANN)", loco_dann, "LOCO"),
        to_metric_row("Brain LOCO (No DANN)", loco_nodann, "LOCO"),
        to_metric_row("Brain 5-seed CV (DANN)", seed_dann, "SeedStability"),
        to_metric_row("Brain 5-seed CV (No DANN)", seed_nodann, "SeedStability"),
    ]
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "table_summary_main.csv", index=False)
    with open(out_dir / "table_summary_main.md", "w", encoding="utf-8") as f:
        f.write(markdown_table(df_summary))

    df_loco = pd.concat(
        [
            build_loco_table("DANN", loco_dann),
            build_loco_table("NoDANN", loco_nodann),
        ],
        ignore_index=True,
    )
    df_loco.to_csv(out_dir / "table_loco_per_cohort.csv", index=False)

    df_seed = pd.concat(
        [
            build_seed_table("DANN", seed_dann),
            build_seed_table("NoDANN", seed_nodann),
        ],
        ignore_index=True,
    )
    df_seed.to_csv(out_dir / "table_seed_stability_per_seed.csv", index=False)

    plot_loco(df_loco, out_dir)
    plot_seed_stability(df_seed, out_dir)
    plot_lineage(out_dir, results_dir)

    manifest = {
        "summary_table_csv": str(out_dir / "table_summary_main.csv"),
        "summary_table_md": str(out_dir / "table_summary_main.md"),
        "loco_table_csv": str(out_dir / "table_loco_per_cohort.csv"),
        "seed_table_csv": str(out_dir / "table_seed_stability_per_seed.csv"),
        "figures": [
            str(out_dir / "figure_loco_cohort_performance.png"),
            str(out_dir / "figure_loco_cohort_performance.pdf"),
            str(out_dir / "figure_seed_stability.png"),
            str(out_dir / "figure_seed_stability.pdf"),
            str(out_dir / "figure_model_lineage_neonatal.png"),
            str(out_dir / "figure_model_lineage_neonatal.pdf"),
        ],
    }
    with open(out_dir / "publication_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Saved publication-ready artifacts:")
    for k, v in manifest.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()

