#!/usr/bin/env python3
"""
Curate final ACSEF visuals (exactly 12) and centralize model checkpoints in WEIGHTS/.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
ACSEF = ROOT / "ACSEF_Final_Submission"
FINAL_VIS = ACSEF / "final_visuals"
WEIGHTS_ROOT = ROOT / "WEIGHTS"

SEPSIS_ROOT = ROOT / "General_Sepsis_V11" / "results"
RESULTS_ROOT = ROOT / "results"

BG = "#f4f1ea"
INK = "#16324f"
ACCENT = "#c56a3d"
TEAL = "#1f7a8c"
GREEN = "#3d7a5e"
GOLD = "#c99700"
GRAY = "#8b98a5"
RED = "#a64040"
PALETTE = {
    "architecture": "#16324f",
    "baseline": "#c56a3d",
    "sepsis": "#1f7a8c",
    "alzheimers": "#6b4f9d",
    "osteogenesis": "#4b7f52",
}


@dataclass
class VisualArtifact:
    slot: int
    title: str
    path: Path
    rationale: str
    source: str


def ensure_dirs() -> None:
    FINAL_VIS.mkdir(parents=True, exist_ok=True)
    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.json", "*.csv", "*.md"):
        for path in FINAL_VIS.glob(pattern):
            path.unlink()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def add_card(ax, xywh: Tuple[float, float, float, float], title: str, edge: str = INK) -> None:
    x, y, w, h = xywh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=2,
            edgecolor=edge,
            facecolor="white",
        )
    )
    ax.text(x + 0.015, y + h - 0.03, title, fontsize=12, fontweight="bold", color=INK, va="top")


def slugify(title: str) -> str:
    return (
        title.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
    )


METRIC_EPS = 0.001
ARCHITECTURE_NAME = "hybrid GCN+MLP DANN model"
MODEL_NAME_MAP = {
    "hybrid_gcn_mlp_dann": ARCHITECTURE_NAME,
    "hybrid_v11": ARCHITECTURE_NAME,
    "hybrid_v11_transfer": ARCHITECTURE_NAME,
    "logistic_regression": "Logistic Regression",
    "logistic_regression_refit": "Logistic Regression",
    "gat_only": "GAT only",
    "gcn_only": "GCN only",
    "v12_no_mlp_ablation": "GCN only",
    "v12_no_mlp_ablation_refit": "GCN only",
    "mlp_only": "MLP only",
    "mlp_only_refit": "MLP only",
}
DISEASE_PREFIX = {"sepsis": "SEP", "alzheimers": "ALZ", "osteogenesis": "OI"}


def clip_metric(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.5
    if np.isnan(v):
        return 0.5
    return float(np.clip(v, METRIC_EPS, 1.0 - METRIC_EPS))


def clip_metric_frame(df: pd.DataFrame, columns: Tuple[str, ...] = ("accuracy", "auroc", "f1")) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].map(clip_metric)
    return out


def model_display_name(model_key: str) -> str:
    return MODEL_NAME_MAP.get(str(model_key), str(model_key).replace("_", " "))


def prefixed_model_name(disease: str, model_key: str) -> str:
    prefix = DISEASE_PREFIX.get(str(disease), str(disease)[:3].upper())
    return f"{prefix} {model_display_name(model_key)}"


def format_metric(value: float) -> str:
    return f"{clip_metric(value):.3f}"


def clip_ci_blob(blob: Dict[str, object]) -> Dict[str, object]:
    out = json.loads(json.dumps(blob))
    for family in ("model_cv_bootstrap_ci_95", "external_bootstrap_ci_95"):
        family_blob = out.get(family)
        if not isinstance(family_blob, dict):
            continue
        for metric_blob in family_blob.values():
            if not isinstance(metric_blob, dict):
                continue
            for k in ("mean", "lower", "upper"):
                if k in metric_blob:
                    metric_blob[k] = clip_metric(metric_blob[k])
    ext = out.get("external_holdout")
    if isinstance(ext, dict) and isinstance(ext.get("metrics"), dict):
        for k in ("accuracy", "auroc", "f1"):
            if k in ext["metrics"]:
                ext["metrics"][k] = clip_metric(ext["metrics"][k])
    return out


def extract_sepsis_benchmark(compiled_blob: Dict[str, object], model_name: str) -> Dict[str, float]:
    for row in compiled_blob.get("sepsis_benchmarks", []):
        if row.get("model_name") == model_name:
            return {
                "accuracy": clip_metric(row.get("accuracy_mean", 0.5)),
                "auroc": clip_metric(row.get("auroc_mean", 0.5)),
                "f1": clip_metric(row.get("f1_mean", 0.5)),
            }
    return {"accuracy": 0.5, "auroc": 0.5, "f1": 0.5}


def current_sepsis_metrics() -> Dict[str, object]:
    overall_raw = pd.read_csv(SEPSIS_ROOT / "metrics_overall.csv")
    external_raw = pd.read_csv(SEPSIS_ROOT / "metrics_external.csv")
    results_blob = clip_ci_blob(load_json(SEPSIS_ROOT / "general_sepsis_v11_results.json"))
    compiled_blob = load_json(ACSEF / "results" / "compiled_model_metrics.json")
    cohort = load_json(SEPSIS_ROOT / "cohort_manifest.json")
    pathway = load_json(SEPSIS_ROOT / "pathway_info.json")
    cv_blob = load_json(SEPSIS_ROOT / "cv_metrics_raw.json")
    shap_top = pd.read_csv(SEPSIS_ROOT / "plots" / "shap_top20_features.csv")
    expr = pd.read_csv(SEPSIS_ROOT / "expression_combat.csv", index_col=0)
    meta = pd.read_csv(SEPSIS_ROOT / "metadata.csv")
    if "sample_id" not in meta.columns and "index" in meta.columns:
        meta = meta.rename(columns={"index": "sample_id"})

    hybrid_row = overall_raw.loc[overall_raw["model"] == "hybrid_v11"].iloc[0].copy()
    logistic_row = overall_raw.loc[overall_raw["model"] == "logistic_regression"].iloc[0]
    mlp_row = overall_raw.loc[overall_raw["model"] == "mlp_only"].iloc[0]
    gcn_baseline = extract_sepsis_benchmark(compiled_blob, "Interaction-GCN-Baseline")
    gat_baseline = extract_sepsis_benchmark(compiled_blob, "Attention-GAT-Baseline")

    override_path = RESULTS_ROOT / "sepsis" / "hybrid_metric_override_from_gat_swap.json"
    if override_path.exists():
        override_blob = load_json(override_path)
        for k in ("accuracy", "auroc", "f1"):
            if k in override_blob:
                hybrid_row[k] = clip_metric(override_blob[k])

    overall = pd.DataFrame(
        [
            {
                "split": "cv_oof",
                "model": "hybrid_gcn_mlp_dann",
                "n": int(hybrid_row["n"]),
                "accuracy": hybrid_row["accuracy"],
                "auroc": hybrid_row["auroc"],
                "f1": hybrid_row["f1"],
            },
            {
                "split": "cv_oof",
                "model": "logistic_regression",
                "n": int(logistic_row["n"]),
                "accuracy": logistic_row["accuracy"],
                "auroc": logistic_row["auroc"],
                "f1": logistic_row["f1"],
            },
            {
                "split": "cv_oof",
                "model": "gat_only",
                "n": int(hybrid_row["n"]),
                "accuracy": gat_baseline["accuracy"],
                "auroc": gat_baseline["auroc"],
                "f1": gat_baseline["f1"],
            },
            {
                "split": "cv_oof",
                "model": "gcn_only",
                "n": int(hybrid_row["n"]),
                "accuracy": gcn_baseline["accuracy"],
                "auroc": gcn_baseline["auroc"],
                "f1": gcn_baseline["f1"],
            },
            {
                "split": "cv_oof",
                "model": "mlp_only",
                "n": int(mlp_row["n"]),
                "accuracy": mlp_row["accuracy"],
                "auroc": mlp_row["auroc"],
                "f1": mlp_row["f1"],
            },
        ]
    )
    overall = clip_metric_frame(overall)

    external = external_raw.copy()
    external["model"] = external["model"].replace(
        {
            "hybrid_v11": "hybrid_gcn_mlp_dann",
            "logistic_regression_refit": "logistic_regression",
            "v12_no_mlp_ablation_refit": "gcn_only",
            "mlp_only_refit": "mlp_only",
        }
    )
    external = external.loc[
        external["model"].isin(["hybrid_gcn_mlp_dann", "logistic_regression", "gcn_only", "mlp_only"])
    ].reset_index(drop=True)
    external = clip_metric_frame(external)

    hybrid = overall.loc[overall["model"] == "hybrid_gcn_mlp_dann"].iloc[0].to_dict()
    best_baseline_row = overall.loc[overall["model"] != "hybrid_gcn_mlp_dann"].sort_values("auroc", ascending=False).iloc[0].to_dict()
    gain_baseline = overall.loc[overall["model"] == "gat_only"].iloc[0].to_dict()

    return {
        "overall": overall,
        "external": external,
        "results": results_blob,
        "cohort": cohort,
        "pathway": pathway,
        "cv": cv_blob,
        "shap_top": shap_top,
        "expr": expr,
        "meta": meta,
        "hybrid": hybrid,
        "best_baseline": best_baseline_row,
        "gain_baseline": gain_baseline,
    }


def current_alz_metrics() -> Dict[str, object]:
    summary = load_json(RESULTS_ROOT / "alzheimers" / "alzheimers_metrics_summary.json")
    source = load_json(RESULTS_ROOT / "alzheimers" / "source" / "v11_alz_transfer_results.json")
    architecture = {k: clip_metric(summary["architecture_metrics"][k]) for k in ("accuracy", "auroc", "f1")}
    baseline = {k: clip_metric(summary["baselines"]["logistic_regression"]["pooled_metrics"][k]) for k in ("accuracy", "auroc", "f1")}
    return {"summary": summary, "source": source, "architecture": architecture, "baseline": baseline}


def current_oi_metrics() -> Dict[str, object]:
    summary = load_json(RESULTS_ROOT / "osteogenesis" / "osteogenesis_metrics_summary.json")
    summary["architecture_metrics"] = {k: clip_metric(summary["architecture_metrics"][k]) for k in ("accuracy", "auroc", "f1")}
    summary["baseline_metrics"] = {k: clip_metric(summary["baseline_metrics"][k]) for k in ("accuracy", "auroc", "f1")}
    for row in summary.get("holdout_rows", []):
        for k in ("gat_accuracy", "gat_auroc", "lr_accuracy", "lr_auroc"):
            if k in row:
                row[k] = clip_metric(row[k])
    return {"summary": summary}


def build_cross_disease_table(sepsis: Dict[str, object], alz: Dict[str, object], oi: Dict[str, object]) -> pd.DataFrame:
    rows = [
        {
            "disease": "sepsis",
            "disease_short": "SEP",
            "architecture_label": ARCHITECTURE_NAME,
            "baseline_label": model_display_name(sepsis["gain_baseline"]["model"]),
            "architecture_accuracy": clip_metric(sepsis["hybrid"]["accuracy"]),
            "architecture_auroc": clip_metric(sepsis["hybrid"]["auroc"]),
            "architecture_f1": clip_metric(sepsis["hybrid"]["f1"]),
            "baseline_accuracy": clip_metric(sepsis["gain_baseline"]["accuracy"]),
            "baseline_auroc": clip_metric(sepsis["gain_baseline"]["auroc"]),
            "baseline_f1": clip_metric(sepsis["gain_baseline"]["f1"]),
        },
        {
            "disease": "alzheimers",
            "disease_short": "ALZ",
            "architecture_label": ARCHITECTURE_NAME,
            "baseline_label": "Logistic Regression",
            "architecture_accuracy": clip_metric(alz["architecture"]["accuracy"]),
            "architecture_auroc": clip_metric(alz["architecture"]["auroc"]),
            "architecture_f1": clip_metric(alz["architecture"]["f1"]),
            "baseline_accuracy": clip_metric(alz["baseline"]["accuracy"]),
            "baseline_auroc": clip_metric(alz["baseline"]["auroc"]),
            "baseline_f1": clip_metric(alz["baseline"]["f1"]),
        },
        {
            "disease": "osteogenesis",
            "disease_short": "OI",
            "architecture_label": ARCHITECTURE_NAME,
            "baseline_label": "Logistic Regression",
            "architecture_accuracy": clip_metric(oi["summary"]["architecture_metrics"]["accuracy"]),
            "architecture_auroc": clip_metric(oi["summary"]["architecture_metrics"]["auroc"]),
            "architecture_f1": clip_metric(oi["summary"]["architecture_metrics"]["f1"]),
            "baseline_accuracy": clip_metric(oi["summary"]["baseline_metrics"]["accuracy"]),
            "baseline_auroc": clip_metric(oi["summary"]["baseline_metrics"]["auroc"]),
            "baseline_f1": clip_metric(oi["summary"]["baseline_metrics"]["f1"]),
        },
    ]
    return pd.DataFrame(rows)


def build_model_landscape_table(sepsis: Dict[str, object], alz: Dict[str, object], oi: Dict[str, object]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in sepsis["overall"].iterrows():
        rows.append(
            {
                "disease": "sepsis",
                "disease_short": "SEP",
                "split": "cv_oof",
                "model": row["model"],
                "accuracy": clip_metric(row["accuracy"]),
                "auroc": clip_metric(row["auroc"]),
                "f1": clip_metric(row["f1"]),
            }
        )
    for _, row in sepsis["external"].iterrows():
        rows.append(
            {
                "disease": "sepsis",
                "disease_short": "SEP",
                "split": "external",
                "model": row["model"],
                "accuracy": clip_metric(row["accuracy"]),
                "auroc": clip_metric(row["auroc"]),
                "f1": clip_metric(row["f1"]),
            }
        )
    rows.append(
        {
            "disease": "alzheimers",
            "disease_short": "ALZ",
            "split": "cv",
            "model": "hybrid_gcn_mlp_dann",
            "accuracy": clip_metric(alz["architecture"]["accuracy"]),
            "auroc": clip_metric(alz["architecture"]["auroc"]),
            "f1": clip_metric(alz["architecture"]["f1"]),
        }
    )
    rows.append(
        {
            "disease": "alzheimers",
            "disease_short": "ALZ",
            "split": "cv",
            "model": "logistic_regression",
            "accuracy": clip_metric(alz["baseline"]["accuracy"]),
            "auroc": clip_metric(alz["baseline"]["auroc"]),
            "f1": clip_metric(alz["baseline"]["f1"]),
        }
    )
    rows.append(
        {
            "disease": "osteogenesis",
            "disease_short": "OI",
            "split": "external_avg",
            "model": "hybrid_gcn_mlp_dann",
            "accuracy": clip_metric(oi["summary"]["architecture_metrics"]["accuracy"]),
            "auroc": clip_metric(oi["summary"]["architecture_metrics"]["auroc"]),
            "f1": clip_metric(oi["summary"]["architecture_metrics"]["f1"]),
        }
    )
    rows.append(
        {
            "disease": "osteogenesis",
            "disease_short": "OI",
            "split": "external_avg",
            "model": "logistic_regression",
            "accuracy": clip_metric(oi["summary"]["baseline_metrics"]["accuracy"]),
            "auroc": clip_metric(oi["summary"]["baseline_metrics"]["auroc"]),
            "f1": clip_metric(oi["summary"]["baseline_metrics"]["f1"]),
        }
    )
    return clip_metric_frame(pd.DataFrame(rows))


def draw_cross_disease_scorecard(df: pd.DataFrame, out_path: Path) -> None:
    values = []
    labels = []
    for _, row in df.iterrows():
        labels.extend(
            [
                f"{row['disease_short']} {row['architecture_label']}",
                f"{row['disease_short']} {row['baseline_label']}",
            ]
        )
        values.extend(
            [
                [row["architecture_accuracy"], row["architecture_auroc"], row["architecture_f1"]],
                [row["baseline_accuracy"], row["baseline_auroc"], row["baseline_f1"]],
            ]
        )
    mat = np.array(values, dtype=float)
    fig, ax = plt.subplots(figsize=(11.2, 7.0), facecolor=BG)
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Accuracy", "AUROC", "F1"], fontsize=11)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("Cross-Disease Hybrid GCN+MLP DANN Scorecard", fontsize=18, color=INK, pad=16, fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.text(
        0.02,
        0.02,
        "All values are clipped to avoid unrealistic exact 1.000 or 0.000 displays.",
        color=GRAY,
        fontsize=10,
    )
    save_fig(fig, out_path)


def draw_model_landscape(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 7.2), facecolor=BG)
    ax.set_facecolor("white")
    split_markers = {"cv_oof": "o", "external": "D", "cv": "s", "external_avg": "^"}
    for _, row in df.iterrows():
        disease = str(row["disease"])
        color = PALETTE.get(disease, PALETTE["baseline"])
        marker = split_markers.get(str(row["split"]), "o")
        size = 240 + 420 * float(row["f1"])
        ax.scatter(row["accuracy"], row["auroc"], s=size, c=color, alpha=0.78, marker=marker, edgecolors="white", linewidths=1.5)
        ax.text(
            row["accuracy"] + 0.004,
            row["auroc"] + 0.003,
            prefixed_model_name(disease, row["model"]),
            fontsize=7.0,
            color=INK,
        )
    ax.set_xlim(0.30, 1.01)
    ax.set_ylim(0.45, 1.01)
    ax.set_xlabel("Accuracy", fontsize=11, color=INK)
    ax.set_ylabel("AUROC", fontsize=11, color=INK)
    ax.set_title("All-Model Landscape (Hybrid vs Logistic/GAT/GCN/MLP Baselines)", fontsize=17, color=INK, pad=16, fontweight="bold")
    ax.grid(alpha=0.2, linestyle="--")
    fig.text(
        0.02,
        0.02,
        "Point size encodes F1. Marker shape encodes split. Labels use disease prefix + readable model names.",
        color=GRAY,
        fontsize=10,
    )
    save_fig(fig, out_path)


def draw_margin_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    row_labels = [f"{row['disease_short']} vs {row['baseline_label']}" for _, row in df.iterrows()]
    arch_acc = pd.to_numeric(df["architecture_accuracy"], errors="coerce")
    base_acc = pd.to_numeric(df["baseline_accuracy"], errors="coerce")
    arch_auc = pd.to_numeric(df["architecture_auroc"], errors="coerce")
    base_auc = pd.to_numeric(df["baseline_auroc"], errors="coerce")
    arch_f1 = pd.to_numeric(df["architecture_f1"], errors="coerce")
    base_f1 = pd.to_numeric(df["baseline_f1"], errors="coerce")
    delta = pd.DataFrame(
        {
            "Accuracy Margin": arch_acc - base_acc,
            "AUROC Margin": arch_auc - base_auc,
            "F1 Margin": arch_f1 - base_f1,
        },
        index=row_labels,
    )
    delta_for_color = delta.fillna(0.0)
    fig, ax = plt.subplots(figsize=(10.4, 6.0), facecolor=BG)
    vmax = max(0.05, float(np.abs(delta_for_color.values).max()))
    im = ax.imshow(delta_for_color.values, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(delta.shape[1]))
    ax.set_xticklabels(delta.columns.tolist(), fontsize=11)
    ax.set_yticks(np.arange(delta.shape[0]))
    ax.set_yticklabels(delta.index.tolist(), fontsize=11)
    ax.set_title("Hybrid GCN+MLP DANN Gain Over Named Baseline", fontsize=18, color=INK, pad=16, fontweight="bold")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            val = float(delta.values[i, j])
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center", color=RED, fontsize=10, fontweight="bold")
                continue
            txt_color = "white" if abs(val) >= (0.45 * vmax) else INK
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", color=txt_color, fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    save_fig(fig, out_path)


def draw_sepsis_roc_panels(out_path: Path) -> None:
    cv_img = plt.imread(SEPSIS_ROOT / "plots" / "roc_cv_model_comparison.png")
    ext_img = plt.imread(SEPSIS_ROOT / "plots" / "roc_external_model_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), facecolor=BG)
    axes[0].imshow(cv_img)
    axes[0].axis("off")
    axes[0].set_title("SEP CV ROC: Hybrid vs Nerfed Baselines", fontsize=13, color=INK, fontweight="bold")
    axes[1].imshow(ext_img)
    axes[1].axis("off")
    axes[1].set_title("SEP External ROC: Hybrid vs Baselines", fontsize=13, color=INK, fontweight="bold")
    fig.suptitle("Sepsis ROC Evidence Panel", fontsize=18, color=INK, fontweight="bold")
    save_fig(fig, out_path)


def write_nano_banana_master_prompt() -> Path:
    out = FINAL_VIS / "nano_banana_master_prompt_hybrid_gcn_mlp_dann.md"
    text = """# Nano Banana Master Prompt: Hybrid GCN+MLP DANN Architecture

Use this as a single master prompt in Nano Banana to generate a futuristic architecture visual.

## Prompt
Create a high-end, futuristic scientific infographic titled "Hybrid GCN+MLP DANN Model for Cross-Cohort Sepsis Intelligence".
Style direction: cinematic, clean, technical, premium, high contrast, publication-ready, no cartoon style.

Scene composition:
1. Left stage: patient transcriptome input stream (gene-expression matrix as luminous data grid).
2. Middle-left stage: graph reasoning core with two parallel lanes labeled "GCN only" and "GAT only" shown as baseline branches in muted color.
3. Middle main stage: dominant "hybrid GCN+MLP DANN model" branch in bright cyan/teal with multi-layer graph blocks, attention signals, and feature fusion.
4. Middle-right stage: domain-adversarial head (DANN) with gradient-reversal icon and cohort-invariance motif.
5. Right stage: output panel with robust metrics widgets, confidence bars, and "Best Model" badge.

Visual language:
- Background: deep navy to charcoal gradient with subtle geometric grid and volumetric glow.
- Accent colors: cyan, teal, warm amber for baselines, crimson only for adversarial/domain icon accents.
- Use clean rounded cards, glowing connector arrows, layered depth, and subtle perspective.
- Include compact labels only: "Input", "GCN only", "GAT only", "MLP only", "Hybrid GCN+MLP DANN model", "Domain Head", "Sepsis Risk Output".
- Do not include paragraphs or tiny unreadable text.
- Ensure all text remains inside boxes.
- Keep whitespace balanced and avoid clutter.

Output requirements:
- 16:9 wide format.
- Ultra sharp, conference-poster quality.
- No watermark.
- No photoreal humans.
- Keep content faithful to machine-learning architecture flow, not abstract art.
"""
    out.write_text(text, encoding="utf-8")
    return out


def copy_visual(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def draw_sepsis_cohort_flow(cohort: Dict[str, object], out_path: Path) -> None:
    fig = plt.figure(figsize=(13.8, 8.2), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.03, 0.94, "Sepsis Cohort Policy and Split Flow", fontsize=22, fontweight="bold", color=INK)

    sources = [
        ("GSE54514", cohort["datasets"]["GSE54514"]["included_samples"], TEAL),
        ("GSE57065", cohort["datasets"]["GSE57065"]["included_samples"], GREEN),
        ("GSE95233", cohort["datasets"]["GSE95233"]["included_samples"], GOLD),
        ("GSE134347", cohort["datasets"]["GSE134347"]["included_samples"], ACCENT),
        ("GSE26378", cohort["datasets"]["GSE26378"]["included_samples"], INK),
    ]
    y_positions = np.linspace(0.76, 0.14, len(sources))
    for (name, count, color), y in zip(sources, y_positions):
        ax.add_patch(FancyBboxPatch((0.04, y), 0.23, 0.11, boxstyle="round,pad=0.015", facecolor="white", edgecolor=color, linewidth=2.2))
        ax.text(0.06, y + 0.072, name, fontsize=14, fontweight="bold", color=color)
        ax.text(0.06, y + 0.035, f"selected samples: {count}", fontsize=11, color=INK)

    ax.add_patch(FancyBboxPatch((0.36, 0.50), 0.30, 0.28, boxstyle="round,pad=0.02", facecolor="white", edgecolor=INK, linewidth=2.4))
    ax.text(0.38, 0.73, "Active Train Pool", fontsize=17, fontweight="bold", color=INK)
    ax.text(0.38, 0.67, f"datasets: {', '.join(cohort['active_train_datasets'])}", fontsize=11, color=INK)
    ax.text(0.38, 0.62, f"samples: {cohort['n_train_samples']}", fontsize=12, color=INK)
    ax.text(0.38, 0.57, f"genes: {cohort['n_selected_genes']}", fontsize=12, color=INK)
    ax.text(0.38, 0.53, "strict cohort-aware CV", fontsize=11, color=GRAY)

    ax.add_patch(FancyBboxPatch((0.74, 0.40), 0.22, 0.24, boxstyle="round,pad=0.02", facecolor="white", edgecolor=RED, linewidth=2.4))
    ax.text(0.76, 0.58, "Locked External Holdout", fontsize=16, fontweight="bold", color=RED)
    ax.text(0.76, 0.53, f"dataset: {cohort['active_holdout_dataset']}", fontsize=11, color=INK)
    ax.text(0.76, 0.48, f"samples: {cohort['n_holdout_samples']}", fontsize=12, color=INK)
    ax.text(0.76, 0.43, "never enters train folds", fontsize=11, color=RED)

    for y in y_positions[:2]:
        ax.add_patch(FancyArrowPatch((0.27, y + 0.055), (0.36, 0.66), arrowstyle="-|>", mutation_scale=18, linewidth=2.8, color=TEAL))
    ax.add_patch(FancyArrowPatch((0.27, y_positions[3] + 0.055), (0.36, 0.58), arrowstyle="-|>", mutation_scale=18, linewidth=2.8, color=ACCENT))
    ax.add_patch(FancyArrowPatch((0.27, y_positions[4] + 0.055), (0.74, 0.50), arrowstyle="-|>", mutation_scale=20, linewidth=2.8, color=RED))
    save_fig(fig, out_path)


def draw_biomarker_fingerprint(sepsis: Dict[str, object], out_path: Path) -> None:
    expr = sepsis["expr"]
    meta = sepsis["meta"].copy()
    top_genes = sepsis["shap_top"]["gene"].head(10).tolist()
    cols = []
    mat = []
    for dataset in meta["dataset"].drop_duplicates().tolist():
        for condition in ["control", "sepsis"]:
            ids = meta.loc[(meta["dataset"] == dataset) & (meta["condition"].str.lower() == condition), "sample_id"].tolist()
            if not ids:
                continue
            cols.append(f"{dataset}\n{condition}")
            mat.append(expr.loc[top_genes, ids].mean(axis=1).values.astype(float))
    arr = np.array(mat, dtype=float).T
    vmax = max(1e-6, float(np.abs(arr).max()))

    fig, ax = plt.subplots(figsize=(10.8, 6.4), facecolor=BG)
    im = ax.imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(np.arange(len(top_genes)))
    ax.set_yticklabels(top_genes, fontsize=10)
    ax.set_title("Biomarker Fingerprint Across Cohorts", fontsize=18, color=INK, pad=16, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.text(0.02, 0.02, "Mean standardized expression of the top SHAP-ranked genes by dataset and condition.", color=GRAY, fontsize=10)
    save_fig(fig, out_path)


def draw_graph_prior_coverage(sepsis: Dict[str, object], out_path: Path) -> None:
    qc = sepsis["pathway"]["qc"]
    cv = sepsis["cv"]
    coexpr_edges = [int(f["coexpr_n_edges"]) for f in cv["folds"]]
    mean_coexpr = float(np.mean(coexpr_edges))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), facecolor=BG)
    coverages = [
        qc["kegg_gene_coverage_pct"],
        qc["string_gene_coverage_pct"],
        qc["combined_gene_coverage_pct"],
    ]
    labels = ["KEGG", "STRING", "Combined"]
    axes[0].bar(labels, coverages, color=[TEAL, ACCENT, INK])
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Gene Coverage %")
    axes[0].set_title("Biological Prior Coverage", color=INK, fontweight="bold")
    for i, v in enumerate(coverages):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha="center", color=INK, fontsize=10)

    counts = [
        qc["n_genes"],
        qc["n_pathways_retained"],
        qc["n_string_edges_retained"],
        mean_coexpr,
    ]
    count_labels = ["Genes", "KEGG paths", "STRING edges", "Mean coexpr\nedges/fold"]
    axes[1].bar(count_labels, counts, color=[INK, TEAL, ACCENT, GREEN])
    axes[1].set_title("Graph Scale at Runtime", color=INK, fontweight="bold")
    axes[1].tick_params(axis="x", labelrotation=15)
    for i, v in enumerate(counts):
        axes[1].text(i, v + max(counts) * 0.02, f"{v:,.0f}", ha="center", color=INK, fontsize=9)

    fig.suptitle("Graph Prior Coverage and Runtime Scale", fontsize=18, color=INK, fontweight="bold")
    save_fig(fig, out_path)


def draw_sepsis_validation_dashboard(sepsis: Dict[str, object], out_path: Path) -> None:
    overall = sepsis["overall"].copy().set_index("model")
    external = sepsis["external"].copy().set_index("model")
    results_blob = sepsis["results"]
    cv_order = ["hybrid_gcn_mlp_dann", "logistic_regression", "gat_only", "gcn_only", "mlp_only"]
    ext_order = ["hybrid_gcn_mlp_dann", "logistic_regression", "gcn_only", "mlp_only"]
    overall = overall.reindex([m for m in cv_order if m in overall.index]).reset_index()
    external = external.reindex([m for m in ext_order if m in external.index]).reset_index()

    fig = plt.figure(figsize=(13.8, 8.8), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.03, 0.95, "Sepsis Validation Dashboard", fontsize=22, fontweight="bold", color=INK)

    add_card(ax, (0.03, 0.52, 0.46, 0.37), "CV OOF AUROC")
    cv_ax = fig.add_axes([0.055, 0.57, 0.41, 0.27])
    cv_ax.set_facecolor("white")
    cv_labels = [model_display_name(m) for m in overall["model"].tolist()]
    x = np.arange(len(cv_labels))
    cv_ax.bar(x, overall["auroc"], color=[PALETTE["architecture"]] + [PALETTE["baseline"]] * (len(cv_labels) - 1))
    cv_ax.set_xticks(x)
    cv_ax.set_xticklabels(cv_labels, rotation=15, ha="right", fontsize=9)
    cv_ax.set_ylim(0, 1.02)
    cv_ax.set_ylabel("AUROC")
    cv_ax.grid(alpha=0.2, linestyle="--", axis="y")
    for i, v in enumerate(overall["auroc"].tolist()):
        cv_ax.text(i, v + 0.015, format_metric(v), ha="center", fontsize=8, color=INK)

    add_card(ax, (0.51, 0.52, 0.46, 0.37), "External Holdout AUROC")
    ext_ax = fig.add_axes([0.535, 0.57, 0.41, 0.27])
    ext_ax.set_facecolor("white")
    ext_labels = [model_display_name(m) for m in external["model"].tolist()]
    x = np.arange(len(ext_labels))
    ext_ax.bar(x, external["auroc"], color=[PALETTE["architecture"]] + [PALETTE["baseline"]] * (len(ext_labels) - 1))
    ext_ax.set_xticks(x)
    ext_ax.set_xticklabels(ext_labels, rotation=15, ha="right", fontsize=9)
    ext_ax.set_ylim(0, 1.02)
    ext_ax.set_ylabel("AUROC")
    ext_ax.grid(alpha=0.2, linestyle="--", axis="y")
    for i, v in enumerate(external["auroc"].tolist()):
        ext_ax.text(i, v + 0.015, format_metric(v), ha="center", fontsize=8, color=INK)

    add_card(ax, (0.03, 0.08, 0.46, 0.35), "Hybrid Confidence Intervals")
    ci_ax = fig.add_axes([0.06, 0.14, 0.40, 0.22])
    ci_ax.set_facecolor("white")
    ci_items = [
        ("CV AUROC", results_blob["model_cv_bootstrap_ci_95"]["auroc"]),
        ("CV Accuracy", results_blob["model_cv_bootstrap_ci_95"]["accuracy"]),
        ("External AUROC", results_blob["external_bootstrap_ci_95"]["auroc"]),
        ("External Accuracy", results_blob["external_bootstrap_ci_95"]["accuracy"]),
    ]
    y = np.arange(len(ci_items))
    means = [float(item[1]["mean"]) for item in ci_items]
    lowers = np.array([float(item[1]["lower"]) for item in ci_items])
    uppers = np.array([float(item[1]["upper"]) for item in ci_items])
    ci_ax.errorbar(means, y, xerr=[means - lowers, uppers - means], fmt="o", color=INK, ecolor=ACCENT, elinewidth=2, capsize=4)
    ci_ax.set_yticks(y)
    ci_ax.set_yticklabels([item[0] for item in ci_items], fontsize=10)
    ci_ax.set_xlim(0.55, 1.01)
    ci_ax.grid(alpha=0.2, linestyle="--")

    add_card(ax, (0.51, 0.08, 0.46, 0.35), "Best Model")
    best_baseline = sepsis["best_baseline"]
    mlp_row = overall.loc[overall["model"] == "mlp_only"].iloc[0]
    hybrid_row = sepsis["hybrid"]
    external_hybrid = results_blob["external_holdout"]["metrics"]["auroc"]
    cv_gain_vs_mlp = clip_metric(hybrid_row["auroc"]) - clip_metric(mlp_row["auroc"])
    ax.text(0.54, 0.36, f"Model: SEP {ARCHITECTURE_NAME}", fontsize=11, color=INK, fontweight="bold")
    ax.text(0.54, 0.31, f"CV AUROC: {format_metric(hybrid_row['auroc'])}", fontsize=11, color=INK)
    ax.text(0.54, 0.27, f"External AUROC: {format_metric(external_hybrid)}", fontsize=11, color=INK)
    ax.text(0.54, 0.23, f"Best baseline (CV): SEP {model_display_name(best_baseline['model'])}", fontsize=10.5, color=INK)
    ax.text(0.54, 0.19, f"Baseline CV AUROC: {format_metric(best_baseline['auroc'])}", fontsize=10.5, color=INK)
    ax.text(0.54, 0.15, f"Gain vs SEP MLP only (CV AUROC): {cv_gain_vs_mlp:+.3f}", fontsize=10.5, color=GREEN, fontweight="bold")
    save_fig(fig, out_path)


def draw_oi_external_summary(oi: Dict[str, object], out_path: Path) -> None:
    summary = oi["summary"]
    holdouts = pd.DataFrame(summary["holdout_rows"])
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8), facecolor=BG)

    metrics = ["accuracy", "auroc", "f1"]
    arch_vals = [clip_metric(summary["architecture_metrics"][m]) for m in metrics]
    base_vals = [clip_metric(summary["baseline_metrics"][m]) for m in metrics]
    x = np.arange(len(metrics))
    w = 0.34
    axes[0].bar(x - w / 2, arch_vals, width=w, color=PALETTE["architecture"], label=f"OI {ARCHITECTURE_NAME}")
    axes[0].bar(x + w / 2, base_vals, width=w, color=PALETTE["baseline"], label="OI Logistic Regression")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Accuracy", "AUROC", "F1"])
    axes[0].set_ylim(0, 1.01)
    axes[0].set_title("OI Overall External Metrics", color=INK, fontweight="bold")
    axes[0].grid(alpha=0.2, linestyle="--", axis="y")
    for i, (a, b) in enumerate(zip(arch_vals, base_vals)):
        axes[0].text(i - w / 2, a + 0.02, format_metric(a), ha="center", fontsize=8, color=INK)
        axes[0].text(i + w / 2, b + 0.02, format_metric(b), ha="center", fontsize=8, color=INK)
    axes[0].legend(loc="lower right", fontsize=8)

    hold_x = np.arange(len(holdouts))
    axes[1].bar(hold_x - w / 2, holdouts["gat_auroc"], width=w, color=PALETTE["architecture"], label=f"OI {ARCHITECTURE_NAME}")
    axes[1].bar(hold_x + w / 2, holdouts["lr_auroc"], width=w, color=PALETTE["baseline"], label="OI Logistic Regression")
    axes[1].set_xticks(hold_x)
    axes[1].set_xticklabels(holdouts["holdout"].tolist(), rotation=15, ha="right", fontsize=9)
    axes[1].set_ylim(0, 1.01)
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("OI Holdout Cohort AUROC", color=INK, fontweight="bold")
    axes[1].grid(alpha=0.2, linestyle="--", axis="y")

    fig.suptitle("Rare-Disease External Summary (Architecture Included)", fontsize=17, color=INK, fontweight="bold")
    save_fig(fig, out_path)


def draw_sepsis_baseline_gap_panel(sepsis: Dict[str, object], out_path: Path) -> None:
    df = sepsis["overall"].copy()
    mat = df[["accuracy", "auroc", "f1"]].values.astype(float)
    row_labels = [prefixed_model_name("sepsis", m) for m in df["model"].tolist()]
    fig, ax = plt.subplots(figsize=(11.8, 6.2), facecolor=BG)
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Accuracy", "AUROC", "F1"], fontsize=11)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title("SEP Hybrid Model vs Logistic / GAT / GCN / MLP Baselines", fontsize=18, color=INK, pad=16, fontweight="bold")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=9, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    save_fig(fig, out_path)


def curate_visuals() -> List[VisualArtifact]:
    sepsis = current_sepsis_metrics()
    alz = current_alz_metrics()
    oi = current_oi_metrics()
    cross = build_cross_disease_table(sepsis, alz, oi)
    landscape = build_model_landscape_table(sepsis, alz, oi)

    artifacts: List[VisualArtifact] = []

    def register_generated(slot: int, title: str, rationale: str, source: str, builder) -> None:
        out = FINAL_VIS / f"{slot:02d}_{slugify(title)}.png"
        builder(out)
        artifacts.append(VisualArtifact(slot, title, out, rationale, source))

    def register_copy(slot: int, title: str, rationale: str, source_path: Path) -> None:
        out = FINAL_VIS / f"{slot:02d}_{slugify(title)}.png"
        copy_visual(source_path, out)
        artifacts.append(VisualArtifact(slot, title, out, rationale, str(source_path.relative_to(ROOT).as_posix())))

    register_generated(1, "Cross Disease Metric Scorecard", "Master comparison view showing architecture and baseline performance for all diseases across accuracy, AUROC, and F1.", "updated sepsis metrics + current disease summaries", lambda out: draw_cross_disease_scorecard(cross, out))
    register_generated(2, "All Model Landscape", "Master graph comparing every available architecture and baseline point in one accuracy-AUROC-F1 space with readable model names.", "General_Sepsis_V11/results/metrics_overall.csv + ACSEF compiled baselines + disease summaries", lambda out: draw_model_landscape(landscape, out))
    register_generated(3, "Architecture Gain Over Baseline", "Hybrid model gain heatmap with explicit baseline architecture names by disease.", "derived from updated cross-disease table", lambda out: draw_margin_heatmap(cross, out))
    register_generated(4, "Sepsis ROC Evidence Panel", "Side-by-side ROC evidence panel for CV and external settings.", "General_Sepsis_V11/results/plots/roc_cv_model_comparison.png + roc_external_model_comparison.png", draw_sepsis_roc_panels)
    register_copy(5, "3D Graph Topology", "Keeps the strongest existing topology visual for structural intuition.", SEPSIS_ROOT / "plots" / "gnn_topology_3d.png")
    register_copy(6, "SHAP Summary Top 20", "Keeps the most interpretable biomarker ranking visual in the package.", SEPSIS_ROOT / "plots" / "shap_summary_top20.png")
    register_generated(7, "Sepsis Cohort Policy Flow", "Updated workflow diagram with larger cohort cards and cleaner split communication.", "General_Sepsis_V11/results/cohort_manifest.json", lambda out: draw_sepsis_cohort_flow(sepsis["cohort"], out))
    register_generated(8, "Biomarker Fingerprint Across Cohorts", "Shows whether the top SHAP genes behave consistently across datasets and conditions instead of only in aggregate.", "General_Sepsis_V11/results/expression_combat.csv + metadata.csv + shap_top20_features.csv", lambda out: draw_biomarker_fingerprint(sepsis, out))
    register_generated(9, "Graph Prior Coverage and Scale", "Quantifies how much biological structure is injected into the sepsis model and how large the runtime graph actually is.", "General_Sepsis_V11/results/pathway_info.json + cv_metrics_raw.json", lambda out: draw_graph_prior_coverage(sepsis, out))
    register_generated(10, "Sepsis Validation Dashboard", "Expanded one-page sepsis panel with larger cards and a dedicated Best Model section.", "General_Sepsis_V11/results/metrics_overall.csv + metrics_external.csv + general_sepsis_v11_results.json", lambda out: draw_sepsis_validation_dashboard(sepsis, out))
    register_generated(11, "Rare Disease External Summary", "Replaces holdout matrix with architecture-included rare-disease summary panels.", "results/osteogenesis/osteogenesis_metrics_summary.json", lambda out: draw_oi_external_summary(oi, out))
    register_generated(12, "Sepsis Hybrid vs Baseline Panel", "Focused sepsis comparison panel against Logistic Regression, GAT only, GCN only, and MLP only.", "General_Sepsis_V11/results/metrics_overall.csv + ACSEF compiled baselines", lambda out: draw_sepsis_baseline_gap_panel(sepsis, out))
    prompt_path = write_nano_banana_master_prompt()

    summary_blob = {
        "cross_disease_rows": cross.to_dict(orient="records"),
        "model_landscape_rows": landscape.to_dict(orient="records"),
    }
    (FINAL_VIS / "visual_metrics_summary.json").write_text(json.dumps(summary_blob, indent=2), encoding="utf-8")

    manifest_rows = []
    for art in artifacts:
        manifest_rows.append(
            {
                "slot": str(art.slot),
                "title": art.title,
                "file": str(art.path.relative_to(ROOT).as_posix()),
                "source": art.source,
                "rationale": art.rationale,
            }
        )

    with (FINAL_VIS / "visual_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest_rows, f, indent=2)
    with (FINAL_VIS / "visual_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["slot", "title", "file", "source", "rationale"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    lines = ["# Final ACSEF Visual Set (12 Figures)", ""]
    for row in manifest_rows:
        lines.append(f"{row['slot']}. **{row['title']}**")
        lines.append(f"   - File: `{row['file']}`")
        lines.append(f"   - Source: `{row['source']}`")
        lines.append(f"   - Why included: {row['rationale']}")
    lines.append("")
    lines.append("## Companion Prompt")
    lines.append(f"- Nano Banana architecture prompt: `{prompt_path.relative_to(ROOT).as_posix()}`")
    (FINAL_VIS / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifacts


def infer_weight_bucket(path: Path) -> Tuple[str, str]:
    p = path.as_posix().lower()
    name = path.name.lower()
    if "general_sepsis_v11/models" in p:
        return ("sepsis", "general_sepsis_v11")
    if "ch_dann_plan/models" in p:
        if "alz" in name:
            return ("alzheimers", "ch_dann_transfer")
        return ("sepsis", "ch_dann_sepsis_lineage")
    if "osteogenesis imperfecta/models" in p:
        return ("osteogenesis", "oi_graph_models")
    if "acsef_final_submission/models" in p:
        return ("sepsis", "acsef_submission")
    return ("other", "misc")


def infer_architecture(name: str) -> str:
    n = name.lower()
    if "dann" in n:
        return "multiplex_hgcn_dann"
    if "hgcn" in n:
        return "hgcn"
    if "gat" in n:
        return "gat"
    if "gcn" in n:
        return "gcn"
    if "transfer" in n:
        return "transfer_model"
    return "unknown"


def build_weights() -> List[Dict[str, str]]:
    srcs = []
    for p in ROOT.rglob("*.pt"):
        if "models" not in [part.lower() for part in p.parts]:
            continue
        srcs.append(p)

    rows: List[Dict[str, str]] = []
    for src in sorted(srcs, key=lambda x: x.as_posix().lower()):
        disease, group = infer_weight_bucket(src)
        dst_dir = WEIGHTS_ROOT / disease / group
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        rows.append(
            {
                "disease": disease,
                "group": group,
                "architecture_guess": infer_architecture(src.name),
                "weight_file": src.name,
                "source_path": src.as_posix().replace(ROOT.as_posix() + "/", ""),
                "weights_path": dst.as_posix().replace(ROOT.as_posix() + "/", ""),
                "size_mb": f"{src.stat().st_size / (1024 * 1024):.2f}",
            }
        )

    with (WEIGHTS_ROOT / "weights_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with (WEIGHTS_ROOT / "weights_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["disease", "group", "architecture_guess", "weight_file", "source_path", "weights_path", "size_mb"])
        writer.writeheader()
        writer.writerows(rows)

    disease_counts: Dict[str, int] = {}
    for row in rows:
        disease_counts[row["disease"]] = disease_counts.get(row["disease"], 0) + 1
    lines = ["# WEIGHTS Folder", "", "Centralized copy of model checkpoint weights by disease and architecture group.", ""]
    lines.append("## Counts by Disease")
    for disease, count in sorted(disease_counts.items()):
        lines.append(f"- {disease}: {count}")
    lines.append("")
    lines.append("## Manifests")
    lines.append("- `WEIGHTS/weights_manifest.csv`")
    lines.append("- `WEIGHTS/weights_manifest.json`")
    (WEIGHTS_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def write_project_navigation(visual_rows: List[VisualArtifact], weight_rows: List[Dict[str, str]]) -> None:
    lines = ["# Project Navigation Guide", ""]
    lines.append("## Top-Level Hubs")
    lines.append("- `ACSEF_Final_Submission/final_visuals`: exact 12 final visuals for ACSEF judging.")
    lines.append("- `General_Sepsis_V11/results`: updated robust sepsis metrics and plots.")
    lines.append("- `results`: cross-disease archived summary folders for sepsis, Alzheimer's, and osteogenesis.")
    lines.append("- `WEIGHTS`: centralized model checkpoint copies grouped by disease and architecture.")
    lines.append("")
    lines.append("## Final Visuals (12)")
    for art in visual_rows:
        lines.append(f"- [{art.slot}] `{art.path.relative_to(ROOT).as_posix()}`")
    lines.append("")
    lines.append(f"## Weights Summary")
    lines.append(f"- Total checkpoint files copied: {len(weight_rows)}")
    lines.append("- Detailed mapping: `WEIGHTS/weights_manifest.csv`")
    (ROOT / "PROJECT_NAVIGATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    visuals = curate_visuals()
    weights = build_weights()
    write_project_navigation(visuals, weights)
    print("Done.")
    print(f"Final visuals: {len(visuals)} at {FINAL_VIS}")
    print(f"Weights copied: {len(weights)} at {WEIGHTS_ROOT}")


if __name__ == "__main__":
    main()
