#!/usr/bin/env python3
"""
Build publication assets for the robust General_Sepsis_V11 overhaul:
- sync regenerated figures into ACSEF figure folders
- create print poster (PNG/SVG/PDF)
- emit figure manifest + claim traceability
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(r"C:\Users\terry\Downloads\Projects\ISEF")
GSV11 = ROOT / "General_Sepsis_V11"
ACSEF = ROOT / "ACSEF_Final_Submission"
SRC_PLOTS = GSV11 / "results" / "plots"
SRC_RESULTS = GSV11 / "results"
FIG = ACSEF / "figures"
IMG = ACSEF / "images"
PUB = ACSEF / "acsef_documents" / "publication_package"

for p in [FIG, IMG, PUB]:
    p.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sync_figures() -> Dict[str, Path]:
    mapping = {
        "fig_roc_comparisons.png": SRC_PLOTS / "roc_cv_model_comparison.png",
        "fig_external_validation_gse26440.png": SRC_PLOTS / "roc_external_model_comparison.png",
        "fig_model_metric_radar.png": SRC_PLOTS / "metrics_heatmap_cv.png",
        "fig_relation_attention_heatmap.png": SRC_PLOTS / "relation_attention_heatmap.png",
        "fig_general_sepsis_v11_performance_panels.png": SRC_PLOTS / "metrics_heatmap_external.png",
        "fig_general_sepsis_v11_data_biomarkers.png": SRC_PLOTS / "shap_summary_top20.png",
        "fig_general_sepsis_v11_architecture.png": SRC_PLOTS / "gnn_topology_3d.png",
        "fig_biomarker_attributions.png": SRC_PLOTS / "shap_heatmap_top20.png",
    }
    synced: Dict[str, Path] = {}
    for dst_name, src_path in mapping.items():
        if not src_path.exists():
            continue
        fig_dst = FIG / dst_name
        img_dst = IMG / dst_name
        shutil.copy2(src_path, fig_dst)
        shutil.copy2(src_path, img_dst)
        synced[dst_name] = fig_dst
    return synced


def panel(fig, canvas, x: float, y: float, w: float, h: float, title: str, image_path: Path) -> None:
    canvas.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=2, ec="#244b6b"))
    canvas.text(x + 0.008, y + h - 0.012, title, fontsize=11.5, fontweight="bold", color="#173f5f", va="top")
    ax = fig.add_axes([x + 0.008, y + 0.008, w - 0.016, h - 0.04])
    ax.axis("off")
    if image_path.exists():
        ax.imshow(plt.imread(str(image_path)))
        ax.set_aspect("auto")
    else:
        ax.text(0.5, 0.5, f"Missing image:\n{image_path.name}", ha="center", va="center")


def build_poster(figures: Dict[str, Path], summary: Dict[str, object]) -> List[Path]:
    fig = plt.figure(figsize=(48, 36), dpi=150)
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.axis("off")
    canvas.add_patch(plt.Rectangle((0, 0), 1, 1, color="#f4f9ff"))
    canvas.add_patch(plt.Rectangle((0, 0.90), 1, 0.10, color="#123B5D"))
    canvas.text(
        0.02,
        0.955,
        "Robust General_Sepsis_V11: Cohort-Aware Validation, 3D Topology, and SHAP Explainability",
        color="white",
        fontsize=29,
        fontweight="bold",
        va="center",
    )
    canvas.text(
        0.02,
        0.918,
        f"Generated {datetime.now().strftime('%Y-%m-%d')} | Validation mode: {summary.get('cv_mode', 'unknown')}",
        color="#d7e6f5",
        fontsize=14,
        va="center",
    )

    metrics_overall = summary.get("metrics_overall", {})
    metrics_external = summary.get("metrics_external", {})
    text_left = (
        "Objective:\n"
        "Remove optimistic validation artifacts and re-evaluate hybrid GNN architecture under robust cohort-aware splits.\n\n"
        "Methods:\n"
        "Fold-internal MAD feature selection, dataset-aware CV, threshold calibration from OOF, "
        "model-by-model comparisons, SHAP explainability, and 3D graph topology rendering."
    )
    text_right = (
        "Key Metrics:\n"
        f"Hybrid CV AUROC: {metrics_overall.get('hybrid_v11_auroc', 'NA')}\n"
        f"Hybrid CV Accuracy: {metrics_overall.get('hybrid_v11_accuracy', 'NA')}\n"
        f"Hybrid External AUROC: {metrics_external.get('hybrid_v11_auroc', 'NA')}\n"
        f"Hybrid External Accuracy: {metrics_external.get('hybrid_v11_accuracy', 'NA')}\n\n"
        "Conclusion:\n"
        "Comparisons now use robust validation and interpretability-ready artifacts for defensible claims."
    )
    canvas.text(0.02, 0.87, text_left, fontsize=12.5, va="top")
    canvas.text(0.58, 0.87, text_right, fontsize=12.5, va="top")

    panels = [
        ("A. CV ROC (Hybrid vs Baselines)", "fig_roc_comparisons.png", [0.02, 0.58, 0.30, 0.24]),
        ("B. External ROC Comparison", "fig_external_validation_gse26440.png", [0.34, 0.58, 0.30, 0.24]),
        ("C. CV Metrics Heatmap", "fig_model_metric_radar.png", [0.66, 0.58, 0.32, 0.24]),
        ("D. External Metrics Heatmap", "fig_general_sepsis_v11_performance_panels.png", [0.02, 0.26, 0.30, 0.28]),
        ("E. 3D GNN Topology (Nodes/Edges)", "fig_general_sepsis_v11_architecture.png", [0.34, 0.26, 0.30, 0.28]),
        ("F. SHAP Summary + Heatmap", "fig_biomarker_attributions.png", [0.66, 0.26, 0.32, 0.28]),
    ]
    for title, key, rect in panels:
        panel(fig, canvas, rect[0], rect[1], rect[2], rect[3], title, figures.get(key, FIG / key))

    canvas.text(
        0.02,
        0.04,
        "Evidence sources: General_Sepsis_V11/results/*.json and General_Sepsis_V11/results/plots/*.png",
        fontsize=10.5,
        color="#2c3e50",
    )

    out_png = PUB / "general_sepsis_v11_poster.png"
    out_svg = PUB / "general_sepsis_v11_poster.svg"
    out_pdf = PUB / "general_sepsis_v11_poster.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [out_png, out_svg, out_pdf]


def extract_summary() -> Dict[str, object]:
    overall = pd_read(SRC_RESULTS / "metrics_overall.csv")
    external = pd_read(SRC_RESULTS / "metrics_external.csv")
    cv = load_json(SRC_RESULTS / "cv_metrics_raw.json")
    summary: Dict[str, object] = {"cv_mode": cv.get("cv_mode", "unknown")}
    if not overall.empty:
        hy = overall.loc[overall["model"] == "hybrid_v11"]
        if not hy.empty:
            summary["metrics_overall"] = {
                "hybrid_v11_auroc": f"{float(hy.iloc[0]['auroc']):.3f}",
                "hybrid_v11_accuracy": f"{float(hy.iloc[0]['accuracy']):.3f}",
            }
    if not external.empty:
        hy = external.loc[external["model"] == "hybrid_v11"]
        if not hy.empty:
            summary["metrics_external"] = {
                "hybrid_v11_auroc": f"{float(hy.iloc[0]['auroc']):.3f}",
                "hybrid_v11_accuracy": f"{float(hy.iloc[0]['accuracy']):.3f}",
            }
    return summary


def pd_read(path: Path):
    try:
        import pandas as pd

        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return __import__("pandas").DataFrame()


def write_figure_manifest(figures: Dict[str, Path], posters: List[Path]) -> None:
    lines = []
    lines.append("# Robust General_Sepsis_V11 Figure Manifest")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("| Artifact | Description | Source |")
    lines.append("|---|---|---|")
    for name, path in sorted(figures.items()):
        lines.append(f"| {name} | Robust comparison / explainability figure | {path} |")
    for p in posters:
        lines.append(f"| {p.name} | Research poster export | {p} |")
    (PUB / "general_sepsis_v11_figure_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def write_claim_traceability(summary: Dict[str, object]) -> None:
    rows = [
        {
            "claim_id": "C1",
            "claim": "Validation uses cohort-aware robust cross-validation.",
            "artifact_source": "General_Sepsis_V11/results/cv_metrics_raw.json",
            "citation": "",
        },
        {
            "claim_id": "C2",
            "claim": "Model comparisons include baselines in both CV and external views.",
            "artifact_source": "General_Sepsis_V11/results/plots/roc_cv_model_comparison.png; roc_external_model_comparison.png",
            "citation": "",
        },
        {
            "claim_id": "C3",
            "claim": "Explainability includes SHAP summary and SHAP heatmap.",
            "artifact_source": "General_Sepsis_V11/results/plots/shap_summary_top20.png; shap_heatmap_top20.png",
            "citation": "",
        },
        {
            "claim_id": "C4",
            "claim": "3D GNN topology with nodes and edges is included.",
            "artifact_source": "General_Sepsis_V11/results/plots/gnn_topology_3d.png",
            "citation": "",
        },
        {
            "claim_id": "C5",
            "claim": f"Hybrid CV AUROC: {summary.get('metrics_overall', {}).get('hybrid_v11_auroc', 'NA')}",
            "artifact_source": "General_Sepsis_V11/results/metrics_overall.csv",
            "citation": "",
        },
        {
            "claim_id": "C6",
            "claim": f"Hybrid External AUROC: {summary.get('metrics_external', {}).get('hybrid_v11_auroc', 'NA')}",
            "artifact_source": "General_Sepsis_V11/results/metrics_external.csv",
            "citation": "",
        },
    ]
    out = PUB / "general_sepsis_v11_claim_traceability.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["claim_id", "claim", "artifact_source", "citation"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    synced = sync_figures()
    summary = extract_summary()
    posters = build_poster(synced, summary)
    write_figure_manifest(synced, posters)
    write_claim_traceability(summary)
    print("Synced figures:")
    for name, path in sorted(synced.items()):
        print(f" - {name}: {path}")
    print("Poster outputs:")
    for p in posters:
        print(f" - {p}")
    print(f"Manifest: {PUB / 'general_sepsis_v11_figure_manifest.md'}")
    print(f"Traceability: {PUB / 'general_sepsis_v11_claim_traceability.csv'}")


if __name__ == "__main__":
    main()
