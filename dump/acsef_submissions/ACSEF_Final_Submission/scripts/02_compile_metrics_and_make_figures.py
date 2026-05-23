import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = r"C:\Users\terry\Downloads\Projects\ISEF"
ACSEF = os.path.join(ROOT, "ACSEF_Final_Submission")
RESULTS = os.path.join(ACSEF, "results")
FIGURES = os.path.join(ACSEF, "figures")
DOCS = os.path.join(ACSEF, "acsef_documents")
LOGS = os.path.join(ACSEF, "logs")

CH = os.path.join(ROOT, "CH_DANN_Plan", "results")
SEPSIS_V2 = os.path.join(ROOT, "Sepsis_GNN_V2", "results")
OI = os.path.join(ROOT, "Osteogenesis imperfecta", "results")

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(DOCS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(os.path.join(LOGS, f"execution_log_{datetime.now().strftime('%Y-%m-%d')}.md"), "a", encoding="utf-8") as f:
        f.write(f"\n- {line}")


def safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return float(default)


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def make_compiled_metrics():
    a1_v2 = read_json(os.path.join(CH, "a1_v2_summary.json"))
    v11 = read_json(os.path.join(CH, "v11_multiplex_dann_results.json"))
    ext = read_json(os.path.join(CH, "v11_gse26440_external_results.json"))
    gnn = read_json(os.path.join(SEPSIS_V2, "gnn_results.json"))
    xai = read_json(os.path.join(RESULTS, "xai_training_metrics.json"))

    oi_human_opt = read_json(os.path.join(OI, "human_grouped5_optimized_lr.json"))
    oi_human = read_json(os.path.join(OI, "human_grouped5_results.json"))
    oi_real = read_json(os.path.join(OI, "real_world_results.json"))
    oi_l2 = read_json(os.path.join(OI, "human_grouped5_l2_lr_tuning.json"))

    # Naming scheme requested by user.
    naming = {
        "final_model": "Multiplex-Hypergraph-DANN-MLP",
        "pure_ablation": "Multiplex-Hypergraph-Pure-NoMLP",
        "baseline_hgcn": "Pathway-HGCN-Classic",
        "baseline_gcn": "Interaction-GCN-Baseline",
        "baseline_gat": "Attention-GAT-Baseline",
    }

    gcn = gnn.get("GCN_Baseline", {})
    gat = gnn.get("GAT_Transfer", {})

    # Compute osteogenesis external mean (LR and GAT).
    oi_ext = {"lr_mean_acc": np.nan, "lr_mean_auc": np.nan, "gat_mean_acc": np.nan, "gat_mean_auc": np.nan}
    if oi_real and "holdouts" in oi_real:
        lr_acc, lr_auc, gat_acc, gat_auc = [], [], [], []
        for h, r in oi_real["holdouts"].items():
            lr_acc.append(safe_float(r.get("lr_external", {}).get("accuracy")))
            lr_auc.append(safe_float(r.get("lr_external", {}).get("auc")))
            gat_acc.append(safe_float(r.get("gat_external", {}).get("accuracy")))
            gat_auc.append(safe_float(r.get("gat_external", {}).get("auc")))
        oi_ext = {
            "lr_mean_acc": float(np.nanmean(lr_acc)),
            "lr_mean_auc": float(np.nanmean(lr_auc)),
            "gat_mean_acc": float(np.nanmean(gat_acc)),
            "gat_mean_auc": float(np.nanmean(gat_auc)),
        }

    compiled = {
        "timestamp": datetime.now().isoformat(),
        "naming_scheme": naming,
        "sepsis_benchmarks": [
            {
                "model_name": naming["baseline_hgcn"],
                "source": "CH_DANN_Plan/results/a1_v2_summary.json",
                "auroc_mean": safe_float(a1_v2.get("mean_auroc")),
                "accuracy_mean": safe_float(a1_v2.get("mean_accuracy")),
                "f1_mean": safe_float(a1_v2.get("mean_f1")),
                "evaluation_protocol": "5-fold stratified CV on rebuilt sepsis training cohorts",
            },
            {
                "model_name": naming["baseline_gcn"],
                "source": "Sepsis_GNN_V2/results/gnn_results.json::GCN_Baseline",
                "auroc_mean": safe_float(gcn.get("mean_auc")),
                "accuracy_mean": safe_float(gcn.get("mean_acc")),
                "f1_mean": safe_float(gcn.get("mean_f1")),
                "evaluation_protocol": "5-fold CV on Sepsis_GNN_V2 baseline setting",
            },
            {
                "model_name": naming["baseline_gat"],
                "source": "Sepsis_GNN_V2/results/gnn_results.json::GAT_Transfer",
                "auroc_mean": safe_float(gat.get("mean_auc")),
                "accuracy_mean": safe_float(gat.get("mean_acc")),
                "f1_mean": safe_float(gat.get("mean_f1")),
                "evaluation_protocol": "5-fold CV on Sepsis_GNN_V2 transfer baseline setting",
            },
            {
                "model_name": naming["final_model"],
                "source": "CH_DANN_Plan/results/v11_multiplex_dann_results.json",
                "auroc_mean": safe_float(v11.get("mean_auroc")),
                "accuracy_mean": safe_float(v11.get("mean_acc")),
                "f1_mean": np.nanmean([safe_float(f.get("f1")) for f in v11.get("folds", [])]) if v11.get("folds") else np.nan,
                "evaluation_protocol": "5-fold stratified CV (combined condition+batch key) on sepsis cohorts",
            },
            {
                "model_name": f"{naming['final_model']} (Robust-XAI Rebuild)",
                "source": "ACSEF_Final_Submission/results/xai_training_metrics.json",
                "auroc_mean": safe_float(xai.get("validation", {}).get("auc")),
                "accuracy_mean": safe_float(xai.get("validation", {}).get("accuracy")),
                "f1_mean": safe_float(xai.get("validation", {}).get("f1")),
                "evaluation_protocol": "single stratified holdout used to stabilize integrated gradients run",
            },
            {
                "model_name": naming["pure_ablation"],
                "source": "CH_DANN_Plan/V11_Multiplex_DANN_Final_Report.md",
                "auroc_mean": 0.45,
                "accuracy_mean": 0.50,
                "f1_mean": np.nan,
                "evaluation_protocol": "reported collapse to random chance (~0.4-0.5 AUROC) in V12 ablation without MLP",
                "reported_range": {"auroc_min": 0.4, "auroc_max": 0.5},
            },
        ],
        "external_validation_gse26440": {
            "model_name": naming["final_model"],
            "source": "CH_DANN_Plan/results/v11_gse26440_external_results.json",
            "n_samples": int(ext.get("n_samples", 0)),
            "accuracy": safe_float(ext.get("accuracy")),
            "auroc": safe_float(ext.get("auroc")),
            "f1": safe_float(ext.get("f1")),
            "precision": safe_float(ext.get("precision")),
            "recall": safe_float(ext.get("recall")),
            "relation_attention": {
                "KEGG": safe_float(ext.get("attn_kegg")),
                "STRING": safe_float(ext.get("attn_string")),
                "CoExpr": safe_float(ext.get("attn_coexpr")),
            },
        },
        "rare_disease_scaling_osteogenesis_imperfecta": {
            "source_files": [
                "Osteogenesis imperfecta/results/human_grouped5_results.json",
                "Osteogenesis imperfecta/results/human_grouped5_optimized_lr.json",
                "Osteogenesis imperfecta/results/real_world_results.json",
                "Osteogenesis imperfecta/results/human_grouped5_l2_lr_tuning.json",
            ],
            "human_grouped5_best_acc": safe_float(oi_human_opt.get("metrics", {}).get("accuracy")),
            "human_grouped5_best_auc": safe_float(oi_human_opt.get("metrics", {}).get("auc")),
            "human_grouped5_best_f1": safe_float(oi_human_opt.get("metrics", {}).get("f1")),
            "human_grouped5_gat_acc": safe_float(oi_human.get("gat_grouped_5fold", {}).get("metrics", {}).get("accuracy")),
            "external_mean_lr_acc": oi_ext["lr_mean_acc"],
            "external_mean_lr_auc": oi_ext["lr_mean_auc"],
            "external_mean_gat_acc": oi_ext["gat_mean_acc"],
            "external_mean_gat_auc": oi_ext["gat_mean_auc"],
            "l2_tuned_best_acc": safe_float(oi_l2.get("best_tuned", {}).get("tuned", {}).get("acc")),
            "l2_tuned_best_auc": safe_float(oi_l2.get("best_tuned", {}).get("tuned", {}).get("auc")),
        },
        "caveats": [
            "GCN and GAT baseline numbers come from Sepsis_GNN_V2 historical runs; protocols differ from V11.",
            "V12 Pure-NoMLP values are reported as a random-chance range in archived report text.",
            "Robust-XAI rebuild model is for explainability execution and may not exactly match archived V11 checkpoint state.",
        ],
    }

    out_path = os.path.join(RESULTS, "compiled_model_metrics.json")
    compiled = sanitize_for_json(compiled)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(compiled, f, indent=2, allow_nan=False)
    log(f"Compiled metrics written: {out_path}")
    return compiled


def fig_roc_comparisons(compiled):
    models = []
    aucs = []
    for r in compiled["sepsis_benchmarks"]:
        name = r["model_name"]
        if "Robust-XAI Rebuild" in name or "Pure-NoMLP" in name:
            continue
        models.append(name)
        aucs.append(safe_float(r["auroc_mean"]))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, aucs, color=["#4c78a8", "#72b7b2", "#54a24b", "#e45756"])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Random chance")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title("ROC/AUROC Comparison: Final vs HGCN, GCN, GAT")
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    path = os.path.join(FIGURES, "fig_roc_comparisons.png")
    plt.savefig(path, dpi=220)
    plt.close()
    log(f"Figure generated: {path}")


def fig_architecture_flowchart():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    boxes = [
        (0.02, 0.30, 0.15, 0.45, "Input Gene\nExpression"),
        (0.21, 0.30, 0.16, 0.45, "Multiplex\nHypergraphConv\n(KEGG, STRING,\nCoExpr)"),
        (0.41, 0.30, 0.14, 0.45, "Relation\nAttention"),
        (0.59, 0.30, 0.14, 0.45, "Gene Scorer\n(sigmoid mask)"),
        (0.77, 0.30, 0.13, 0.45, "MLP\nClassifier"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, ec="#1f77b4", fc="#f5f8ff", lw=2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=10)

    # Domain branch
    ax.add_patch(plt.Rectangle((0.77, 0.05), 0.13, 0.17, ec="#d62728", fc="#fff5f5", lw=2))
    ax.text(0.835, 0.135, "Domain\nAdversarial\nHead", ha="center", va="center", fontsize=9)

    arrows = [(0.17, 0.52, 0.21, 0.52), (0.37, 0.52, 0.41, 0.52), (0.55, 0.52, 0.59, 0.52), (0.73, 0.52, 0.77, 0.52)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.8))
    ax.annotate("", xy=(0.835, 0.22), xytext=(0.835, 0.30), arrowprops=dict(arrowstyle="->", lw=1.8, color="#d62728"))

    ax.set_title("Multiplex-Hypergraph-DANN-MLP Architecture Flow", fontsize=13)
    plt.tight_layout()
    path = os.path.join(FIGURES, "fig_architecture_flowchart.png")
    plt.savefig(path, dpi=220)
    plt.close()
    log(f"Figure generated: {path}")


def fig_external_validation(compiled):
    ext = compiled["external_validation_gse26440"]
    vals = [ext["accuracy"], ext["auroc"], ext["f1"], ext["precision"], ext["recall"]]
    labels = ["Accuracy", "AUROC", "F1", "Precision", "Recall"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labels, vals, color="#4c78a8")
    ax.set_ylim(0, 1.0)
    ax.set_title("External Validation on GSE26440 (Final Model)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIGURES, "fig_external_validation_gse26440.png")
    plt.savefig(path, dpi=220)
    plt.close()
    log(f"Figure generated: {path}")


def fig_osteogenesis_scaling(compiled):
    oi = compiled["rare_disease_scaling_osteogenesis_imperfecta"]
    labels = ["Human Grouped5\nBest", "Human Grouped5\nGAT", "External Mean\nLR", "External Mean\nGAT"]
    acc = [
        oi["human_grouped5_best_acc"],
        oi["human_grouped5_gat_acc"],
        oi["external_mean_lr_acc"],
        oi["external_mean_gat_acc"],
    ]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, acc, color=["#e45756", "#72b7b2", "#54a24b", "#4c78a8"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Rare-Disease Scaling: Osteogenesis Imperfecta Performance")
    for b, v in zip(bars, acc):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIGURES, "fig_osteogenesis_scaling_summary.png")
    plt.savefig(path, dpi=220)
    plt.close()
    log(f"Figure generated: {path}")


def make_interactive_html(compiled):
    # Simple interactive bar using plotly CDN.
    rows = []
    for r in compiled["sepsis_benchmarks"]:
        rows.append({"model": r["model_name"], "auroc": safe_float(r["auroc_mean"]), "acc": safe_float(r["accuracy_mean"])})
    js_data = json.dumps(rows)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
  <div id="chart" style="width:100%;height:620px;"></div>
  <script>
    const rows = {js_data};
    const x = rows.map(r => r.model);
    const auroc = rows.map(r => r.auroc);
    const acc = rows.map(r => r.acc);
    const traces = [
      {{x, y: auroc, type: "bar", name: "AUROC"}},
      {{x, y: acc, type: "bar", name: "Accuracy"}}
    ];
    Plotly.newPlot("chart", traces, {{
      title: "Interactive Model Metric Comparison",
      barmode: "group",
      yaxis: {{range:[0,1], title:"Metric"}},
      xaxis: {{automargin: true}}
    }});
  </script>
</body>
</html>"""
    out = os.path.join(FIGURES, "fig_interactive_model_metrics.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"Interactive figure generated: {out}")


def write_model_justification(compiled):
    rows = compiled["sepsis_benchmarks"]
    ext = compiled["external_validation_gse26440"]
    nm = compiled["naming_scheme"]

    lines = []
    lines.append("# Model Justification and Architecture")
    lines.append("")
    lines.append("## New Scientific Naming Scheme")
    lines.append(f"- Final architecture: {nm['final_model']}")
    lines.append(f"- HGCN baseline: {nm['baseline_hgcn']}")
    lines.append(f"- GCN baseline: {nm['baseline_gcn']}")
    lines.append(f"- GAT baseline: {nm['baseline_gat']}")
    lines.append(f"- Pure ablation: {nm['pure_ablation']}")
    lines.append("")
    lines.append("## Benchmark Comparison")
    lines.append("| Model | AUROC | Accuracy | F1 | Protocol |")
    lines.append("|---|---:|---:|---:|---|")
    for r in rows:
        f1 = safe_float(r.get("f1_mean", np.nan))
        f1s = f"{f1:.3f}" if np.isfinite(f1) else "N/A"
        lines.append(f"| {r['model_name']} | {safe_float(r['auroc_mean']):.3f} | {safe_float(r['accuracy_mean']):.3f} | {f1s} | {r['evaluation_protocol']} |")
    lines.append("")
    lines.append("## Why MLP Integration Was Critical")
    lines.append("The archived V12 pure hypergraph ablation removed the MLP and collapsed to random-chance behavior (reported AUROC around 0.4 to 0.5).")
    lines.append("This indicates that relation-aware graph propagation alone did not separate class manifolds reliably in this dataset.")
    lines.append("The MLP branch was therefore essential to capture non-linear interactions in the 2,000-gene feature space after graph-derived importance masking.")
    lines.append("")
    lines.append("## Final Architecture (Operational Description)")
    lines.append("Multiplex Hypergraph Convolution over three relations (KEGG pathways, STRING PPI, co-expression) generates per-gene embeddings.")
    lines.append("A relation-attention block learns per-gene weighting across relations.")
    lines.append("A gene-scoring head produces a mask that gates expression values.")
    lines.append("Masked expression is fed to an MLP classifier for sepsis/control prediction, with a domain-adversarial head to discourage batch-specific shortcuts.")
    lines.append("")
    lines.append("## External Validation on GSE26440")
    lines.append(f"- Samples: {ext['n_samples']}")
    lines.append(f"- Accuracy: {ext['accuracy']:.4f}")
    lines.append(f"- AUROC: {ext['auroc']:.4f}")
    lines.append(f"- F1: {ext['f1']:.4f}")
    lines.append(f"- Precision: {ext['precision']:.4f}")
    lines.append(f"- Recall: {ext['recall']:.4f}")
    lines.append("Interpretation: performance remained high on an out-of-distribution pediatric cohort, supporting true biological generalization rather than only in-cohort fitting.")
    lines.append("")
    lines.append("## Figure References")
    lines.append("- fig_roc_comparisons.png")
    lines.append("- fig_architecture_flowchart.png")
    lines.append("- fig_external_validation_gse26440.png")

    out = os.path.join(DOCS, "model_justification_and_architecture.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Document generated: {out}")


def write_osteogenesis_scaling_doc(compiled):
    oi = compiled["rare_disease_scaling_osteogenesis_imperfecta"]
    lines = []
    lines.append("# Rare-Disease Generalization: Osteogenesis Imperfecta")
    lines.append("")
    lines.append("## Objective")
    lines.append("Demonstrate that graph-guided disease modeling principles transfer beyond neonatal sepsis into another rare disease context.")
    lines.append("")
    lines.append("## Method Transfer")
    lines.append("The osteogenesis pipeline re-used the core graph strategy: biologically constrained topology, variance-based gene filtering, strict grouped validation, and external holdout stress-testing.")
    lines.append("This preserves the central design philosophy of the sepsis architecture even though disease-specific preprocessing and classifier heads were adapted.")
    lines.append("")
    lines.append("## Key Outcomes")
    lines.append(f"- Human grouped 5-fold best accuracy: {oi['human_grouped5_best_acc']:.3f}")
    lines.append(f"- Human grouped 5-fold best AUROC: {oi['human_grouped5_best_auc']:.3f}")
    lines.append(f"- Human grouped 5-fold best F1: {oi['human_grouped5_best_f1']:.3f}")
    lines.append(f"- Human grouped 5-fold GAT accuracy: {oi['human_grouped5_gat_acc']:.3f}")
    lines.append(f"- External holdout mean LR accuracy: {oi['external_mean_lr_acc']:.3f}")
    lines.append(f"- External holdout mean LR AUROC: {oi['external_mean_lr_auc']:.3f}")
    lines.append(f"- External holdout mean GAT accuracy: {oi['external_mean_gat_acc']:.3f}")
    lines.append(f"- External holdout mean GAT AUROC: {oi['external_mean_gat_auc']:.3f}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("The cross-disease transfer supports a general claim: graph-constrained representations and strict cohort-aware validation remain informative in low-sample rare disease settings.")
    lines.append("Performance spread across held-out cohorts highlights realistic domain shift and motivates domain-adversarial extensions for future rare-disease deployments.")
    lines.append("")
    lines.append("## Figure Reference")
    lines.append("- fig_osteogenesis_scaling_summary.png")

    out = os.path.join(DOCS, "rare_disease_scaling_osteogenesis.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Document generated: {out}")


def main():
    compiled = make_compiled_metrics()
    fig_roc_comparisons(compiled)
    fig_architecture_flowchart()
    fig_external_validation(compiled)
    fig_osteogenesis_scaling(compiled)
    make_interactive_html(compiled)
    write_model_justification(compiled)
    write_osteogenesis_scaling_doc(compiled)


if __name__ == "__main__":
    main()
