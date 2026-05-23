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
PUB = os.path.join(DOCS, "publication_package")

os.makedirs(PUB, exist_ok=True)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_interactive_biomarkers(biomarkers):
    top = biomarkers.head(20).copy()
    top["signed_score"] = top["attribution_score"].astype(float)
    top = top.sort_values("signed_score", ascending=True)

    rows = []
    for _, r in top.iterrows():
        rows.append(
            {
                "gene": r["gene"],
                "score": float(r["signed_score"]),
                "direction": r["direction"],
            }
        )

    js_data = json.dumps(rows)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <script src=\"https://cdn.plot.ly/plotly-2.27.0.min.js\"></script>
</head>
<body>
  <div id=\"chart\" style=\"width:100%;height:720px;\"></div>
  <script>
    const rows = {js_data};
    const y = rows.map(r => r.gene);
    const x = rows.map(r => r.score);
    const c = rows.map(r => r.score >= 0 ? "#d62728" : "#1f77b4");
    const text = rows.map(r => r.direction);

    const trace = {{
      x,
      y,
      type: "bar",
      orientation: "h",
      marker: {{color: c}},
      text,
      hovertemplate: "<b>%{{y}}</b><br>Attribution: %{{x:.4f}}<br>%{{text}}<extra></extra>",
    }};

    Plotly.newPlot("chart", [trace], {{
      title: "Interactive Top-20 Biomarker Attributions",
      xaxis: {{title: "Integrated Gradients Attribution"}},
      yaxis: {{title: "Gene"}},
      margin: {{l: 120, r: 30, t: 60, b: 60}},
    }});
  </script>
</body>
</html>"""

    out = os.path.join(FIGURES, "fig_interactive_top20_biomarkers.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)


def make_metric_radar(compiled):
    models = [
        "Pathway-HGCN-Classic",
        "Interaction-GCN-Baseline",
        "Attention-GAT-Baseline",
        "Multiplex-Hypergraph-DANN-MLP",
    ]

    row_map = {r["model_name"]: r for r in compiled["sepsis_benchmarks"]}
    labels = ["AUROC", "Accuracy", "F1"]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, polar=True)

    palette = {
        "Pathway-HGCN-Classic": "#4c78a8",
        "Interaction-GCN-Baseline": "#72b7b2",
        "Attention-GAT-Baseline": "#54a24b",
        "Multiplex-Hypergraph-DANN-MLP": "#e45756",
    }

    for m in models:
        r = row_map.get(m, {})
        vals = [
            float(r.get("auroc_mean", np.nan)),
            float(r.get("accuracy_mean", np.nan)),
            float(r.get("f1_mean", np.nan)),
        ]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=m, color=palette[m])
        ax.fill(angles, vals, alpha=0.08, color=palette[m])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("Model Metric Radar (Sepsis)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig_model_metric_radar.png"), dpi=220)
    plt.close()


def write_publication_tables(compiled):
    rows = []
    for r in compiled["sepsis_benchmarks"]:
        rows.append(
            {
                "model_name": r["model_name"],
                "auroc": r["auroc_mean"],
                "accuracy": r["accuracy_mean"],
                "f1": r["f1_mean"],
                "protocol": r["evaluation_protocol"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(PUB, "table_main_metrics.csv"), index=False)

    ext = compiled["external_validation_gse26440"]
    ext_df = pd.DataFrame(
        [
            {
                "dataset": "GSE26440",
                "samples": ext["n_samples"],
                "accuracy": ext["accuracy"],
                "auroc": ext["auroc"],
                "f1": ext["f1"],
                "precision": ext["precision"],
                "recall": ext["recall"],
            }
        ]
    )
    ext_df.to_csv(os.path.join(PUB, "table_external_validation.csv"), index=False)


def write_manifest():
    now = datetime.now().strftime("%Y-%m-%d")
    lines = []
    lines.append("# Publication Package Manifest")
    lines.append("")
    lines.append(f"Generated on: {now}")
    lines.append("")
    lines.append("## Figures")
    lines.append("- fig_roc_comparisons.png: AUROC comparison across HGCN, GCN, GAT, and final model")
    lines.append("- fig_architecture_flowchart.png: architecture schematic")
    lines.append("- fig_biomarker_attributions.png: top biomarker IG attribution bar chart")
    lines.append("- fig_external_validation_gse26440.png: external cohort performance")
    lines.append("- fig_osteogenesis_scaling_summary.png: rare-disease scaling summary")
    lines.append("- fig_model_metric_radar.png: compact multi-metric overview")
    lines.append("- fig_interactive_model_metrics.html: interactive grouped bars")
    lines.append("- fig_interactive_top20_biomarkers.html: interactive biomarker attribution view")
    lines.append("")
    lines.append("## Tables")
    lines.append("- table_main_metrics.csv")
    lines.append("- table_external_validation.csv")
    lines.append("")
    lines.append("## Notes")
    lines.append("All graphics and tables were generated locally from repository outputs; no external web rendering pipeline was used.")

    with open(os.path.join(PUB, "figure_manifest.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    compiled = read_json(os.path.join(RESULTS, "compiled_model_metrics.json"))
    biomarkers = pd.read_csv(os.path.join(RESULTS, "top_100_biomarkers.csv"))

    make_interactive_biomarkers(biomarkers)
    make_metric_radar(compiled)
    write_publication_tables(compiled)
    write_manifest()


if __name__ == "__main__":
    main()
