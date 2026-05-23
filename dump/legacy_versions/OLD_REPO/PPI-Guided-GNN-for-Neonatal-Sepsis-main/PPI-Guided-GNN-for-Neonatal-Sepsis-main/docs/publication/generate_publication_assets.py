import os
import json
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = r"C:\Users\terry\Downloads\Projects\ISEF"
OI = os.path.join(ROOT, "Osteogenesis imperfecta")
RES = os.path.join(OI, "results")
FIG = os.path.join(OI, "figures")
DOCS = os.path.join(ROOT, "docs", "publication")
VIS = os.path.join(DOCS, "visuals")
INTER = os.path.join(DOCS, "interactive")
TABLES = os.path.join(DOCS, "tables")
ARCH = os.path.join(DOCS, "architecture")

for d in [VIS, INTER, TABLES, ARCH]:
    os.makedirs(d, exist_ok=True)


def load_json(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


baseline = load_json("baseline_results.json")
gnn = load_json("gnn_results.json")
real = load_json("real_world_results.json")
human = load_json("human_grouped5_results.json")
opt = load_json("human_grouped5_optimized_lr.json")
l2 = load_json("human_grouped5_l2_lr_tuning.json")

rows = []

if baseline:
    for model in ["LogisticRegression", "RandomForest"]:
        if model in baseline:
            rows.append({
                "Evaluation": "Baseline CV (2 cohorts)",
                "Model": model,
                "Accuracy": baseline[model]["mean_acc"],
                "AUC": baseline[model]["mean_auc"],
                "F1": baseline[model]["mean_f1"],
                "Notes": "Combined CV on early cohorts, optimistic",
            })

if gnn:
    for model in ["GAT_v2", "GCN"]:
        if model in gnn:
            rows.append({
                "Evaluation": "Baseline CV (2 cohorts)",
                "Model": model,
                "Accuracy": gnn[model]["mean_acc"],
                "AUC": gnn[model]["mean_auc"],
                "F1": gnn[model]["mean_f1"],
                "Notes": "Combined CV on early cohorts, optimistic",
            })

if real and "holdouts" in real:
    holdouts = real["holdouts"]
    gat_acc = [holdouts[h]["gat_external"]["accuracy"] for h in holdouts]
    gat_auc = [holdouts[h]["gat_external"]["auc"] for h in holdouts]
    gat_f1 = [holdouts[h]["gat_external"]["f1"] for h in holdouts]
    lr_acc = [holdouts[h]["lr_external"]["accuracy"] for h in holdouts]
    lr_auc = [holdouts[h]["lr_external"]["auc"] for h in holdouts]
    lr_f1 = [holdouts[h]["lr_external"]["f1"] for h in holdouts]

    rows.append({
        "Evaluation": "External holdout mean (human multicohort)",
        "Model": "GAT_v2",
        "Accuracy": float(np.mean(gat_acc)),
        "AUC": float(np.mean(gat_auc)),
        "F1": float(np.mean(gat_f1)),
        "Notes": "Leave one dataset out external test",
    })
    rows.append({
        "Evaluation": "External holdout mean (human multicohort)",
        "Model": "LogisticRegression",
        "Accuracy": float(np.mean(lr_acc)),
        "AUC": float(np.mean(lr_auc)),
        "F1": float(np.mean(lr_f1)),
        "Notes": "Leave one dataset out external test",
    })

if human:
    bt = human.get("best_tabular")
    if bt:
        rows.append({
            "Evaluation": "Human grouped 5-fold",
            "Model": f"{bt['model']} (top_k={bt['top_k']})",
            "Accuracy": bt["metrics"]["accuracy"],
            "AUC": bt["metrics"]["auc"],
            "F1": bt["metrics"]["f1"],
            "Notes": "Human only, StratifiedGroupKFold",
        })
    gat = human.get("gat_grouped_5fold", {}).get("metrics")
    if gat:
        rows.append({
            "Evaluation": "Human grouped 5-fold",
            "Model": "GAT_v2",
            "Accuracy": gat["accuracy"],
            "AUC": gat["auc"],
            "F1": gat["f1"],
            "Notes": "Human only, StratifiedGroupKFold",
        })

if opt:
    rows.append({
        "Evaluation": "Human grouped 5-fold",
        "Model": "LR (tuned threshold)",
        "Accuracy": opt["metrics"]["accuracy"],
        "AUC": opt["metrics"]["auc"],
        "F1": opt["metrics"]["f1"],
        "Notes": f"Threshold tuned from OOF, top_k={opt['setup']['top_k_genes_per_fold']}, C={opt['setup']['C']}, thr={opt['setup']['threshold_tuned_on_oof']}",
    })

if l2:
    bf = l2.get("best_fixed")
    bt = l2.get("best_tuned")
    if bf:
        rows.append({
            "Evaluation": "Human grouped 5-fold",
            "Model": f"L2 LR fixed (feat={bf['cfg']['feat']}, top_k={bf['cfg']['top_k']}, C={bf['cfg']['C']})",
            "Accuracy": bf["fixed"]["acc"],
            "AUC": bf["fixed"]["auc"],
            "F1": bf["fixed"]["f1"],
            "Notes": "Fixed threshold 0.5",
        })
    if bt:
        rows.append({
            "Evaluation": "Human grouped 5-fold",
            "Model": f"L2 LR tuned (feat={bt['cfg']['feat']}, top_k={bt['cfg']['top_k']}, C={bt['cfg']['C']})",
            "Accuracy": bt["tuned"]["acc"],
            "AUC": bt["tuned"]["auc"],
            "F1": bt["tuned"]["f1"],
            "Notes": f"Tuned threshold {bt['tuned']['thr']}",
        })

metrics_df = pd.DataFrame(rows)
metrics_csv = os.path.join(TABLES, "metrics_summary.csv")
metrics_df.to_csv(metrics_csv, index=False)

# Markdown table
md_lines = ["# Metrics Summary", "", "| Evaluation | Model | Accuracy | AUC | F1 | Notes |", "|---|---|---:|---:|---:|---|"]
for _, r in metrics_df.iterrows():
    md_lines.append(f"| {r['Evaluation']} | {r['Model']} | {r['Accuracy']:.3f} | {r['AUC']:.3f} | {r['F1']:.3f} | {r['Notes']} |")
with open(os.path.join(TABLES, "metrics_summary.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

# Dataset counts table
meta_path = os.path.join(OI, "data", "processed", "multicohort_metadata.csv")
if os.path.exists(meta_path):
    meta = pd.read_csv(meta_path)
    counts = meta.groupby(["Dataset", "Condition"]).size().unstack(fill_value=0)
    counts["Total"] = counts.sum(axis=1)
    counts.to_csv(os.path.join(TABLES, "dataset_counts.csv"))

    md = ["# Dataset Composition", "", "| Dataset | OI | Control | Total |", "|---|---:|---:|---:|"]
    for ds, row in counts.iterrows():
        md.append(f"| {ds} | {int(row.get('OI', 0))} | {int(row.get('Control', 0))} | {int(row['Total'])} |")
    with open(os.path.join(TABLES, "dataset_counts.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

# Static figure: dataset composition
if os.path.exists(meta_path):
    counts = pd.read_csv(os.path.join(TABLES, "dataset_counts.csv"), index_col=0)
    if "OI" in counts.columns and "Control" in counts.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.index, counts["Control"], label="Control", color="#4c78a8")
        ax.bar(counts.index, counts["OI"], bottom=counts["Control"], label="OI", color="#f58518")
        ax.set_ylabel("Samples")
        ax.set_title("Human Multicohort Composition")
        ax.legend()
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(VIS, "dataset_composition.png"), dpi=150)
        plt.close()

# Static figure: metrics summary (accuracy)
if not metrics_df.empty:
    # choose a subset to keep the plot readable
    plot_df = metrics_df.copy()
    plot_df = plot_df[plot_df["Evaluation"].isin([
        "External holdout mean (human multicohort)",
        "Human grouped 5-fold",
    ])]
    # keep one row per model for external and grouped
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(plot_df["Model"], plot_df["Accuracy"], color="#72b7b2")
    ax.axhline(0.76, color="red", linestyle="--", linewidth=1, label="0.76 baseline")
    ax.axhline(0.90, color="green", linestyle="--", linewidth=1, label="0.90 target")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Key Evaluation Accuracies")
    ax.legend()
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS, "metrics_summary_accuracy.png"), dpi=150)
    plt.close()

# Static figure: pipeline overview
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")

boxes = [
    (0.02, 0.35, "Public GEO\nRNA-seq"),
    (0.20, 0.35, "Cleaning\nlog2 + QC"),
    (0.38, 0.35, "ComBat\nBatch Correction"),
    (0.58, 0.35, "Feature\nSelection"),
    (0.76, 0.35, "Graph\nConstruction"),
    (0.92, 0.35, "GAT\nClassifier"),
]

for x, y, label in boxes:
    ax.add_patch(plt.Rectangle((x, y), 0.16, 0.3, fill=False, edgecolor="#4c78a8", linewidth=2))
    ax.text(x + 0.08, y + 0.15, label, ha="center", va="center", fontsize=9)

for x in [0.18, 0.36, 0.56, 0.74, 0.90]:
    ax.annotate("", xy=(x + 0.02, 0.50), xytext=(x - 0.02, 0.50), arrowprops=dict(arrowstyle="->", lw=1.5))

ax.set_title("OI Pipeline Overview")
plt.tight_layout()
plt.savefig(os.path.join(VIS, "pipeline_overview.png"), dpi=150)
plt.close()

# Static figure: model architecture
fig, ax = plt.subplots(figsize=(6, 5))
ax.axis("off")

layers = [
    (0.1, 0.82, "Input\nGene Features"),
    (0.1, 0.65, "TransformerConv x1\n+ LN + Residual"),
    (0.1, 0.48, "TransformerConv x1\n+ LN + Residual"),
    (0.1, 0.31, "TransformerConv x1\n+ LN"),
    (0.1, 0.14, "Mean/Max Pool\nConcat"),
]

for x, y, label in layers:
    ax.add_patch(plt.Rectangle((x, y), 0.8, 0.12, fill=False, edgecolor="#f58518", linewidth=2))
    ax.text(x + 0.4, y + 0.06, label, ha="center", va="center", fontsize=9)

ax.annotate("", xy=(0.5, 0.80), xytext=(0.5, 0.74), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(0.5, 0.63), xytext=(0.5, 0.57), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(0.5, 0.46), xytext=(0.5, 0.40), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(0.5, 0.29), xytext=(0.5, 0.23), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(0.5, 0.04, "MLP Head -> OI vs Control", ha="center", va="center", fontsize=9)

ax.set_title("OIGATv2 Architecture")
plt.tight_layout()
plt.savefig(os.path.join(ARCH, "oigatv2_architecture.png"), dpi=150)
plt.close()

# Copy key existing figures
copy_pairs = [
    ("pca_before_combat.png", "pca_before_combat.png"),
    ("pca_after_combat.png", "pca_after_combat.png"),
    ("real_external_accuracy_by_holdout.png", "real_external_accuracy_by_holdout.png"),
    ("human_grouped5_accuracy.png", "human_grouped5_accuracy.png"),
    ("human_grouped5_l2_lr_tuning_best.png", "human_grouped5_l2_lr_tuning_best.png"),
    ("human_grouped5_optimized_lr_roc.png", "human_grouped5_optimized_lr_roc.png"),
    ("real_roc_GSE160207.png", "real_roc_GSE160207.png"),
    ("real_roc_GSE163812.png", "real_roc_GSE163812.png"),
    ("real_roc_GSE180838.png", "real_roc_GSE180838.png"),
    ("real_roc_GSE186141.png", "real_roc_GSE186141.png"),
    ("roc_gat_v2.png", "roc_gat_v2.png"),
    ("roc_gcn.png", "roc_gcn.png"),
    ("roc_logisticregression.png", "roc_logisticregression.png"),
    ("roc_randomforest.png", "roc_randomforest.png"),
]

import shutil
for src_name, dst_name in copy_pairs:
    src = os.path.join(FIG, src_name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(VIS, dst_name))

# Interactive HTML with Plotly CDN

def write_plotly_bar(path, title, x, y1, y2=None, y1_label="Series 1", y2_label="Series 2", y_label="Value"):
    import json as _json
    data = {
        "x": x,
        "y1": y1,
        "y2": y2,
        "y1_label": y1_label,
        "y2_label": y2_label,
        "y_label": y_label,
        "title": title,
    }
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
  <div id="chart" style="width:100%;height:600px;"></div>
  <script>
    const data = {_json.dumps(data)};
    const traces = [];
    traces.push({{x: data.x, y: data.y1, type: 'bar', name: data.y1_label}});
    if (data.y2) {{
      traces.push({{x: data.x, y: data.y2, type: 'bar', name: data.y2_label}});
    }}
    const layout = {{title: data.title, barmode: data.y2 ? 'group' : 'overlay', yaxis: {{title: data.y_label}}}};
    Plotly.newPlot('chart', traces, layout);
  </script>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def write_plotly_scatter(path, title, x, y, color=None, x_label="X", y_label="Y"):
    import json as _json
    data = {
        "x": x,
        "y": y,
        "color": color,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
    }
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
  <div id="chart" style="width:100%;height:600px;"></div>
  <script>
    const data = {_json.dumps(data)};
    const trace = {{x: data.x, y: data.y, mode: 'markers', type: 'scatter', marker: {{color: data.color || data.y}}}};
    const layout = {{title: data.title, xaxis: {{title: data.x_label}}, yaxis: {{title: data.y_label}}}};
    Plotly.newPlot('chart', [trace], layout);
  </script>
</body>
</html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

# Interactive: external accuracy by holdout
if real and "holdouts" in real:
    holdouts = list(real["holdouts"].keys())
    gat_acc = [real["holdouts"][h]["gat_external"]["accuracy"] for h in holdouts]
    lr_acc = [real["holdouts"][h]["lr_external"]["accuracy"] for h in holdouts]
    write_plotly_bar(
        os.path.join(INTER, "external_accuracy_by_holdout.html"),
        "External Accuracy by Holdout",
        holdouts,
        gat_acc,
        lr_acc,
        y1_label="GAT",
        y2_label="LR",
        y_label="Accuracy",
    )

# Interactive: dataset composition
if os.path.exists(meta_path):
    counts = pd.read_csv(os.path.join(TABLES, "dataset_counts.csv"), index_col=0)
    if "OI" in counts.columns and "Control" in counts.columns:
        write_plotly_bar(
            os.path.join(INTER, "dataset_composition.html"),
            "Human Multicohort Composition",
            counts.index.tolist(),
            counts["Control"].tolist(),
            counts["OI"].tolist(),
            y1_label="Control",
            y2_label="OI",
            y_label="Samples",
        )

# Interactive: L2 LR tuning scatter
if l2 and "top10" in l2:
    # plot all configs if available
    configs = []
    if os.path.exists(os.path.join(RES, "human_grouped5_l2_lr_tuning.json")):
        with open(os.path.join(RES, "human_grouped5_l2_lr_tuning.json"), "r", encoding="utf-8") as f:
            j = json.load(f)
        # top10 only in file, but build a plot from top10 for clarity
        top10 = j.get("top10", [])
        xs = [r["cfg"]["C"] for r in top10]
        ys = [r["tuned"]["acc"] for r in top10]
        write_plotly_scatter(
            os.path.join(INTER, "l2_lr_tuning_top10.html"),
            "Top10 L2 LR Tuning (Human Grouped 5-fold)",
            xs,
            ys,
            x_label="C",
            y_label="Accuracy",
        )

# Interactive: metrics summary
if not metrics_df.empty:
    write_plotly_bar(
        os.path.join(INTER, "metrics_summary_accuracy.html"),
        "Key Evaluation Accuracies",
        metrics_df["Model"].tolist(),
        metrics_df["Accuracy"].tolist(),
        y1_label="Accuracy",
        y_label="Accuracy",
    )

print("Publication assets generated")
