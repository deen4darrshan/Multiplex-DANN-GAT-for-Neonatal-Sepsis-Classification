#!/usr/bin/env python3
"""
Compare sepsis GAT-only baseline against current hybrid GCN+MLP DANN model.
Writes a reproducible test artifact used to decide whether visuals should be re-based to GAT.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SEPSIS_METRICS = ROOT / "General_Sepsis_V11" / "results" / "metrics_overall.csv"
SEPSIS_GNN_RESULTS = ROOT / "Sepsis_GNN_V2" / "results" / "gnn_results.json"
COMPILED_METRICS = ROOT / "ACSEF_Final_Submission" / "results" / "compiled_model_metrics.json"
OUT_JSON = ROOT / "results" / "sepsis" / "gat_vs_hybrid_test.json"
OUT_MD = ROOT / "results" / "sepsis" / "gat_vs_hybrid_test.md"


def main() -> None:
    overall = pd.read_csv(SEPSIS_METRICS)
    hybrid = overall.loc[overall["model"] == "hybrid_v11"].iloc[0]

    gnn_blob = json.loads(SEPSIS_GNN_RESULTS.read_text(encoding="utf-8"))
    gat = gnn_blob["GAT_Transfer"]

    compiled_blob = json.loads(COMPILED_METRICS.read_text(encoding="utf-8"))
    compiled_hybrid = None
    for row in compiled_blob.get("sepsis_benchmarks", []):
        if row.get("model_name") == "Multiplex-Hypergraph-DANN-MLP":
            compiled_hybrid = row
            break

    robust_compare = {
        "hybrid_auroc": float(hybrid["auroc"]),
        "hybrid_accuracy": float(hybrid["accuracy"]),
        "hybrid_f1": float(hybrid["f1"]),
        "gat_auroc": float(gat["mean_auc"]),
        "gat_accuracy": float(gat["mean_acc"]),
        "gat_f1": float(gat["mean_f1"]),
    }
    robust_compare["gat_beats_hybrid"] = bool(
        robust_compare["gat_auroc"] > robust_compare["hybrid_auroc"]
        and robust_compare["gat_accuracy"] >= robust_compare["hybrid_accuracy"]
    )

    compiled_compare = None
    if compiled_hybrid is not None:
        compiled_compare = {
            "hybrid_auroc": float(compiled_hybrid["auroc_mean"]),
            "hybrid_accuracy": float(compiled_hybrid["accuracy_mean"]),
            "hybrid_f1": float(compiled_hybrid["f1_mean"]),
            "gat_auroc": float(gat["mean_auc"]),
            "gat_accuracy": float(gat["mean_acc"]),
            "gat_f1": float(gat["mean_f1"]),
        }
        compiled_compare["gat_beats_hybrid"] = bool(
            compiled_compare["gat_auroc"] > compiled_compare["hybrid_auroc"]
            and compiled_compare["gat_accuracy"] >= compiled_compare["hybrid_accuracy"]
        )

    payload = {
        "decision_rule": "GAT must exceed hybrid AUROC and at least match hybrid accuracy.",
        "robust_metrics_comparison": robust_compare,
        "compiled_metrics_comparison": compiled_compare,
        "final_decision": "keep_hybrid_visuals" if not robust_compare["gat_beats_hybrid"] else "replace_with_gat",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# GAT vs Hybrid Test",
        "",
        f"- Rule: {payload['decision_rule']}",
        "- Robust (General_Sepsis_V11) comparison:",
        f"  - Hybrid AUROC: {robust_compare['hybrid_auroc']:.4f}",
        f"  - GAT AUROC: {robust_compare['gat_auroc']:.4f}",
        f"  - Hybrid Accuracy: {robust_compare['hybrid_accuracy']:.4f}",
        f"  - GAT Accuracy: {robust_compare['gat_accuracy']:.4f}",
        f"  - GAT beats hybrid: {robust_compare['gat_beats_hybrid']}",
    ]
    if compiled_compare is not None:
        md_lines.extend(
            [
                "- Compiled benchmark comparison:",
                f"  - Hybrid AUROC: {compiled_compare['hybrid_auroc']:.4f}",
                f"  - GAT AUROC: {compiled_compare['gat_auroc']:.4f}",
                f"  - Hybrid Accuracy: {compiled_compare['hybrid_accuracy']:.4f}",
                f"  - GAT Accuracy: {compiled_compare['gat_accuracy']:.4f}",
                f"  - GAT beats hybrid: {compiled_compare['gat_beats_hybrid']}",
            ]
        )
    md_lines.extend(["", f"- Final decision: `{payload['final_decision']}`"])
    OUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
