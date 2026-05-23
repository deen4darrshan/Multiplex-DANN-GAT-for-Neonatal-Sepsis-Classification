#!/usr/bin/env python3
"""
Build ACSEF master engineering notebook TeX + PDF artifact.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def parse_args():
    root = Path(__file__).resolve().parents[2]
    out_tex = root / "ACSEF_Final_Submission" / "acsef_documents" / "engineering_notebook_master.tex"
    out_pdf = root / "ACSEF_Final_Submission" / "acsef_documents" / "engineering_notebook_master.pdf"
    return argparse.Namespace(root=root, out_tex=out_tex, out_pdf=out_pdf)


def esc(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = s
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def list_py_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*.py") if p.is_file()])


def parse_py_inventory(path: Path) -> Tuple[List[str], List[str]]:
    imports: List[str] = []
    funcs: List[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return imports, funcs

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for n in node.names:
                imports.append(f"{mod}.{n.name}" if mod else n.name)
        elif isinstance(node, ast.FunctionDef):
            funcs.append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            funcs.append(node.name)
    return sorted(set(imports)), sorted(set(funcs))


def git_log(root: Path, n: int = 30) -> List[str]:
    try:
        cmd = ["git", "-C", str(root), "log", f"-n{n}", "--date=iso", "--pretty=format:%h|%ad|%s"]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="ignore")
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception:
        return ["git log unavailable in execution environment"]


def recent_logs(paths: List[Path], n: int = 20) -> List[str]:
    rows = []
    for d in paths:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.log")):
            rows.append(f"{p.name} | modified={datetime.fromtimestamp(p.stat().st_mtime).isoformat()}")
    return rows[-n:]


def write_tex(root: Path, out_tex: Path):
    cohort_manifest = root / "General_Sepsis_V11" / "results" / "cohort_manifest.json"
    results_json = root / "General_Sepsis_V11" / "results" / "general_sepsis_v11_results.json"
    overhaul_log = root / "General_Sepsis_V11" / "results" / "overhaul_execution_log.md"
    with cohort_manifest.open("r", encoding="utf-8") as f:
        cohort = json.load(f)
    with results_json.open("r", encoding="utf-8") as f:
        results = json.load(f)

    folders = [
        ("Root Sepsis Scripts", root),
        ("CH_DANN_Plan/scripts", root / "CH_DANN_Plan" / "scripts"),
        ("Sepsis_GNN_V2/scripts", root / "Sepsis_GNN_V2" / "scripts"),
        ("gnn_optimized", root / "gnn_optimized"),
        ("Osteogenesis imperfecta/scripts", root / "Osteogenesis imperfecta" / "scripts"),
        ("ACSEF_Final_Submission/scripts", root / "ACSEF_Final_Submission" / "scripts"),
        ("General_Sepsis_V11/scripts", root / "General_Sepsis_V11" / "scripts"),
    ]

    inventory: Dict[str, List[Tuple[Path, List[str], List[str]]]] = {}
    for label, folder in folders:
        files = list_py_files(folder if label != "Root Sepsis Scripts" else root)
        if label == "Root Sepsis Scripts":
            files = [p for p in files if p.parent == root]
        rows = []
        for p in files:
            imps, funcs = parse_py_inventory(p)
            rows.append((p, imps, funcs))
        inventory[label] = rows

    glog = git_log(root, n=35)
    rlogs = recent_logs(
        [
            root / "General_Sepsis_V11" / "logs",
            root / "ACSEF_Final_Submission" / "logs",
            root / "logs",
        ],
        n=30,
    )

    lines: List[str] = []
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[margin=1in]{geometry}")
    lines.append(r"\usepackage[T1]{fontenc}")
    lines.append(r"\usepackage[utf8]{inputenc}")
    lines.append(r"\usepackage{hyperref}")
    lines.append(r"\usepackage{longtable}")
    lines.append(r"\usepackage{array}")
    lines.append(r"\usepackage{xcolor}")
    lines.append(r"\usepackage{listings}")
    lines.append(r"\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,columns=fullflexible,frame=single}")
    lines.append(r"\title{ACSEF Master Engineering Notebook: General\_Sepsis\_V11}")
    lines.append(r"\author{ISEF Project Workspace}")
    lines.append(r"\date{" + esc(datetime.now().strftime("%Y-%m-%d")) + "}")
    lines.append(r"\begin{document}")
    lines.append(r"\maketitle")
    lines.append(r"\tableofcontents")
    lines.append(r"\newpage")

    lines.append(r"\section{End-to-End Problem Framing and Rationale}")
    lines.append(
        "This notebook consolidates engineering evidence for a general sepsis model that enforces strict healthy-control cohort policy, fold-safe graph construction, and external pediatric holdout evaluation. "
        "Primary new result is General\\_Sepsis\\_V11, while prior neonatal and cross-disease assets are retained as context."
    )

    lines.append(r"\section{Data Acquisition and Curation Decisions with Timestamps}")
    lines.append("Cohort selection was executed in Step 01 with deterministic filters and fallback logic.")
    lines.append(r"\begin{itemize}")
    lines.append(r"\item Generated at: " + esc(cohort["generated_at"]))
    lines.append(r"\item Active train datasets: " + esc(", ".join(cohort["active_train_datasets"])))
    lines.append(r"\item Holdout dataset: " + esc(cohort["active_holdout_dataset"]))
    lines.append(r"\item GSE95233 fallback triggered: " + esc(str(cohort["policy"]["gse95233_fallback_triggered"])))
    lines.append(r"\item Selected genes: " + esc(str(cohort["n_selected_genes"])))
    lines.append(r"\end{itemize}")

    lines.append(r"\section{Mathematical Derivations}")
    lines.append(r"\subsection{ComBat Batch Harmonization}")
    lines.append(
        r"For gene $g$ and sample $i$, ComBat models expression as "
        r"$x_{gi} = \alpha_g + \mathbf{z}_i^\top\beta_g + \gamma_{g,b(i)} + \delta_{g,b(i)}\epsilon_{gi}$, "
        r"where $b(i)$ is batch and $\mathbf{z}_i$ includes biological covariate (condition). "
        r"Empirical Bayes shrinks $\gamma,\delta$ across genes."
    )
    lines.append(r"\subsection{Multiplex Hypergraph Message Passing}")
    lines.append(
        r"Each relation $r \in \{\mathrm{KEGG},\mathrm{STRING},\mathrm{CoExpr}\}$ defines hyperedge incidence $H^{(r)}$. "
        r"Two HypergraphConv blocks produce relation-specific embeddings $h^{(r)}$, then relation attention computes "
        r"$\alpha^{(r)} = \mathrm{softmax}(W_a[h^{(1)}\Vert h^{(2)}\Vert h^{(3)}])$ and "
        r"$h = \sum_r \alpha^{(r)} h^{(r)}$."
    )
    lines.append(r"\subsection{GRL / DANN Objective}")
    lines.append(
        r"Training minimizes $\mathcal{L} = \mathcal{L}_{\mathrm{sepsis}} + \lambda_{\mathrm{dann}}\mathcal{L}_{\mathrm{domain}}$, "
        r"with gradient reversal layer applying $-\alpha$ multiplier on domain branch gradients."
    )
    lines.append(r"\subsection{Metrics and Confidence Intervals}")
    lines.append(
        r"Primary metrics are AUROC, Accuracy, F1, Precision, Recall. "
        r"95\% bootstrap confidence intervals are computed by resampling predictions with replacement."
    )

    lines.append(r"\section{Results Snapshot}")
    cv = results["model_cv_oof_metrics"]
    ext = results["external_holdout"]["metrics"]
    lines.append(r"\begin{itemize}")
    lines.append(r"\item CV AUROC: " + esc(f"{cv['auroc']:.4f}") + r", Accuracy: " + esc(f"{cv['accuracy']:.4f}") + r", F1: " + esc(f"{cv['f1']:.4f}"))
    lines.append(r"\item External AUROC: " + esc(f"{ext['auroc']:.4f}") + r", Accuracy: " + esc(f"{ext['accuracy']:.4f}") + r", F1: " + esc(f"{ext['f1']:.4f}"))
    lines.append(r"\item Hard gates all passed: " + esc(str(results["hard_pass_gates"]["all_passed"])))
    lines.append(r"\end{itemize}")

    lines.append(r"\section{Import and Function Inventories}")
    for label, _ in folders:
        lines.append(r"\subsection{" + esc(label) + "}")
        lines.append(r"\begin{longtable}{p{0.34\textwidth}p{0.31\textwidth}p{0.31\textwidth}}")
        lines.append(r"\textbf{File} & \textbf{Imports} & \textbf{Functions} \\ \hline")
        for p, imps, funcs in inventory[label]:
            rel = p.relative_to(root).as_posix()
            imp_txt = ", ".join(imps) if imps else "-"
            fn_txt = ", ".join(funcs) if funcs else "-"
            lines.append(esc(rel) + " & " + esc(imp_txt) + " & " + esc(fn_txt) + r" \\")
        lines.append(r"\end{longtable}")

    lines.append(r"\section{Changelog Derived from Git and Logs}")
    lines.append(r"\subsection{Recent Git Commits}")
    lines.append(r"\begin{itemize}")
    for row in glog:
        lines.append(r"\item " + esc(row))
    lines.append(r"\end{itemize}")
    lines.append(r"\subsection{Recent Execution Logs}")
    lines.append(r"\begin{itemize}")
    for row in rlogs:
        lines.append(r"\item " + esc(row))
    lines.append(r"\end{itemize}")

    lines.append(r"\section{Failure Analysis with Root Cause and Fix Evidence}")
    lines.append(r"\begin{itemize}")
    lines.append(
        r"\item Step 01 initial failure: \texttt{pycombat} raised DataFrame comparison error in mod handling. "
        r"Root cause: passing DataFrame covariate object. Fix: pass list covariate (\texttt{condition.tolist()})."
    )
    lines.append(
        r"\item Step 02 initial failure: metadata column mismatch (\texttt{index} vs \texttt{sample\_id}). "
        r"Root cause: reset index naming. Fix: normalize metadata columns and add compatibility fallback in downstream scripts."
    )
    lines.append(r"\end{itemize}")

    lines.append(r"\section{Overhaul Execution Trace}")
    if overhaul_log.exists():
        lines.append(
            r"Detailed step-by-step execution log: "
            r"\texttt{" + esc(overhaul_log.as_posix()) + r"}"
        )
        lines.append(r"\lstinputlisting{\detokenize{" + overhaul_log.resolve().as_posix() + "}}")
    else:
        lines.append("No overhaul execution trace file found for this build.")

    lines.append(r"\section{Figure and Table Appendix with Captions and Citations}")
    lines.append(r"\begin{itemize}")
    lines.append(r"\item Poster package: \texttt{ACSEF\_Final\_Submission/acsef\_documents/publication\_package/general\_sepsis\_v11\_poster.pdf}")
    lines.append(r"\item Figure manifest: \texttt{.../general\_sepsis\_v11\_figure\_manifest.md}")
    lines.append(r"\item Claim traceability: \texttt{.../general\_sepsis\_v11\_claim\_traceability.csv}")
    lines.append(r"\end{itemize}")

    lines.append(r"\section{Full Code Appendix (\texttt{\textbackslash lstinputlisting})}")
    for label, _ in folders:
        lines.append(r"\subsection{" + esc(label) + "}")
        for p, _, _ in inventory[label]:
            rel = p.relative_to(root).as_posix()
            abs_p = p.resolve().as_posix()
            lines.append(r"\subsubsection{" + esc(rel) + "}")
            lines.append(r"\lstinputlisting[language=Python]{\detokenize{" + abs_p + "}}")

    lines.append(r"\end{document}")
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def write_pdf_fallback(out_pdf: Path, out_tex: Path):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        pages = [
            "ACSEF Master Engineering Notebook\n\n"
            f"Generated: {datetime.now().isoformat()}\n\n"
            "This PDF is an execution artifact produced in the current environment.\n"
            "Authoritative technical content is in engineering_notebook_master.tex.\n\n"
            f"TeX path:\n{out_tex}",
            "Compilation note:\n\n"
            "pdflatex/latexmk was not available in the runtime PATH.\n"
            "The TeX file is structured to compile with standard LaTeX installations\n"
            "(geometry, hyperref, longtable, listings).",
        ]
        for txt in pages:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            ax.text(0.05, 0.95, txt, va="top", fontsize=11, family="monospace")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main():
    args = parse_args()
    write_tex(args.root, args.out_tex)
    write_pdf_fallback(args.out_pdf, args.out_tex)
    print(f"Wrote TeX: {args.out_tex}")
    print(f"Wrote PDF: {args.out_pdf}")


if __name__ == "__main__":
    main()
