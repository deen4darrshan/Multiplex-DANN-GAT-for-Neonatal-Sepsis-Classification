"""
Prepare a true domain-labeled Alzheimer dataset from GEO SOFT files.

Default cohorts:
  - GSE63060 (GPL6947)
  - GSE63061 (GPL10558)

Labels kept by default:
  - AD
  - Control

Outputs:
  - CH_DANN_Plan/data/alz/alz_blood_true_domains_2000.pt
  - CH_DANN_Plan/data/alz/gene_list_2000.txt
  - CH_DANN_Plan/data/alz/alz_blood_expression_top2000.csv
  - CH_DANN_Plan/data/alz/alz_blood_metadata_top2000.csv
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import GEOparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import median_abs_deviation
from torch_geometric.data import Data


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build true domain-labeled AD dataset")
    parser.add_argument(
        "--gse-files",
        nargs="+",
        default=[
            os.path.join(
                "ALZHEIMERS_STRATEGIC_PATHWAY",
                "data",
                "adni",
                "raw_expanded",
                "GSE63060_family.soft.gz",
            ),
            os.path.join(
                "ALZHEIMERS_STRATEGIC_PATHWAY",
                "data",
                "adni",
                "raw_expanded",
                "GSE63061_family.soft.gz",
            ),
        ],
    )
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--min-common-genes", type=int, default=5000)
    parser.add_argument(
        "--max-static-edges",
        type=int,
        default=15000,
        help="Max number of absolute-correlation edges used for the static relation graph.",
    )
    parser.add_argument("--out-dir", default=os.path.join("CH_DANN_Plan", "data", "alz"))
    parser.add_argument("--out-prefix", default="alz_blood_true_domains")
    return parser.parse_args()


def normalize_status(raw_status: str) -> str:
    s = raw_status.strip().lower()
    s = s.replace("ā", "").replace("â", "").replace("Â", "").strip()
    s = s.encode("ascii", errors="ignore").decode().strip()

    if any(k in s for k in {"incipient", "moderate", "severe", "affected"}):
        return "AD"
    if "alzheimer" in s:
        return "AD"
    if "normal" in s or "control" in s:
        return "Control"
    if s in {"ad", "alzheimer's disease", "alzheimers disease", "alzheimer disease"}:
        return "AD"
    if s in {"ctl", "control", "normal", "healthy"}:
        return "Control"
    if "mci" in s:
        return "MCI"
    if "vascular dementia" in s:
        return "VaD"
    return "Unknown"


def extract_status(gsm: GEOparse.GSM) -> str:
    for c in gsm.metadata.get("characteristics_ch1", []):
        cl = str(c).lower()
        for key in (
            "status:",
            "diagnosis:",
            "patient diagnosis:",
            "disease state:",
            "clinical diagnosis:",
            "group:",
            "condition:",
        ):
            if key in cl:
                return cl.split(key, 1)[1].strip()

    # Fallback for datasets that encode diagnosis in title/source fields.
    title = str(gsm.metadata.get("title", [""])[0]).lower().replace("ā", "").strip()
    source = str(gsm.metadata.get("source_name_ch1", [""])[0]).lower().replace("ā", "").strip()
    txt = f"{title} {source}"
    if any(k in txt for k in ("control", "normal")):
        return "control"
    if any(k in txt for k in ("incipient", "moderate", "severe", "affected", "alzheimer")):
        return "alzheimer's disease"
    if "vascular dementia" in txt or re.search(r"\bvad\b", txt):
        return "vascular dementia"
    if re.search(r"(^|[_\\s-])ad([_\\s-]|$)", txt):
        return "ad"
    return "unknown"


def make_probe_to_gene(gpl_table: pd.DataFrame) -> Dict[str, str]:
    id_col = "ID" if "ID" in gpl_table.columns else gpl_table.columns[0]
    sym_col = None
    for c in ("Symbol", "Gene Symbol", "GENE_SYMBOL", "symbol", "gene_symbol"):
        if c in gpl_table.columns:
            sym_col = c
            break
    if sym_col is None:
        raise ValueError(f"No symbol-like column found. Columns: {list(gpl_table.columns)}")

    sub = gpl_table[[id_col, sym_col]].copy()
    sub[sym_col] = sub[sym_col].astype(str).str.strip()
    sub = sub[
        (sub[sym_col] != "")
        & (sub[sym_col] != "---")
        & (sub[sym_col] != "nan")
        & (sub[sym_col] != "NA")
        & (sub[sym_col] != "na")
    ]

    # Keep first symbol token for multi-mapped probes.
    sub[sym_col] = (
        sub[sym_col]
        .str.split("///")
        .str[0]
        .str.split("//")
        .str[0]
        .str.split(";")
        .str[0]
        .str.strip()
    )
    sub = sub[sub[sym_col] != ""]
    return dict(zip(sub[id_col].astype(str), sub[sym_col]))


def process_one_gse(filepath: str) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    gse_id = os.path.basename(filepath).split("_")[0]
    gse = GEOparse.get_GEO(filepath=filepath, silent=True)
    gpl_id = list(gse.gpls.keys())[0]
    gpl_table = gse.gpls[gpl_id].table
    probe_to_gene = make_probe_to_gene(gpl_table)

    expr_dict: Dict[str, pd.Series] = {}
    meta_rows: List[Dict[str, object]] = []
    raw_status_counter = Counter()
    kept_counter = Counter()

    for sid, gsm in gse.gsms.items():
        raw_status = extract_status(gsm)
        status = normalize_status(raw_status)
        raw_status_counter[status] += 1

        if status not in {"AD", "Control"}:
            continue

        tab = gsm.table
        if "ID_REF" not in tab.columns or "VALUE" not in tab.columns:
            continue

        df = tab[["ID_REF", "VALUE"]].copy()
        df["gene"] = df["ID_REF"].astype(str).map(probe_to_gene)
        df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
        df = df.dropna(subset=["gene", "VALUE"])
        if df.empty:
            continue

        gene_vec = df.groupby("gene", sort=False)["VALUE"].mean()
        expr_dict[sid] = gene_vec

        kept_counter[status] += 1
        meta_rows.append(
            {
                "SampleID": sid,
                "Condition": status,
                "Label": 1 if status == "AD" else 0,
                "Batch": gse_id,
                "Platform": gpl_id,
            }
        )

    expr_df = pd.DataFrame(expr_dict)
    meta_df = pd.DataFrame(meta_rows)

    print(f"\n{gse_id} ({gpl_id})")
    print(f"  Raw status counts: {dict(raw_status_counter)}")
    print(f"  Kept AD/Control:   {dict(kept_counter)}")
    print(f"  Matrix shape:      {expr_df.shape[0]} genes x {expr_df.shape[1]} samples")
    return gse_id, expr_df, meta_df


def zscore_per_sample(expr: pd.DataFrame) -> pd.DataFrame:
    # Expression matrix is genes x samples.
    means = expr.mean(axis=0)
    stds = expr.std(axis=0).replace(0.0, 1.0)
    return (expr - means) / stds


def build_static_edge_index(expr_top: pd.DataFrame, max_edges: int) -> torch.Tensor:
    # expr_top: genes x samples
    vals = expr_top.values.astype(np.float32)
    corr = np.corrcoef(vals)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)

    tri_i, tri_j = np.triu_indices(corr.shape[0], k=1)
    tri_abs = np.abs(corr[tri_i, tri_j])
    if max_edges <= 0 or max_edges >= tri_abs.shape[0]:
        keep = np.arange(tri_abs.shape[0])
    else:
        keep = np.argpartition(-tri_abs, max_edges - 1)[:max_edges]

    ii = tri_i[keep].astype(np.int64)
    jj = tri_j[keep].astype(np.int64)
    src = np.concatenate([ii, jj], axis=0)
    dst = np.concatenate([jj, ii], axis=0)
    return torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)


def main() -> None:
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    all_expr: Dict[str, pd.DataFrame] = {}
    all_meta: List[pd.DataFrame] = []
    for fp in args.gse_files:
        abs_fp = os.path.abspath(fp)
        if not os.path.exists(abs_fp):
            raise FileNotFoundError(f"GSE file missing: {abs_fp}")
        gse_id, expr_df, meta_df = process_one_gse(abs_fp)
        if expr_df.empty or meta_df.empty:
            raise ValueError(f"No usable AD/Control samples in {gse_id}")
        all_expr[gse_id] = expr_df
        all_meta.append(meta_df)

    common = set(next(iter(all_expr.values())).index)
    for df in all_expr.values():
        common &= set(df.index)
    common_genes = sorted(common)
    print(f"\nCommon genes across cohorts: {len(common_genes)}")
    if len(common_genes) < args.min_common_genes:
        raise ValueError(
            f"Only {len(common_genes)} common genes; below min-common-genes={args.min_common_genes}"
        )

    expr_merged = pd.concat([df.loc[common_genes] for df in all_expr.values()], axis=1)
    meta_merged = pd.concat(all_meta, axis=0, ignore_index=True)
    meta_merged = meta_merged.set_index("SampleID").loc[expr_merged.columns].reset_index()
    if "index" in meta_merged.columns and "SampleID" not in meta_merged.columns:
        meta_merged = meta_merged.rename(columns={"index": "SampleID"})

    # Fill remaining missing values gene-wise.
    expr_merged = expr_merged.T
    expr_merged = expr_merged.fillna(expr_merged.mean())
    expr_merged = expr_merged.T

    expr_norm = zscore_per_sample(expr_merged)

    # Feature selection by MAD across all AD/Control samples.
    mad = expr_norm.apply(median_abs_deviation, axis=1)
    top_genes = mad.sort_values(ascending=False).head(args.top_k).index.tolist()
    expr_top = expr_norm.loc[top_genes]
    static_edge_index = build_static_edge_index(expr_top, max_edges=args.max_static_edges)

    label_counts = meta_merged["Condition"].value_counts().to_dict()
    batch_counts = meta_merged["Batch"].value_counts().to_dict()
    print(f"Final samples: {expr_top.shape[1]} | labels: {label_counts} | batches: {batch_counts}")
    print(f"Static relation edges: {static_edge_index.shape[1] // 2}")

    domain_map = {b: i for i, b in enumerate(sorted(meta_merged["Batch"].unique()))}
    data_list: List[Data] = []
    for _, row in meta_merged.iterrows():
        sid = row["SampleID"]
        label = int(row["Label"])
        batch = str(row["Batch"])
        vec = expr_top[sid].values.astype(np.float32)

        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(1)
        d = Data(x=x, y=torch.tensor(label, dtype=torch.long))
        d.sample_id = sid
        d.batch_label = batch
        d.domain_y = torch.tensor(domain_map[batch], dtype=torch.long)
        d.global_feat = x.squeeze(1).unsqueeze(0)
        d.edge_index = static_edge_index.clone()
        d.num_nodes = int(x.size(0))
        data_list.append(d)

    pt_path = os.path.join(out_dir, f"{args.out_prefix}_{args.top_k}.pt")
    torch.save(data_list, pt_path)

    gene_path = os.path.join(out_dir, f"gene_list_{args.top_k}.txt")
    with open(gene_path, "w", encoding="utf-8") as f:
        for g in top_genes:
            f.write(f"{g}\n")

    expr_csv = os.path.join(out_dir, f"{args.out_prefix}_expression_top{args.top_k}.csv")
    meta_csv = os.path.join(out_dir, f"{args.out_prefix}_metadata_top{args.top_k}.csv")
    expr_top.to_csv(expr_csv)
    meta_merged.to_csv(meta_csv, index=False)

    print("\nSaved:")
    print(f"  {pt_path}")
    print(f"  {gene_path}")
    print(f"  {expr_csv}")
    print(f"  {meta_csv}")


if __name__ == "__main__":
    main()
