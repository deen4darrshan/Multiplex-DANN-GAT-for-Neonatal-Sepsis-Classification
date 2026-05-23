#!/usr/bin/env python3
"""
General_Sepsis_V11 - Step 01
Download GEO cohorts, enforce cohort policy, harmonize expression, and export artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import GEOparse
import numpy as np
import pandas as pd
from combat.pycombat import pycombat


@dataclass
class SampleRecord:
    dataset: str
    gsm_id: str
    sample_id: str
    condition: str
    batch: str
    platform: str
    patient_id: str
    timepoint: str
    split_role: str
    include_reason: str


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    default_out = default_root / "results"
    default_logs = default_root / "logs"
    today = datetime.now().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Download and preprocess General_Sepsis_V11 cohorts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(default_out))
    parser.add_argument("--raw-dir", default=str(default_root / "data" / "raw"))
    parser.add_argument("--log-file", default=str(default_logs / f"{today}_01_download_and_preprocess.log"))
    parser.add_argument("--holdout-dataset", default="GSE26378")
    parser.add_argument("--top-k-genes", type=int, default=2000)
    parser.add_argument("--gse95233-min-sepsis", type=int, default=1)
    parser.add_argument("--gse95233-min-control", type=int, default=1)
    parser.add_argument(
        "--fallback-dataset",
        default="GSE134347",
        help="Deterministic fallback when GSE95233 admission/healthy parsing fails QC.",
    )
    return parser.parse_args()


def init_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("general_sepsis_v11_preprocess")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def load_gse(geo_id: str, raw_dir: Path, logger: logging.Logger):
    raw_dir.mkdir(parents=True, exist_ok=True)
    soft = raw_dir / f"{geo_id}_family.soft.gz"
    if soft.exists():
        logger.info("Loading %s from local SOFT: %s", geo_id, soft)
        gse = GEOparse.get_GEO(filepath=str(soft), silent=True)
    else:
        logger.info("Downloading %s from GEO", geo_id)
        gse = GEOparse.get_GEO(geo=geo_id, destdir=str(raw_dir), how="full", silent=True)
    return gse


def normalize_gene_symbol(symbol: str) -> Optional[str]:
    if symbol is None:
        return None
    s = str(symbol).strip().upper()
    s = s.replace('"', "").replace("'", "")
    s = re.sub(r"\s+", "", s)
    bad = {"", "---", "NA", "N/A", "NULL", "NONE", "?", "NAN"}
    if s in bad:
        return None
    return s


def parse_gene_assignment(value: str) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    txt = str(value)
    if txt.strip() in {"", "---"}:
        return None
    # gene_assignment format often uses "ID // SYMBOL // description ..."
    chunks = [c.strip() for c in re.split(r"///|//", txt) if c.strip()]
    # Prefer tokens that look like symbols and are not transcript IDs.
    for token in chunks:
        if token.startswith(("NM_", "NR_", "ENST", "XM_", "XR_")):
            continue
        if re.fullmatch(r"[A-Za-z0-9\-\.]+", token):
            norm = normalize_gene_symbol(token)
            if norm:
                return norm
    return None


def detect_gene_column(annot: pd.DataFrame) -> Optional[str]:
    priority = [
        "Gene Symbol",
        "GENE_SYMBOL",
        "Gene symbol",
        "Symbol",
        "SYMBOL",
        "symbol",
        "ILMN_Gene",
        "gene_assignment",
        "GENE",
    ]
    for col in priority:
        if col in annot.columns:
            return col
    for col in annot.columns:
        low = col.lower()
        if "gene" in low and "symbol" in low:
            return col
    return None


def build_probe_to_gene_map(gse, dataset: str, logger: logging.Logger) -> Tuple[Dict[str, str], str]:
    gpl_ids = list(gse.gpls.keys())
    if not gpl_ids:
        raise RuntimeError(f"{dataset}: no GPL annotation found")
    gpl_id = gpl_ids[0]
    annot = gse.gpls[gpl_id].table.copy()
    id_col = "ID" if "ID" in annot.columns else annot.columns[0]
    gene_col = detect_gene_column(annot)
    if not gene_col:
        raise RuntimeError(f"{dataset}: no gene symbol column found in GPL {gpl_id}")

    logger.info("%s: GPL=%s, gene column=%s", dataset, gpl_id, gene_col)

    probe_to_gene: Dict[str, str] = {}
    for _, row in annot[[id_col, gene_col]].iterrows():
        probe = str(row[id_col]).strip()
        raw = row[gene_col]
        if gene_col == "gene_assignment":
            gene = parse_gene_assignment(raw)
        else:
            token = str(raw).split("///")[0].split("//")[0].strip() if pd.notna(raw) else ""
            gene = normalize_gene_symbol(token)
        if probe and gene:
            probe_to_gene[probe] = gene

    logger.info("%s: probe->gene entries=%d", dataset, len(probe_to_gene))
    return probe_to_gene, gpl_id


def extract_expression_matrix(
    gse,
    dataset: str,
    selected_gsms: List[str],
    probe_to_gene: Dict[str, str],
    sample_id_map: Dict[str, str],
    logger: logging.Logger,
) -> pd.DataFrame:
    cols: Dict[str, pd.Series] = {}
    for gsm_id in selected_gsms:
        gsm = gse.gsms[gsm_id]
        table = gsm.table
        if "ID_REF" not in table.columns or "VALUE" not in table.columns:
            continue
        s = pd.to_numeric(table["VALUE"], errors="coerce")
        s.index = table["ID_REF"].astype(str)
        s = s.dropna()
        mapped = s.index.map(lambda p: probe_to_gene.get(str(p)))
        keep = mapped.notna()
        if keep.sum() == 0:
            continue
        s = s.loc[keep]
        s.index = mapped[keep].astype(str)
        s = s.groupby(s.index).mean()
        cols[sample_id_map[gsm_id]] = s

    if not cols:
        raise RuntimeError(f"{dataset}: no expression columns extracted for selected samples")
    expr = pd.DataFrame(cols)
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(how="all")
    expr = expr.groupby(expr.index).mean()
    logger.info("%s: mapped expression shape=%s", dataset, expr.shape)
    return expr


def _mk_sample_id(dataset: str, gsm_id: str) -> str:
    return f"{dataset}:{gsm_id}"


def select_gse54514(gse, logger: logging.Logger) -> Tuple[List[SampleRecord], Counter]:
    included: List[SampleRecord] = []
    excluded = Counter()
    gpl = list(gse.gpls.keys())[0]
    for gsm_id, gsm in gse.gsms.items():
        title = gsm.metadata.get("title", [""])[0]
        low = title.lower()
        day = re.search(r"(day_\d+)", low)
        timepoint = day.group(1).upper() if day else "NA"
        pid_match = re.search(r"id\s*=\s*([0-9]+)", low)
        pid = pid_match.group(1) if pid_match else gsm_id

        if "day_1" not in low:
            excluded["not_day_1"] += 1
            continue
        if "control" in low:
            condition = "control"
        elif "sepsis_" in low:
            condition = "sepsis"
        else:
            excluded["not_control_or_sepsis"] += 1
            continue

        included.append(
            SampleRecord(
                dataset="GSE54514",
                gsm_id=gsm_id,
                sample_id=_mk_sample_id("GSE54514", gsm_id),
                condition=condition,
                batch="GSE54514",
                platform=gpl,
                patient_id=f"GSE54514_{pid}",
                timepoint=timepoint,
                split_role="train",
                include_reason="Control Day_1 or sepsis_* Day_1",
            )
        )
    logger.info("GSE54514 selected=%d excluded=%d", len(included), sum(excluded.values()))
    return included, excluded


def select_gse57065(gse, logger: logging.Logger) -> Tuple[List[SampleRecord], Counter]:
    included: List[SampleRecord] = []
    excluded = Counter()
    gpl = list(gse.gpls.keys())[0]
    for gsm_id, gsm in gse.gsms.items():
        title = gsm.metadata.get("title", [""])[0]
        low = title.lower()
        m_sepsis = re.match(r"blood_(p[0-9]+)_(h[0-9]+)$", low)
        m_hv = re.match(r"blood_(hv[0-9]+)$", low)

        if m_sepsis:
            pid = m_sepsis.group(1).upper()
            tp = m_sepsis.group(2).upper()
            if tp != "H00":
                excluded["sepsis_not_h00"] += 1
                continue
            included.append(
                SampleRecord(
                    dataset="GSE57065",
                    gsm_id=gsm_id,
                    sample_id=_mk_sample_id("GSE57065", gsm_id),
                    condition="sepsis",
                    batch="GSE57065",
                    platform=gpl,
                    patient_id=f"GSE57065_{pid}",
                    timepoint=tp,
                    split_role="train",
                    include_reason="Sepsis H00",
                )
            )
            continue

        if m_hv:
            hv = m_hv.group(1).upper()
            included.append(
                SampleRecord(
                    dataset="GSE57065",
                    gsm_id=gsm_id,
                    sample_id=_mk_sample_id("GSE57065", gsm_id),
                    condition="control",
                    batch="GSE57065",
                    platform=gpl,
                    patient_id=f"GSE57065_{hv}",
                    timepoint="HV",
                    split_role="train",
                    include_reason="Healthy volunteer HV..",
                )
            )
            continue

        excluded["not_h00_or_hv"] += 1

    logger.info("GSE57065 selected=%d excluded=%d", len(included), sum(excluded.values()))
    return included, excluded


def select_gse95233_strict(gse, logger: logging.Logger) -> Tuple[List[SampleRecord], Counter]:
    included: List[SampleRecord] = []
    excluded = Counter()
    gpl = list(gse.gpls.keys())[0]
    for gsm_id, gsm in gse.gsms.items():
        title = gsm.metadata.get("title", [""])[0]
        src = gsm.metadata.get("source_name_ch1", [""])[0]
        chars = "; ".join(gsm.metadata.get("characteristics_ch1", []))
        low = f"{title} {src} {chars}".lower()

        m = re.match(r"blood\-([a-z]+)_([0-9]+)_(d[0-9]+)$", title.lower())
        if not m:
            excluded["unmatched_title_pattern"] += 1
            continue
        prefix = m.group(1).upper()
        pid = m.group(2)
        tp = m.group(3).upper()

        # Strict policy: controls are D00 healthy/control samples; sepsis must be D00/admission.
        if prefix in {"CS", "PC"} and tp == "D00":
            included.append(
                SampleRecord(
                    dataset="GSE95233",
                    gsm_id=gsm_id,
                    sample_id=_mk_sample_id("GSE95233", gsm_id),
                    condition="control",
                    batch="GSE95233",
                    platform=gpl,
                    patient_id=f"GSE95233_{prefix}_{pid}",
                    timepoint=tp,
                    split_role="train",
                    include_reason="Healthy/control D00",
                )
            )
            continue

        is_admission = tp == "D00" or ("admission" in low)
        if prefix in {"SV", "NS"} and is_admission:
            included.append(
                SampleRecord(
                    dataset="GSE95233",
                    gsm_id=gsm_id,
                    sample_id=_mk_sample_id("GSE95233", gsm_id),
                    condition="sepsis",
                    batch="GSE95233",
                    platform=gpl,
                    patient_id=f"GSE95233_{prefix}_{pid}",
                    timepoint=tp,
                    split_role="train",
                    include_reason="Sepsis admission (D00/admission)",
                )
            )
            continue

        excluded["non_admission_or_non_target_group"] += 1

    logger.info("GSE95233 selected=%d excluded=%d", len(included), sum(excluded.values()))
    return included, excluded


def select_gse26378_holdout(gse, logger: logging.Logger) -> Tuple[List[SampleRecord], Counter]:
    included: List[SampleRecord] = []
    excluded = Counter()
    gpl = list(gse.gpls.keys())[0]
    for gsm_id, gsm in gse.gsms.items():
        title = gsm.metadata.get("title", [""])[0]
        low = title.lower()
        if low.startswith("septic shock"):
            condition = "sepsis"
        elif low.startswith("control"):
            condition = "control"
        else:
            excluded["non_target_title"] += 1
            continue

        included.append(
            SampleRecord(
                dataset="GSE26378",
                gsm_id=gsm_id,
                sample_id=_mk_sample_id("GSE26378", gsm_id),
                condition=condition,
                batch="GSE26378",
                platform=gpl,
                patient_id=f"GSE26378_{gsm_id}",
                timepoint="single",
                split_role="holdout",
                include_reason="Fixed external holdout cohort",
            )
        )
    logger.info("GSE26378 selected=%d excluded=%d", len(included), sum(excluded.values()))
    return included, excluded


def select_gse134347_fallback(gse, logger: logging.Logger) -> Tuple[List[SampleRecord], Counter]:
    included: List[SampleRecord] = []
    excluded = Counter()
    gpl = list(gse.gpls.keys())[0]
    for gsm_id, gsm in gse.gsms.items():
        title = gsm.metadata.get("title", [""])[0]
        chars = "; ".join(gsm.metadata.get("characteristics_ch1", []))
        low = chars.lower()
        m = re.search(r"disease state:\s*([^;]+)", low)
        disease = m.group(1).strip() if m else "unknown"
        if disease == "sepsis":
            condition = "sepsis"
        elif disease == "healthy":
            condition = "control"
        elif disease == "noninfectious":
            excluded["noninfectious_excluded"] += 1
            continue
        else:
            excluded["unknown_disease_state"] += 1
            continue
        pid = re.sub(r"\s+", "", str(title))
        included.append(
            SampleRecord(
                dataset="GSE134347",
                gsm_id=gsm_id,
                sample_id=_mk_sample_id("GSE134347", gsm_id),
                condition=condition,
                batch="GSE134347",
                platform=gpl,
                patient_id=f"GSE134347_{pid}",
                timepoint="single",
                split_role="train",
                include_reason="Fallback healthy+sepsis only",
            )
        )
    logger.info("GSE134347 selected=%d excluded=%d", len(included), sum(excluded.values()))
    return included, excluded


def summarize_records(records: List[SampleRecord], excluded: Counter) -> Dict[str, object]:
    cond = Counter(r.condition for r in records)
    time = Counter(r.timepoint for r in records)
    return {
        "included_samples": len(records),
        "condition_counts": dict(cond),
        "timepoint_counts": dict(time),
        "excluded_reasons": dict(excluded),
    }


def robust_combat(
    train_expr: pd.DataFrame,
    train_meta: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    logger.info("Running ComBat on train pool: genes=%d samples=%d", train_expr.shape[0], train_expr.shape[1])
    # pycombat expects genes x samples DataFrame.
    batch = train_meta.set_index("sample_id").loc[train_expr.columns, "batch"].astype(str).tolist()
    cond = train_meta.set_index("sample_id").loc[train_expr.columns, "condition"].astype(str)
    # combat.pycombat expects list-like covariates (DataFrame triggers an internal == [] bug).
    mod = cond.tolist()

    cleaned = train_expr.copy()
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.T.fillna(cleaned.T.mean()).T
    cleaned = cleaned.dropna(how="all")
    var = cleaned.var(axis=1)
    cleaned = cleaned.loc[var > 1e-12]

    corrected = pycombat(cleaned, batch=batch, mod=mod)
    if isinstance(corrected, np.ndarray):
        corrected = pd.DataFrame(corrected, index=cleaned.index, columns=cleaned.columns)
    corrected = corrected.astype(float)
    logger.info("ComBat output shape=%s", corrected.shape)
    return corrected


def compute_mad(df: pd.DataFrame) -> pd.Series:
    med = df.median(axis=1)
    return (df.sub(med, axis=0).abs()).median(axis=1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    log_file = Path(args.log_file).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = init_logger(log_file)
    logger.info("=== General_Sepsis_V11 Step 01: download_and_preprocess ===")
    logger.info("seed=%d output_dir=%s raw_dir=%s holdout=%s", args.seed, output_dir, raw_dir, args.holdout_dataset)

    # 1) Load required cohorts.
    required = ["GSE54514", "GSE57065", "GSE95233", args.holdout_dataset]
    if args.fallback_dataset not in required:
        required.append(args.fallback_dataset)
    gses = {gid: load_gse(gid, raw_dir, logger) for gid in required}

    # 2) Select samples per locked policy.
    sel_54514, ex_54514 = select_gse54514(gses["GSE54514"], logger)
    sel_57065, ex_57065 = select_gse57065(gses["GSE57065"], logger)
    sel_95233, ex_95233 = select_gse95233_strict(gses["GSE95233"], logger)
    sel_holdout, ex_holdout = select_gse26378_holdout(gses[args.holdout_dataset], logger)

    n_95233_sepsis = sum(1 for r in sel_95233 if r.condition == "sepsis")
    n_95233_control = sum(1 for r in sel_95233 if r.condition == "control")
    use_fallback = n_95233_sepsis < args.gse95233_min_sepsis or n_95233_control < args.gse95233_min_control

    if use_fallback:
        logger.warning(
            "GSE95233 QC failed (sepsis=%d control=%d). Triggering fallback dataset: %s",
            n_95233_sepsis,
            n_95233_control,
            args.fallback_dataset,
        )
        sel_fallback, ex_fallback = select_gse134347_fallback(gses[args.fallback_dataset], logger)
        active_train_records = sel_54514 + sel_57065 + sel_fallback
    else:
        sel_fallback, ex_fallback = [], Counter()
        active_train_records = sel_54514 + sel_57065 + sel_95233

    all_records = active_train_records + sel_holdout
    if not all_records:
        raise RuntimeError("No samples selected after policy filtering.")

    # 3) Build per-dataset expression matrices.
    records_by_dataset: Dict[str, List[SampleRecord]] = defaultdict(list)
    for r in all_records:
        records_by_dataset[r.dataset].append(r)

    dataset_expr: Dict[str, pd.DataFrame] = {}
    dataset_platform: Dict[str, str] = {}
    for dataset, recs in records_by_dataset.items():
        gse = gses[dataset]
        probe_to_gene, gpl = build_probe_to_gene_map(gse, dataset, logger)
        selected_gsms = [r.gsm_id for r in recs]
        sample_id_map = {r.gsm_id: r.sample_id for r in recs}
        expr = extract_expression_matrix(gse, dataset, selected_gsms, probe_to_gene, sample_id_map, logger)
        dataset_expr[dataset] = expr
        dataset_platform[dataset] = gpl

    # 4) Assemble metadata.
    meta_rows = []
    for r in all_records:
        meta_rows.append(
            {
                "sample_id": r.sample_id,
                "condition": r.condition,
                "batch": r.batch,
                "platform": r.platform,
                "dataset": r.dataset,
                "patient_id": r.patient_id,
                "timepoint": r.timepoint,
                "split_role": r.split_role,
            }
        )
    metadata = pd.DataFrame(meta_rows)
    if metadata["sample_id"].duplicated().any():
        dup = metadata.loc[metadata["sample_id"].duplicated(), "sample_id"].tolist()
        raise RuntimeError(f"Duplicate sample_id detected: {dup[:5]}")

    # 5) Train pool harmonization (non-holdout only).
    train_records = [r for r in all_records if r.split_role == "train"]
    holdout_records = [r for r in all_records if r.split_role == "holdout"]
    train_ids = [r.sample_id for r in train_records]
    holdout_ids = [r.sample_id for r in holdout_records]

    train_datasets = sorted(set(r.dataset for r in train_records))
    holdout_dataset = sorted(set(r.dataset for r in holdout_records))
    logger.info("Train datasets=%s | Holdout datasets=%s", train_datasets, holdout_dataset)

    train_gene_sets = [set(dataset_expr[d].index) for d in train_datasets]
    common_train_genes = sorted(set.intersection(*train_gene_sets))
    logger.info("Common train genes across active train cohorts: %d", len(common_train_genes))
    if len(common_train_genes) < 1000:
        raise RuntimeError(f"Too few common genes in train pool: {len(common_train_genes)}")

    train_blocks = []
    for d in train_datasets:
        cols = [r.sample_id for r in train_records if r.dataset == d]
        block = dataset_expr[d].reindex(common_train_genes).loc[:, cols]
        train_blocks.append(block)
    train_matrix = pd.concat(train_blocks, axis=1)
    train_matrix = train_matrix.loc[:, train_ids]
    logger.info("Train matrix shape before ComBat: %s", train_matrix.shape)

    train_meta = metadata[metadata["sample_id"].isin(train_ids)].copy()
    combat_train = robust_combat(train_matrix, train_meta, logger)

    mad = compute_mad(combat_train).sort_values(ascending=False)
    k = min(args.top_k_genes, mad.shape[0])
    selected_genes = mad.head(k).index.tolist()
    logger.info("Selected top genes by MAD: %d", len(selected_genes))

    train_raw_top = train_matrix.reindex(selected_genes)
    train_raw_top = train_raw_top.apply(pd.to_numeric, errors="coerce")
    train_raw_top = train_raw_top.T.fillna(train_raw_top.T.mean()).T

    combat_train_top = combat_train.reindex(selected_genes)
    train_mean = combat_train_top.mean(axis=1)
    train_std = combat_train_top.std(axis=1).replace(0, 1.0).fillna(1.0)
    train_std_expr = combat_train_top.sub(train_mean, axis=0).div(train_std, axis=0)

    # 6) Transform holdout to train gene list and train normalization params only.
    holdout_dataset_name = args.holdout_dataset
    holdout_expr_raw = dataset_expr[holdout_dataset_name].reindex(selected_genes)
    holdout_expr_raw = holdout_expr_raw.loc[:, holdout_ids]
    holdout_expr_raw = holdout_expr_raw.apply(pd.to_numeric, errors="coerce")
    holdout_expr_raw = holdout_expr_raw.T.fillna(train_mean).T
    holdout_std_expr = holdout_expr_raw.sub(train_mean, axis=0).div(train_std, axis=0)

    raw_train_mean = train_raw_top.mean(axis=1)
    holdout_raw_selected = dataset_expr[holdout_dataset_name].reindex(selected_genes)
    holdout_raw_selected = holdout_raw_selected.loc[:, holdout_ids]
    holdout_raw_selected = holdout_raw_selected.apply(pd.to_numeric, errors="coerce")
    holdout_raw_selected = holdout_raw_selected.T.fillna(raw_train_mean).T
    raw_selected_expr = pd.concat([train_raw_top, holdout_raw_selected], axis=1)
    raw_selected_expr = raw_selected_expr.loc[:, train_ids + holdout_ids]

    # 7) Final matrix + metadata export.
    final_expr = pd.concat([train_std_expr, holdout_std_expr], axis=1)
    final_expr = final_expr.loc[:, train_ids + holdout_ids]
    meta_out = metadata.set_index("sample_id").loc[final_expr.columns].reset_index()
    if "index" in meta_out.columns and "sample_id" not in meta_out.columns:
        meta_out = meta_out.rename(columns={"index": "sample_id"})

    expr_path = output_dir / "expression_combat.csv"
    raw_expr_path = output_dir / "expression_raw_selected.csv"
    meta_path = output_dir / "metadata.csv"
    genes_path = output_dir / "gene_list.json"
    manifest_path = output_dir / "cohort_manifest.json"

    final_expr.to_csv(expr_path)
    raw_selected_expr.to_csv(raw_expr_path)
    meta_out.to_csv(meta_path, index=False)
    with genes_path.open("w", encoding="utf-8") as f:
        json.dump(selected_genes, f, indent=2)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "seed": args.seed,
        "policy": {
            "scope": "sepsis_vs_healthy_only",
            "holdout_dataset": args.holdout_dataset,
            "fallback_dataset": args.fallback_dataset,
            "gse95233_qc_min": {
                "sepsis": args.gse95233_min_sepsis,
                "control": args.gse95233_min_control,
            },
            "gse95233_fallback_triggered": use_fallback,
        },
        "datasets": {
            "GSE54514": summarize_records(sel_54514, ex_54514),
            "GSE57065": summarize_records(sel_57065, ex_57065),
            "GSE95233": summarize_records(sel_95233, ex_95233),
            "GSE26378": summarize_records(sel_holdout, ex_holdout),
            args.fallback_dataset: summarize_records(sel_fallback, ex_fallback),
        },
        "active_train_datasets": train_datasets,
        "active_holdout_dataset": args.holdout_dataset,
        "n_train_samples": len(train_ids),
        "n_holdout_samples": len(holdout_ids),
        "n_selected_genes": len(selected_genes),
        "artifacts": {
            "expression_combat_csv": str(expr_path),
            "expression_raw_selected_csv": str(raw_expr_path),
            "metadata_csv": str(meta_path),
            "gene_list_json": str(genes_path),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Saved: %s", expr_path)
    logger.info("Saved: %s", raw_expr_path)
    logger.info("Saved: %s", meta_path)
    logger.info("Saved: %s", genes_path)
    logger.info("Saved: %s", manifest_path)
    logger.info(
        "Completed Step 01 with final matrix genes=%d samples=%d (train=%d holdout=%d)",
        final_expr.shape[0],
        final_expr.shape[1],
        len(train_ids),
        len(holdout_ids),
    )


if __name__ == "__main__":
    main()
