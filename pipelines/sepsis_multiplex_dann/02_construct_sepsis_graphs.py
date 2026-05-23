#!/usr/bin/env python3
"""
General_Sepsis_V11 - Step 02
Build static KEGG + STRING graph artifacts from selected gene list.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    today = datetime.now().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Build KEGG+STRING static graph artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(root / "results"))
    parser.add_argument("--log-file", default=str(root / "logs" / f"{today}_02_build_graphs.log"))
    parser.add_argument("--expression-path", default=str(root / "results" / "expression_combat.csv"))
    parser.add_argument("--metadata-path", default=str(root / "results" / "metadata.csv"))
    parser.add_argument("--gene-list-path", default=str(root / "results" / "gene_list.json"))
    parser.add_argument(
        "--string-file",
        default=str(Path(__file__).resolve().parents[2] / "data" / "raw" / "9606.protein.links.v12.0.txt.gz"),
    )
    parser.add_argument("--kegg-library", default="KEGG_2021_Human")
    parser.add_argument("--kegg-min-overlap", type=int, default=3)
    parser.add_argument("--string-threshold", type=int, default=700)
    parser.add_argument("--max-string-edges", type=int, default=250000)
    parser.add_argument("--mygene-chunk-size", type=int, default=1000)
    return parser.parse_args()


def init_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("general_sepsis_v11_graphs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_gene_list(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "genes" in data:
        return list(data["genes"])
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"Unsupported gene_list.json structure: {path}")


def safe_coexpression_rankcorr(
    expr: pd.DataFrame,
    threshold: float = 0.7,
    max_edges: int = 60000,
) -> Dict[str, object]:
    """
    SciPy-deadlock-safe co-expression utility:
    1) rank with pandas
    2) correlation with NumPy
    Returns edge pairs by absolute correlation.
    """
    ranked = expr.rank(axis=1, method="average")
    arr = ranked.values.astype(np.float32)
    corr = np.corrcoef(arr)
    tri_i, tri_j = np.triu_indices(corr.shape[0], k=1)
    tri_v = np.abs(corr[tri_i, tri_j])
    keep = tri_v >= threshold
    i = tri_i[keep]
    j = tri_j[keep]
    v = tri_v[keep]
    if max_edges > 0 and i.shape[0] > max_edges:
        top = np.argpartition(-v, max_edges - 1)[:max_edges]
        i = i[top]
        j = j[top]
        v = v[top]
    return {
        "threshold": threshold,
        "n_edges": int(i.shape[0]),
        "example_edges": [[int(a), int(b), float(c)] for a, b, c in zip(i[:10], j[:10], v[:10])],
    }


def load_kegg_pathways(
    genes: Sequence[str],
    library: str,
    min_overlap: int,
    logger: logging.Logger,
) -> List[Dict[str, object]]:
    try:
        import gseapy as gp
    except Exception as exc:  # pragma: no cover
        logger.warning("gseapy unavailable; KEGG relation empty (%s)", exc)
        return []

    gene_set = set(genes)
    pathways: List[Dict[str, object]] = []
    logger.info("Loading KEGG library=%s", library)
    try:
        kegg_dict = gp.get_library(name=library, organism="Human")
    except Exception as exc:
        logger.warning("KEGG load failed; proceeding with empty KEGG (%s)", exc)
        return []

    for name, members in kegg_dict.items():
        overlap = sorted(gene_set.intersection({m.upper() for m in members}))
        if len(overlap) < min_overlap:
            continue
        pathways.append({"pathway": name, "genes": overlap, "size": len(overlap)})

    logger.info("KEGG retained pathways=%d (min_overlap=%d)", len(pathways), min_overlap)
    return pathways


def build_symbol_to_proteins(
    genes: Sequence[str],
    chunk_size: int,
    logger: logging.Logger,
) -> Dict[str, Set[str]]:
    try:
        import mygene
    except Exception as exc:
        logger.warning("mygene unavailable; STRING relation empty (%s)", exc)
        return {}

    mg = mygene.MyGeneInfo()
    out: Dict[str, Set[str]] = {g: set() for g in genes}

    genes_list = list(genes)
    for i in range(0, len(genes_list), chunk_size):
        chunk = genes_list[i : i + chunk_size]
        logger.info("mygene map chunk %d-%d / %d", i + 1, min(i + chunk_size, len(genes_list)), len(genes_list))
        res = mg.querymany(
            chunk,
            scopes="symbol",
            fields="ensembl.protein,symbol",
            species="human",
            as_dataframe=False,
            verbose=False,
            returnall=False,
        )
        for row in res:
            if row.get("notfound"):
                continue
            symbol = str(row.get("query", "")).upper()
            if symbol not in out:
                continue
            ens = row.get("ensembl")
            if not ens:
                continue
            if isinstance(ens, dict):
                prot = ens.get("protein")
                if isinstance(prot, list):
                    out[symbol].update(str(p).replace("9606.", "").strip() for p in prot if p)
                elif prot:
                    out[symbol].add(str(prot).replace("9606.", "").strip())
            elif isinstance(ens, list):
                for item in ens:
                    if isinstance(item, dict):
                        prot = item.get("protein")
                        if isinstance(prot, list):
                            out[symbol].update(str(p).replace("9606.", "").strip() for p in prot if p)
                        elif prot:
                            out[symbol].add(str(prot).replace("9606.", "").strip())

    non_empty = sum(1 for v in out.values() if v)
    logger.info("mygene mapped symbols with protein IDs: %d/%d", non_empty, len(genes_list))
    return out


def invert_symbol_protein_map(symbol_to_prot: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    prot_to_sym: Dict[str, Set[str]] = {}
    for sym, prots in symbol_to_prot.items():
        for p in prots:
            prot_to_sym.setdefault(p, set()).add(sym)
    return prot_to_sym


def stream_string_edges(
    string_file: Path,
    genes: Sequence[str],
    symbol_to_proteins: Dict[str, Set[str]],
    threshold: int,
    max_edges: int,
    logger: logging.Logger,
) -> List[List[object]]:
    if not string_file.exists():
        logger.warning("STRING file not found: %s", string_file)
        return []

    prot_to_sym = invert_symbol_protein_map(symbol_to_proteins)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    edges: Dict[Tuple[int, int], int] = {}

    logger.info(
        "Streaming STRING edges from %s with threshold >= %d",
        string_file,
        threshold,
    )
    with gzip.open(string_file, "rt", encoding="utf-8") as f:
        header = next(f, None)
        if header is None:
            return []
        scanned = 0
        kept_raw = 0
        for line in f:
            scanned += 1
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            p1 = parts[0].replace("9606.", "")
            p2 = parts[1].replace("9606.", "")
            try:
                score = int(parts[2])
            except ValueError:
                continue
            if score < threshold:
                continue
            syms1 = prot_to_sym.get(p1)
            syms2 = prot_to_sym.get(p2)
            if not syms1 or not syms2:
                continue
            kept_raw += 1
            for s1 in syms1:
                i = gene_to_idx.get(s1)
                if i is None:
                    continue
                for s2 in syms2:
                    j = gene_to_idx.get(s2)
                    if j is None or i == j:
                        continue
                    a, b = (i, j) if i < j else (j, i)
                    prev = edges.get((a, b), 0)
                    if score > prev:
                        edges[(a, b)] = score
            if max_edges > 0 and len(edges) >= max_edges:
                break

            if scanned % 2_000_000 == 0:
                logger.info(
                    "STRING scanned=%d kept_raw=%d unique_gene_edges=%d",
                    scanned,
                    kept_raw,
                    len(edges),
                )

    out = [[int(a), int(b), int(score)] for (a, b), score in sorted(edges.items(), key=lambda kv: kv[1], reverse=True)]
    if max_edges > 0 and len(out) > max_edges:
        out = out[:max_edges]
    logger.info("STRING retained unique gene edges=%d", len(out))
    return out


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = init_logger(Path(args.log_file).resolve())
    logger.info("=== General_Sepsis_V11 Step 02: build_graphs ===")

    expr = pd.read_csv(args.expression_path, index_col=0)
    metadata = pd.read_csv(args.metadata_path)
    if "sample_id" not in metadata.columns and "index" in metadata.columns:
        metadata = metadata.rename(columns={"index": "sample_id"})
    genes = load_gene_list(Path(args.gene_list_path))
    logger.info("Input expression shape=%s metadata_rows=%d genes=%d", expr.shape, metadata.shape[0], len(genes))

    # Keep exact order from gene list.
    genes = [g for g in genes if g in expr.index]
    if len(genes) == 0:
        raise RuntimeError("No genes from gene_list found in expression matrix.")
    expr = expr.loc[genes]

    pathways = load_kegg_pathways(genes, args.kegg_library, args.kegg_min_overlap, logger)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    pathway_records = []
    kegg_covered: Set[int] = set()
    for p in pathways:
        idxs = sorted(gene_to_idx[g] for g in p["genes"] if g in gene_to_idx)
        if len(idxs) < args.kegg_min_overlap:
            continue
        pathway_records.append(
            {
                "pathway": p["pathway"],
                "gene_indices": idxs,
                "genes": p["genes"],
                "size": len(idxs),
            }
        )
        kegg_covered.update(idxs)

    symbol_to_proteins = build_symbol_to_proteins(genes, args.mygene_chunk_size, logger)
    string_edges = stream_string_edges(
        string_file=Path(args.string_file),
        genes=genes,
        symbol_to_proteins=symbol_to_proteins,
        threshold=args.string_threshold,
        max_edges=args.max_string_edges,
        logger=logger,
    )
    string_covered = set()
    for i, j, _ in string_edges:
        string_covered.add(int(i))
        string_covered.add(int(j))

    # Utility preview to certify fold-safe co-expression function exists and works.
    train_cols = metadata.loc[metadata["split_role"] == "train", "sample_id"].tolist()
    coexpr_preview = safe_coexpression_rankcorr(expr.loc[:, train_cols], threshold=0.7, max_edges=2000)

    coverage = len(kegg_covered.union(string_covered)) / max(1, len(genes))
    qc = {
        "n_genes": len(genes),
        "n_pathways_retained": len(pathway_records),
        "n_string_edges_retained": len(string_edges),
        "kegg_gene_coverage_pct": float(100.0 * len(kegg_covered) / max(1, len(genes))),
        "string_gene_coverage_pct": float(100.0 * len(string_covered) / max(1, len(genes))),
        "combined_gene_coverage_pct": float(100.0 * coverage),
    }

    payload = {
        "generated_at": datetime.now().isoformat(),
        "seed": args.seed,
        "genes": genes,
        "kegg": {
            "library": args.kegg_library,
            "min_overlap": args.kegg_min_overlap,
            "pathways": pathway_records,
        },
        "string": {
            "source_file": str(Path(args.string_file).resolve()),
            "threshold": args.string_threshold,
            "edges": string_edges,
            "mapping_symbols_with_proteins": int(sum(1 for v in symbol_to_proteins.values() if v)),
        },
        "coexpression_runtime_note": {
            "persisted_static_artifact": False,
            "note": "Co-expression is computed per fold from fold-train samples only in training script.",
            "safe_rankcorr_preview": coexpr_preview,
        },
        "qc": qc,
    }

    out_path = output_dir / "pathway_info.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved %s", out_path)
    logger.info("Graph QC: %s", qc)


if __name__ == "__main__":
    main()
