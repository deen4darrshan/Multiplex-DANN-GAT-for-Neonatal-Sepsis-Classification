"""
Module B - Task B.1: ID Mapping - All SOFT Files
Process all three datasets now that they're in SOFT format.
"""

import GEOparse
import pandas as pd
import numpy as np
import os

# Paths
DATA_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

def process_gse(geo_id, filepath):
    """Load GSE and extract expression matrix + phenotype."""
    print(f"\n{'='*60}")
    print(f"PROCESSING {geo_id}")
    print(f"{'='*60}")
    
    gse = GEOparse.get_GEO(filepath=filepath)
    
    samples = list(gse.gsms.keys())
    print(f"Number of samples: {len(samples)}")
    
    # Build expression matrix
    expr_data = {}
    for sample_name, gsm in gse.gsms.items():
        table = gsm.table
        if len(table) == 0:
            continue
        if 'VALUE' in table.columns:
            expr_data[sample_name] = table.set_index('ID_REF')['VALUE']
    
    expr_df = pd.DataFrame(expr_data)
    expr_df = expr_df.apply(pd.to_numeric, errors='coerce')
    print(f"Raw expression matrix: {expr_df.shape}")
    
    # Get platform and map probes to genes
    platforms = list(gse.gpls.keys())
    if not platforms:
        print("No platform found!")
        return None, None
    
    gpl = gse.gpls[platforms[0]]
    print(f"Platform: {platforms[0]}")
    
    annot = gpl.table
    
    # Find gene symbol column
    gene_col = None
    for col in ['Gene Symbol', 'gene_assignment', 'GENE_SYMBOL', 'Symbol', 'symbol', 'ORF']:
        if col in annot.columns:
            gene_col = col
            break
    
    if gene_col is None:
        for col in annot.columns:
            if 'gene' in col.lower() and 'symbol' in col.lower():
                gene_col = col
                break
    
    if gene_col is None:
        print(f"Available columns: {annot.columns.tolist()}")
        print("Could not find gene symbol column!")
        return None, None
    
    print(f"Using column '{gene_col}' for gene symbols")
    
    # Create mapping
    id_col = 'ID' if 'ID' in annot.columns else annot.columns[0]
    annot_sub = annot[[id_col, gene_col]].copy()
    annot_sub = annot_sub.dropna()
    annot_sub = annot_sub[annot_sub[gene_col].astype(str).str.strip() != '']
    annot_sub = annot_sub[annot_sub[gene_col].astype(str) != '---']
    
    # Handle gene_assignment format
    if gene_col == 'gene_assignment':
        def extract_symbol(x):
            if pd.isna(x) or x == '---':
                return None
            parts = str(x).split('//')
            if len(parts) >= 2:
                return parts[1].strip()
            return None
        annot_sub[gene_col] = annot_sub[gene_col].apply(extract_symbol)
        annot_sub = annot_sub.dropna()
    
    probe_to_gene = dict(zip(annot_sub[id_col].astype(str), annot_sub[gene_col].astype(str)))
    print(f"Probe-to-gene mapping: {len(probe_to_gene)} entries")
    
    # Apply mapping
    expr_df = expr_df.copy()
    expr_df.index = expr_df.index.astype(str).map(lambda x: probe_to_gene.get(x, None))
    expr_df = expr_df[expr_df.index.notna()]
    
    # Handle multiple genes
    def get_first_gene(x):
        if isinstance(x, str) and '///' in x:
            return x.split('///')[0].strip()
        return x
    expr_df.index = expr_df.index.map(get_first_gene)
    
    # Average duplicates
    expr_df = expr_df.groupby(expr_df.index).mean()
    print(f"After mapping: {expr_df.shape}")
    
    # Get phenotype
    pheno = {}
    for sample_name, gsm in gse.gsms.items():
        characteristics = gsm.metadata.get('characteristics_ch1', [])
        pheno[sample_name] = {
            'title': gsm.metadata.get('title', [''])[0],
            'source': gsm.metadata.get('source_name_ch1', [''])[0],
            'characteristics': '; '.join(characteristics) if characteristics else ''
        }
    pheno_df = pd.DataFrame(pheno).T
    
    return expr_df, pheno_df

def main():
    datasets = {
        'GSE25504': 'GSE25504_family.soft.gz',
        'GSE69686': 'GSE69686_family.soft.gz',
        'GSE26440': 'GSE26440_family.soft.gz'
    }
    
    results = {}
    
    for geo_id, filename in datasets.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        
        expr_df, pheno_df = process_gse(geo_id, filepath)
        
        if expr_df is not None and len(expr_df) > 0:
            expr_df.to_csv(os.path.join(OUT_DIR, f"{geo_id}_mapped.csv"))
            pheno_df.to_csv(os.path.join(OUT_DIR, f"{geo_id}_phenotype.csv"))
            results[geo_id] = expr_df
            print(f"Saved {geo_id}: {expr_df.shape}")
    
    # Summary
    print("\n" + "="*60)
    print("=== FINAL SUMMARY ===")
    print("="*60)
    for geo_id, df in results.items():
        print(f"{geo_id}: {df.shape[0]} genes x {df.shape[1]} samples")
    
    # Common genes check
    if 'GSE25504' in results and 'GSE69686' in results:
        common = set(results['GSE25504'].index) & set(results['GSE69686'].index)
        print(f"\nCommon genes (GSE25504 ∩ GSE69686): {len(common)}")
        
        if len(common) >= 15000:
            print("✓ Sufficient common genes for merging!")
        else:
            print("⚠ Warning: Less than 15,000 common genes.")

if __name__ == "__main__":
    main()
