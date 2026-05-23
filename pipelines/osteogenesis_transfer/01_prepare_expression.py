import os
import gzip
import io
import pandas as pd
import numpy as np
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
PROC_DIR = os.path.join(ROOT, 'data', 'processed')
os.makedirs(PROC_DIR, exist_ok=True)

GSE160207_COUNTS = os.path.join(RAW_DIR, 'GSE160207_EE_OI_RNAseq_counts.txt.gz')
GSE163812_COUNTS = os.path.join(RAW_DIR, 'GSE163812_ESAT_counts.txt.gz')
GSE163812_MATRIX = os.path.join(RAW_DIR, 'GSE163812_series_matrix.txt.gz')


def load_gse160207():
    print('[GSE160207] loading counts')
    with gzip.open(GSE160207_COUNTS, 'rt') as f:
        df = pd.read_csv(f, sep='\t')

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    if 'symbol' not in df.columns:
        raise ValueError('symbol column not found in GSE160207 counts file')

    df = df.rename(columns={'symbol': 'Symbol'})
    # Keep gene symbols and sample columns
    sample_cols = [c for c in df.columns if c.startswith('OI-') or c.startswith('C-')]
    df = df[['Symbol'] + sample_cols].copy()

    # Clean symbols
    df['Symbol'] = df['Symbol'].astype(str).str.strip()
    df = df[df['Symbol'] != '']
    df = df[df['Symbol'].notna()]

    # Uppercase for consistency
    df['Symbol'] = df['Symbol'].str.upper()

    # Aggregate duplicates
    df = df.groupby('Symbol', as_index=True).mean(numeric_only=True)

    # Build metadata
    meta = []
    for col in sample_cols:
        condition = 'OI' if col.startswith('OI-') else 'Control'
        meta.append({
            'SampleID': col,
            'Batch': 'GSE160207',
            'Condition': condition,
            'Label': 1 if condition == 'OI' else 0,
            'Treatment': 'None'
        })
    meta_df = pd.DataFrame(meta)

    return df, meta_df


def parse_gse163812_metadata():
    print('[GSE163812] parsing metadata')
    with gzip.open(GSE163812_MATRIX, 'rt') as f:
        text = f.read()

    lines = text.splitlines()
    fields = {}
    for line in lines:
        if line.startswith('!Sample_'):
            parts = line.split('\t')
            fields[parts[0]] = parts[1:]

    sources = fields.get('!Sample_source_name_ch1', [])
    titles = fields.get('!Sample_title', [])
    accessions = fields.get('!Sample_geo_accession', [])

    # collect per-sample characteristics
    per_sample_chars = {i: [] for i in range(len(sources))}
    for line in lines:
        if line.startswith('!Sample_characteristics_ch1'):
            vals = line.split('\t')[1:]
            for i, v in enumerate(vals):
                per_sample_chars[i].append(v)

    meta = []
    for i, src in enumerate(sources):
        src_clean = src.strip().strip('"')
        chars = ' ; '.join([c.strip('"') for c in per_sample_chars.get(i, [])])
        chars_lower = chars.lower()

        if 'patient donor' in chars_lower:
            condition = 'OI'
        elif 'healthy donor' in chars_lower:
            condition = 'Control'
        else:
            condition = 'Unknown'

        if 'xbp1s' in chars_lower:
            treatment = 'XBP1s'
        elif 'gfp' in chars_lower:
            treatment = 'GFP'
        else:
            treatment = 'Unknown'

        meta.append({
            'SampleID': src_clean,
            'Batch': 'GSE163812',
            'Condition': condition,
            'Label': 1 if condition == 'OI' else 0,
            'Treatment': treatment,
            'GSM': accessions[i] if i < len(accessions) else '',
            'Title': titles[i] if i < len(titles) else ''
        })

    meta_df = pd.DataFrame(meta)
    return meta_df


def load_gse163812(meta_df):
    print('[GSE163812] loading counts')
    with gzip.open(GSE163812_COUNTS, 'rt') as f:
        df = pd.read_csv(f, sep='\t')

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    if 'Symbol' not in df.columns:
        raise ValueError('Symbol column not found in GSE163812 counts file')

    # Keep gene symbols and sample columns that exist in metadata
    sample_cols = [c for c in df.columns if c in set(meta_df['SampleID'])]
    df = df[['Symbol'] + sample_cols].copy()

    # Clean symbols
    df['Symbol'] = df['Symbol'].astype(str).str.strip()
    df = df[df['Symbol'] != '']
    df = df[df['Symbol'].notna()]

    # Uppercase for consistency
    df['Symbol'] = df['Symbol'].str.upper()

    # Aggregate duplicates
    df = df.groupby('Symbol', as_index=True).mean(numeric_only=True)

    # Filter metadata to only samples present
    meta_df = meta_df[meta_df['SampleID'].isin(sample_cols)].copy()

    return df, meta_df


def main():
    if not os.path.exists(GSE160207_COUNTS) or not os.path.exists(GSE163812_COUNTS):
        raise FileNotFoundError('Missing raw files. Run scripts/00_download_data.py first.')

    # Load datasets
    expr160, meta160 = load_gse160207()
    meta163 = parse_gse163812_metadata()
    expr163, meta163 = load_gse163812(meta163)

    # Filter unknowns in GSE163812
    meta163 = meta163[meta163['Condition'] != 'Unknown'].copy()
    expr163 = expr163[meta163['SampleID'].tolist()]

    # Intersect genes
    common = sorted(list(set(expr160.index) & set(expr163.index)))
    print(f'Common genes: {len(common)}')

    expr160 = expr160.loc[common]
    expr163 = expr163.loc[common]

    # Combine
    combined = pd.concat([expr160, expr163], axis=1)

    # Log2 transform (counts + 1)
    combined = np.log2(combined + 1.0)

    # Metadata
    meta = pd.concat([meta160, meta163], axis=0, ignore_index=True)

    # Save
    expr160.to_csv(os.path.join(PROC_DIR, 'GSE160207_expr.csv'))
    meta160.to_csv(os.path.join(PROC_DIR, 'GSE160207_meta.csv'), index=False)
    expr163.to_csv(os.path.join(PROC_DIR, 'GSE163812_expr.csv'))
    meta163.to_csv(os.path.join(PROC_DIR, 'GSE163812_meta.csv'), index=False)

    combined.to_csv(os.path.join(PROC_DIR, 'combined_expression_log2.csv'))
    meta.to_csv(os.path.join(PROC_DIR, 'combined_metadata.csv'), index=False)

    print('Saved:')
    print(' - combined_expression_log2.csv')
    print(' - combined_metadata.csv')
    print(' - GSE160207_expr.csv / GSE160207_meta.csv')
    print(' - GSE163812_expr.csv / GSE163812_meta.csv')


if __name__ == '__main__':
    main()
