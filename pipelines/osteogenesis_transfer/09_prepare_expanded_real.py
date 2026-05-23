import os
import gzip
import re
import requests
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW = os.path.join(ROOT, 'data', 'raw')
PROC = os.path.join(ROOT, 'data', 'processed')
DATASET_DIR = os.path.join(PROC, 'datasets')
EXP_DIR = os.path.join(PROC, 'expanded_datasets')
os.makedirs(RAW, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

MOUSE_XLSX = os.path.join(RAW, 'GSE154748_ALL_FPKM.txt.gz')
MOUSE_MATRIX = os.path.join(RAW, 'GSE154748_series_matrix.txt.gz')


def clean_symbol(x):
    s = str(x).strip().upper()
    if s in {'', 'NA', 'NAN', '--', '-', 'NONE'}:
        return None
    s = s.split('///')[0].split(';')[0].strip()
    return s if s else None


def postprocess_expr(df, symbol_col):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df[symbol_col] = df[symbol_col].apply(clean_symbol)
    df = df[df[symbol_col].notna()]
    keep = [c for c in df.columns if c != symbol_col]
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.groupby(symbol_col, as_index=True).mean(numeric_only=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how='all')
    df = df.T.fillna(df.T.mean()).T
    # already expression-like scale; keep as-is
    return df


def ensure_mouse_files():
    if not os.path.exists(MOUSE_XLSX):
        u = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154748/suppl/GSE154748_ALL_FPKM.txt.gz'
        r = requests.get(u, timeout=120)
        r.raise_for_status()
        with open(MOUSE_XLSX, 'wb') as f:
            f.write(r.content)
    if not os.path.exists(MOUSE_MATRIX):
        u = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE154nnn/GSE154748/matrix/GSE154748_series_matrix.txt.gz'
        r = requests.get(u, timeout=120)
        r.raise_for_status()
        with open(MOUSE_MATRIX, 'wb') as f:
            f.write(r.content)


def parse_mouse_metadata():
    with gzip.open(MOUSE_MATRIX, 'rt') as f:
        lines = f.read().splitlines()

    # extract per-sample characteristics (order-aligned)
    titles = []
    for ln in lines:
        if ln.startswith('!Sample_title'):
            titles = [x.strip().strip('"') for x in ln.split('\t')[1:]]
            break

    chars_lines = [ln for ln in lines if ln.startswith('!Sample_characteristics_ch1')]
    chars = [[x.strip().strip('"') for x in ln.split('\t')[1:]] for ln in chars_lines]

    n = len(titles)
    rows = []
    for i in range(n):
        genotype = ''
        mouse_id = ''
        for c in chars:
            v = c[i] if i < len(c) else ''
            if v.lower().startswith('genotype:'):
                genotype = v.split(':', 1)[1].strip()
            if v.lower().startswith('mouse id:'):
                mouse_id = v.split(':', 1)[1].strip()

        cond = 'Control' if 'wildtype' in genotype.lower() else 'OI'
        rows.append({'title': titles[i], 'mouse_id': mouse_id, 'genotype': genotype, 'Condition': cond})
    return pd.DataFrame(rows)


def load_mouse_dataset():
    ensure_mouse_files()

    df = pd.read_csv(MOUSE_XLSX, sep='\t', compression='gzip')
    meta = parse_mouse_metadata()

    expr = postprocess_expr(df[['Feature'] + [c for c in df.columns if c.startswith('SZ')]], 'Feature')

    # Map sample columns SZ29 -> title/meta by mouse_id
    id_to_cond = {m['mouse_id'].replace('SZ', 'SZ').zfill(0): m['Condition'] for _, m in meta.iterrows()}

    # robust mapping: match numeric part ignoring leading zeros
    mapped = {}
    new_meta = []
    for c in expr.columns:
        num = re.sub(r'[^0-9]', '', c)
        cond = None
        for _, m in meta.iterrows():
            mn = re.sub(r'[^0-9]', '', str(m['mouse_id']))
            if num == mn:
                cond = m['Condition']
                break
        if cond is None:
            continue

        sid = f'GSE154748|{c}'
        mapped[c] = sid
        new_meta.append({
            'SampleID': sid,
            'Dataset': 'GSE154748',
            'Batch': 'GSE154748',
            'Condition': cond,
            'Label': 1 if cond == 'OI' else 0,
            'GroupID': sid,
            'Species': 'mouse',
        })

    expr = expr.rename(columns=mapped)
    expr = expr[[m['SampleID'] for m in new_meta if m['SampleID'] in expr.columns]]

    return expr, pd.DataFrame(new_meta)


def main():
    # Start from existing human multicohort
    human_meta = pd.read_csv(os.path.join(PROC, 'multicohort_metadata.csv'))
    human_meta['Species'] = 'human'

    human_expr = {}
    for ds in sorted(human_meta['Dataset'].unique()):
        human_expr[ds] = pd.read_csv(os.path.join(DATASET_DIR, f'{ds}_expr.csv'), index_col=0)

    mouse_expr, mouse_meta = load_mouse_dataset()

    # Save per-dataset files
    for ds, ex in human_expr.items():
        ex.to_csv(os.path.join(EXP_DIR, f'{ds}_expr.csv'))
        human_meta[human_meta['Dataset'] == ds].to_csv(os.path.join(EXP_DIR, f'{ds}_meta.csv'), index=False)

    mouse_expr.to_csv(os.path.join(EXP_DIR, 'GSE154748_expr.csv'))
    mouse_meta.to_csv(os.path.join(EXP_DIR, 'GSE154748_meta.csv'), index=False)

    # Build combined with common genes
    all_expr = list(human_expr.values()) + [mouse_expr]
    common = set(all_expr[0].index)
    for ex in all_expr[1:]:
        common &= set(ex.index)
    common = sorted(common)

    combined = pd.concat([ex.loc[common] for ex in all_expr], axis=1)
    combined.to_csv(os.path.join(PROC, 'expanded_expression_common.csv'))

    all_meta = pd.concat([human_meta, mouse_meta], ignore_index=True)
    all_meta.to_csv(os.path.join(PROC, 'expanded_metadata.csv'), index=False)

    print(f'Expanded samples: {len(all_meta)}')
    print(all_meta.groupby(['Dataset', 'Condition']).size().to_string())
    print(f'Common genes across expanded cohorts: {len(common)}')


if __name__ == '__main__':
    main()
