import os
import re
import gzip
import requests
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
PROC_DIR = os.path.join(ROOT, 'data', 'processed')
DATASET_DIR = os.path.join(PROC_DIR, 'datasets')
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

URLS = {
    'GSE160207_counts': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE160nnn/GSE160207/suppl/GSE160207_EE_OI_RNAseq_counts.txt.gz',
    'GSE160207_matrix': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE160nnn/GSE160207/matrix/GSE160207_series_matrix.txt.gz',
    'GSE163812_counts': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE163nnn/GSE163812/suppl/GSE163812_ESAT_counts.txt.gz',
    'GSE163812_matrix': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE163nnn/GSE163812/matrix/GSE163812_series_matrix.txt.gz',
    'GSE180838_xlsx': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE180nnn/GSE180838/suppl/GSE180838_FKBP10.fkpm.xlsx',
    'GSE180838_matrix': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE180nnn/GSE180838/matrix/GSE180838_series_matrix.txt.gz',
    'GSE186141_xlsx': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/GSE186141/suppl/GSE186141_FPKM9.6Col1.vs.2Ctrl.xlsx',
    'GSE186141_matrix': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE186nnn/GSE186141/matrix/GSE186141_series_matrix.txt.gz',
}

FILES = {
    'GSE160207_counts': 'GSE160207_EE_OI_RNAseq_counts.txt.gz',
    'GSE160207_matrix': 'GSE160207_series_matrix.txt.gz',
    'GSE163812_counts': 'GSE163812_ESAT_counts.txt.gz',
    'GSE163812_matrix': 'GSE163812_series_matrix.txt.gz',
    'GSE180838_xlsx': 'GSE180838_FKBP10.fkpm.xlsx',
    'GSE180838_matrix': 'GSE180838_series_matrix.txt.gz',
    'GSE186141_xlsx': 'GSE186141_FPKM9.6Col1.vs.2Ctrl.xlsx',
    'GSE186141_matrix': 'GSE186141_series_matrix.txt.gz',
}


def ensure_file(url, path):
    if os.path.exists(path):
        return
    r = requests.get(url, timeout=90)
    r.raise_for_status()
    with open(path, 'wb') as f:
        f.write(r.content)


def parse_series_matrix(path):
    with gzip.open(path, 'rt') as f:
        lines = f.read().splitlines()

    fields = {}
    for line in lines:
        if line.startswith('!Sample_'):
            parts = line.split('\t')
            key = parts[0]
            vals = [v.strip().strip('"') for v in parts[1:]]
            fields.setdefault(key, []).append(vals)

    # Flatten single-entry fields
    flat = {}
    for k, v in fields.items():
        if len(v) == 1:
            flat[k] = v[0]
        else:
            flat[k] = v

    n = len(flat.get('!Sample_title', []))
    sample_chars = [[] for _ in range(n)]
    for char_vals in fields.get('!Sample_characteristics_ch1', []):
        for i, val in enumerate(char_vals):
            if i < n:
                sample_chars[i].append(val)

    meta_rows = []
    titles = flat.get('!Sample_title', [])
    sources = flat.get('!Sample_source_name_ch1', [''] * n)
    accessions = flat.get('!Sample_geo_accession', [''] * n)
    for i in range(n):
        meta_rows.append({
            'title': titles[i] if i < len(titles) else '',
            'source': sources[i] if i < len(sources) else '',
            'gsm': accessions[i] if i < len(accessions) else '',
            'chars': sample_chars[i],
        })
    return meta_rows


def clean_symbol(x):
    s = str(x).strip().upper()
    if s in {'', 'NA', 'NAN', '--', '-', 'NONE'}:
        return None
    s = s.split('///')[0].strip()
    s = s.split(';')[0].strip()
    return s if s else None


def postprocess_expr(df, symbol_col):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df[symbol_col] = df[symbol_col].apply(clean_symbol)
    df = df[df[symbol_col].notna()]

    keep_cols = [c for c in df.columns if c != symbol_col]
    for c in keep_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.groupby(symbol_col, as_index=True).mean(numeric_only=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how='all')
    df = df.fillna(df.median(axis=1), axis=0)

    # Unified scale across cohorts
    df = np.log2(df + 1.0)
    return df


def load_gse160207():
    counts_path = os.path.join(RAW_DIR, FILES['GSE160207_counts'])
    matrix_path = os.path.join(RAW_DIR, FILES['GSE160207_matrix'])

    counts = pd.read_csv(counts_path, sep='\t', compression='gzip')
    meta_rows = parse_series_matrix(matrix_path)

    title_to_family = {}
    for r in meta_rows:
        fam = ''
        for c in r['chars']:
            m = re.search(r'family:\s*([^;]+)', c, flags=re.IGNORECASE)
            if m:
                fam = m.group(1).strip()
        title_to_family[r['title']] = fam if fam else r['title']

    sample_cols = [c for c in counts.columns if c.startswith('OI-') or c.startswith('C-')]
    expr = counts[['symbol'] + sample_cols]
    expr = postprocess_expr(expr, 'symbol')

    new_cols = {}
    meta = []
    for s in sample_cols:
        sid = f'GSE160207|{s}'
        new_cols[s] = sid
        cond = 'OI' if s.startswith('OI-') else 'Control'
        meta.append({
            'SampleID': sid,
            'Dataset': 'GSE160207',
            'Batch': 'GSE160207',
            'Condition': cond,
            'Label': 1 if cond == 'OI' else 0,
            'GroupID': f"GSE160207|{title_to_family.get(s, s)}",
        })

    expr = expr.rename(columns=new_cols)
    return expr, pd.DataFrame(meta)


def load_gse163812():
    counts_path = os.path.join(RAW_DIR, FILES['GSE163812_counts'])
    matrix_path = os.path.join(RAW_DIR, FILES['GSE163812_matrix'])

    counts = pd.read_csv(counts_path, sep='\t', compression='gzip')
    meta_rows = parse_series_matrix(matrix_path)

    src_map = {}
    for r in meta_rows:
        chars = ' ; '.join(r['chars']).lower()
        if 'patient donor' in chars:
            cond = 'OI'
        elif 'healthy donor' in chars:
            cond = 'Control'
        else:
            cond = 'Unknown'

        if 'treatment: gfp' in chars:
            treatment = 'GFP'
        elif 'treatment: xbp1s' in chars:
            treatment = 'XBP1s'
        else:
            treatment = 'Unknown'

        src_map[r['source']] = {
            'Condition': cond,
            'Treatment': treatment,
            'Title': r['title'],
        }

    sample_cols = [c for c in counts.columns if c in src_map]
    # Keep only baseline GFP to reduce intervention confounding
    sample_cols = [c for c in sample_cols if src_map[c]['Treatment'] == 'GFP' and src_map[c]['Condition'] in {'OI', 'Control'}]

    expr = counts[['Symbol'] + sample_cols]
    expr = postprocess_expr(expr, 'Symbol')

    new_cols = {}
    meta = []
    for s in sample_cols:
        sid = f'GSE163812|{s}'
        new_cols[s] = sid
        cond = src_map[s]['Condition']
        meta.append({
            'SampleID': sid,
            'Dataset': 'GSE163812',
            'Batch': 'GSE163812',
            'Condition': cond,
            'Label': 1 if cond == 'OI' else 0,
            'GroupID': f"GSE163812|{s}",
        })

    expr = expr.rename(columns=new_cols)
    return expr, pd.DataFrame(meta)


def load_gse180838():
    xlsx_path = os.path.join(RAW_DIR, FILES['GSE180838_xlsx'])

    fpkm = pd.read_excel(xlsx_path, sheet_name='FKPM')
    sinfo = pd.read_excel(xlsx_path, sheet_name='sample information')

    sample_cols = [c for c in fpkm.columns if str(c).startswith('B')]
    expr = fpkm[['gene_short_name'] + sample_cols]
    expr = postprocess_expr(expr, 'gene_short_name')

    cond_map = {}
    for _, r in sinfo.iterrows():
        pid = str(r['PatientID']).strip()
        cond = str(r['Condition']).strip().lower()
        label = 'OI' if cond == 'oi' else 'Control'
        cond_map[pid] = label

    new_cols = {}
    meta = []
    for s in sample_cols:
        sid = f'GSE180838|{s}'
        new_cols[s] = sid
        cond = cond_map.get(s, 'Unknown')
        if cond == 'Unknown':
            continue
        meta.append({
            'SampleID': sid,
            'Dataset': 'GSE180838',
            'Batch': 'GSE180838',
            'Condition': cond,
            'Label': 1 if cond == 'OI' else 0,
            'GroupID': f'GSE180838|{s}',
        })

    expr = expr.rename(columns=new_cols)
    expr = expr[[m['SampleID'] for m in meta]]
    return expr, pd.DataFrame(meta)


def load_gse186141():
    xlsx_path = os.path.join(RAW_DIR, FILES['GSE186141_xlsx'])
    matrix_path = os.path.join(RAW_DIR, FILES['GSE186141_matrix'])

    df = pd.read_excel(xlsx_path, sheet_name='Sheet1')
    sample_cols = [c for c in df.columns if re.match(r'^[BS]\d+', str(c))]
    expr = df[['gene_short_name'] + sample_cols]
    expr = postprocess_expr(expr, 'gene_short_name')

    meta_rows = parse_series_matrix(matrix_path)
    title_to_cond = {}
    for r in meta_rows:
        cond = 'Unknown'
        for c in r['chars']:
            lc = c.lower()
            if 'disease state: oi' in lc:
                cond = 'OI'
            elif 'disease state: healthy' in lc or 'disease state: non-oi' in lc:
                cond = 'Control'
        title_to_cond[r['title']] = cond

    new_cols = {}
    meta = []
    for s in sample_cols:
        s0 = str(s).split('_')[0]
        sid = f'GSE186141|{s0}'
        new_cols[s] = sid
        cond = title_to_cond.get(s0, 'Unknown')
        if cond == 'Unknown':
            cond = 'Control' if s0 in {'B14', 'B15'} else 'OI'
        meta.append({
            'SampleID': sid,
            'Dataset': 'GSE186141',
            'Batch': 'GSE186141',
            'Condition': cond,
            'Label': 1 if cond == 'OI' else 0,
            'GroupID': f'GSE186141|{s0}',
        })

    expr = expr.rename(columns=new_cols)
    expr = expr[[m['SampleID'] for m in meta]]
    return expr, pd.DataFrame(meta)


def main():
    # Ensure all files are present
    for key, url in URLS.items():
        ensure_file(url, os.path.join(RAW_DIR, FILES[key]))

    loaders = [load_gse160207, load_gse163812, load_gse180838, load_gse186141]

    all_expr = {}
    all_meta = []
    for fn in loaders:
        expr, meta = fn()
        ds = meta['Dataset'].iloc[0]
        all_expr[ds] = expr
        all_meta.append(meta)

        expr.to_csv(os.path.join(DATASET_DIR, f'{ds}_expr.csv'))
        meta.to_csv(os.path.join(DATASET_DIR, f'{ds}_meta.csv'), index=False)
        print(f'{ds}: {expr.shape[0]} genes x {expr.shape[1]} samples, labels={meta.Condition.value_counts().to_dict()}')

    meta_all = pd.concat(all_meta, ignore_index=True)
    meta_all.to_csv(os.path.join(PROC_DIR, 'multicohort_metadata.csv'), index=False)

    # Global intersection for reference
    common_genes = set(next(iter(all_expr.values())).index)
    for df in all_expr.values():
        common_genes &= set(df.index)
    common_genes = sorted(common_genes)

    combined = pd.concat([all_expr[d].loc[common_genes] for d in sorted(all_expr.keys())], axis=1)
    combined.to_csv(os.path.join(PROC_DIR, 'multicohort_expression_common.csv'))

    print('\nSaved multicohort files.')
    print(f'Common genes across all cohorts: {len(common_genes)}')
    print(f'Total samples: {len(meta_all)}')


if __name__ == '__main__':
    main()
