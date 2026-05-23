import os
import shutil
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

URLS = {
    # GSE160207 (whole blood RNA-seq counts)
    'GSE160207_counts': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE160nnn/GSE160207/suppl/GSE160207_EE_OI_RNAseq_counts.txt.gz',
    'GSE160207_matrix': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE160nnn/GSE160207/matrix/GSE160207_series_matrix.txt.gz',

    # GSE163812 (fibroblast ESAT counts)
    'GSE163812_counts': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE163nnn/GSE163812/suppl/GSE163812_ESAT_counts.txt.gz',
    'GSE163812_matrix': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE163nnn/GSE163812/matrix/GSE163812_series_matrix.txt.gz',

    # STRING PPI (human)
    'STRING': 'https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz'
}

ALT_STRING = os.path.abspath(os.path.join(ROOT, '..', 'ALZHEIMERS_STRATEGIC_PATHWAY', 'data', 'adni', 'raw', '9606.protein.links.v12.0.txt.gz'))


def download(url, out_path):
    if os.path.exists(out_path):
        print(f"[skip] {os.path.basename(out_path)} exists")
        return
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"[saved] {out_path}")


def main():
    for name, url in URLS.items():
        filename = url.split('/')[-1]
        out_path = os.path.join(RAW_DIR, filename)
        if name == 'STRING':
            if os.path.exists(out_path):
                print(f"[skip] STRING already present: {out_path}")
                continue
            if os.path.exists(ALT_STRING):
                print(f"[copy] STRING from {ALT_STRING}")
                shutil.copy2(ALT_STRING, out_path)
                continue
        download(url, out_path)

    print("\nDone.")


if __name__ == '__main__':
    main()
