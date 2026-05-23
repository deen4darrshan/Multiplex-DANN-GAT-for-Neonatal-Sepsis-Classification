import os
import requests
import gzip
import shutil

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

URLS = {
    "GSE25504": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE25nnn/GSE25504/matrix/GSE25504_series_matrix.txt.gz",
    "GSE69686": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE69nnn/GSE69686/matrix/GSE69686_series_matrix.txt.gz",
    "GSE26440": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE26nnn/GSE26440/matrix/GSE26440_series_matrix.txt.gz",
    "STRING": "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
    # Using a specific BioGRID version to ensure stability
    "BioGRID": "https://downloads.thebiogrid.org/BioGRID/Release-Archive/BIOGRID-4.4.229/BIOGRID-ORGANISM-Homo_sapiens-4.4.229.tab3.zip"
}

def download_file(url, filename):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"File {filename} already exists. Skipping.")
        return filepath
    
    print(f"Downloading {filename}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {filename}")
        return filepath
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return None

def main():
    for name, url in URLS.items():
        filename = url.split('/')[-1]
        filepath = download_file(url, filename)
        
        # Verify file size
        if filepath and os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"{name}: {filename} ({size_mb:.2f} MB)")
        else:
            print(f"Error: {name} file missing!")

if __name__ == "__main__":
    main()
