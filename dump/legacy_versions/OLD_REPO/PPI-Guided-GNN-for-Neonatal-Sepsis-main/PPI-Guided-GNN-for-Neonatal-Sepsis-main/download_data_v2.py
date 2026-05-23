import os
import requests
import gseapy as gp
import shutil

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Download GEO datasets using gseapy
geo_ids = ["GSE25504", "GSE69686", "GSE26440"]
for acc in geo_ids:
    print(f"Downloading {acc} via gseapy...")
    try:
        # destdir is where gseapy saves the files
        # It creates a folder or saves directly.
        gp.get_geo(geo=acc, destdir=DATA_DIR)
        print(f"Successfully downloaded {acc}")
    except Exception as e:
        print(f"Failed to download {acc}: {e}")

# 2. Download Interaction Networks
URLS = {
    "STRING": "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
    "BioGRID": "https://downloads.thebiogrid.org/BioGRID/Release-Archive/BIOGRID-4.4.229/BIOGRID-ORGANISM-Homo_sapiens-4.4.229.tab3.zip"
}

def download_file(url, filename):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000000: # Check if > 1MB
        print(f"File {filename} exists and seems valid. Skipping.")
        return filepath
    
    print(f"Downloading {filename}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        with requests.get(url, stream=True, headers=headers) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {filename}")
        return filepath
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return None

for name, url in URLS.items():
    filename = url.split('/')[-1]
    download_file(url, filename)
