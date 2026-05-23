import os
import requests
import shutil

DATA_DIR = "data/raw"

# NCBI Download API for robust fetching
# Note: file= argument needs URL encoded name
GSE25504_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE25504&format=file&file=GSE25504%5Fseries%5Fmatrix%2Etxt%2Egz"

# BioGRID: Try a different version or main link if specific version fails
# Trying 4.4.229 again but verifying size
BIOGRID_URL = "https://downloads.thebiogrid.org/BioGRID/Release-Archive/BIOGRID-4.4.229/BIOGRID-ORGANISM-Homo_sapiens-4.4.229.tab3.zip"

def download_url(url, output_path):
    print(f"Downloading to {output_path}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        size = os.path.getsize(output_path)
        print(f"Downloaded {output_path} ({size/1024/1024:.2f} MB)")
        return True
    except Exception as e:
        print(f"Error downloading {output_path}: {e}")
        return False

def main():
    # 1. GSE25504
    gse25504_path = os.path.join(DATA_DIR, "GSE25504_series_matrix.txt.gz")
    if not os.path.exists(gse25504_path):
        download_url(GSE25504_URL, gse25504_path)
    else:
        print("GSE25504 already exists.")

    # 2. BioGRID
    biogrid_path = os.path.join(DATA_DIR, "BIOGRID-ORGANISM-Homo_sapiens-4.4.229.tab3.zip")
    if not os.path.exists(biogrid_path):
        success = download_url(BIOGRID_URL, biogrid_path)
        # Check if small (error page)
        if success and os.path.getsize(biogrid_path) < 100000: # <100KB
            print("BioGRID file too small, likely an error page. Deleting.")
            os.remove(biogrid_path)
    else:
        print("BioGRID already exists.")

if __name__ == "__main__":
    main()
