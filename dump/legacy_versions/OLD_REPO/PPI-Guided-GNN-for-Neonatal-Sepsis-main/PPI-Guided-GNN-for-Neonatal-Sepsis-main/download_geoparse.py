import GEOparse
import os

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

# Download GSE25504 using GEOparse
print("Downloading GSE25504 via GEOparse...")
try:
    gse = GEOparse.get_GEO(geo="GSE25504", destdir=DATA_DIR)
    print(f"GSE25504 downloaded successfully!")
    print(f"Number of samples: {len(gse.gsms)}")
    print(f"Platform: {list(gse.gpls.keys())}")
except Exception as e:
    print(f"Error downloading GSE25504: {e}")
