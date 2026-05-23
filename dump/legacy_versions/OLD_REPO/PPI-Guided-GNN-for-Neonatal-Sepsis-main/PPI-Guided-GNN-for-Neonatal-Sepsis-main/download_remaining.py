"""
Download GSE69686 and GSE26440 in SOFT format using GEOparse.
Series matrix files appear to have no expression data embedded.
"""

import GEOparse
import os

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

datasets = ["GSE69686", "GSE26440"]

for geo_id in datasets:
    print(f"\nDownloading {geo_id} via GEOparse (full SOFT format)...")
    try:
        gse = GEOparse.get_GEO(geo=geo_id, destdir=DATA_DIR)
        print(f"{geo_id} downloaded successfully!")
        print(f"Number of samples: {len(gse.gsms)}")
        print(f"Platforms: {list(gse.gpls.keys())}")
    except Exception as e:
        print(f"Error downloading {geo_id}: {e}")
