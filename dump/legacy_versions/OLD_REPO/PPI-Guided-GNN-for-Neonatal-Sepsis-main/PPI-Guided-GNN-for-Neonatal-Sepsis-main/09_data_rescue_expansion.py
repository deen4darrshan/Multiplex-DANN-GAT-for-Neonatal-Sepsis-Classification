"""
Data Rescue & Expansion Script
------------------------------
Hijacks 26 neonatal samples (Age <= 0.3 years) from GSE26440 to augment the training set.
Separates GSE26440 into:
1. GSE26440_Neo (Training, N=26)
2. GSE26440_Valid (Validation, N=104)
"""
import pandas as pd
import os
import re

# Constants
NEO_THRESHOLD = 0.3  # <= 0.3 years (approx 3.6 months)
DATA_DIR = "data/processed"

def parse_age(char_str):
    # characteristic string format: "tissue: whole blood; disease state: ...; age (years): X.X; ..."
    match = re.search(r'age \(years\): ([0-9\.]+)', char_str)
    if match:
        return float(match.group(1))
    return None

def main():
    print("🚀 Initiating Data Rescue Protocol...")
    
    # 1. Load GSE26440 Data
    print("Loading GSE26440 data...")
    expr_df = pd.read_csv(os.path.join(DATA_DIR, "GSE26440_mapped.csv"), index_col=0)
    pheno_df = pd.read_csv(os.path.join(DATA_DIR, "GSE26440_phenotype.csv"))
    
    # 2. Identify Neonatal Samples
    pheno_df['age'] = pheno_df['characteristics'].apply(parse_age)
    
    # Filter: <= 0.3 years
    neo_mask = pheno_df['age'] <= NEO_THRESHOLD
    neo_samples = pheno_df[neo_mask]['title'].tolist() # Use title as ID (starts with GSM in logic)
    # Actually, pheno_df first column is GSM ID (unnamed)
    # Let's fix that - the first column is the ID
    pheno_df = pd.read_csv(os.path.join(DATA_DIR, "GSE26440_phenotype.csv"), index_col=0)
    pheno_df['age'] = pheno_df['characteristics'].apply(parse_age)
    neo_mask = pheno_df['age'] <= NEO_THRESHOLD
    
    neo_ids = pheno_df[neo_mask].index.tolist()
    valid_ids = pheno_df[~neo_mask].index.tolist()
    
    print(f"Found {len(neo_ids)} Neonatal Samples (<= {NEO_THRESHOLD} years)")
    print(f"Remaining {len(valid_ids)} Validation Samples")
    
    if len(neo_ids) != 26:
        print(f"⚠ WARNING: Expected 26 samples, found {len(neo_ids)}. Check threshold logic.")
        
    # 3. Split Expression Data
    # Check intersection with expression columns
    available_neo = [col for col in neo_ids if col in expr_df.columns]
    available_valid = [col for col in valid_ids if col in expr_df.columns]
    
    print(f"Available in Expression Matrix: {len(available_neo)} Neo, {len(available_valid)} Valid")
    
    expr_neo = expr_df[available_neo]
    expr_valid = expr_df[available_valid]
    
    # 4. Save Split Files
    print("Saving split datasets...")
    expr_neo.to_csv(os.path.join(DATA_DIR, "GSE26440_Neo_mapped.csv"))
    expr_valid.to_csv(os.path.join(DATA_DIR, "GSE26440_Valid_mapped.csv"))
    
    # Save phenotypes too for reference
    pheno_df.loc[available_neo].to_csv(os.path.join(DATA_DIR, "GSE26440_Neo_phenotype.csv"))
    pheno_df.loc[available_valid].to_csv(os.path.join(DATA_DIR, "GSE26440_Valid_phenotype.csv"))
    
    print("✅ Data Rescue Complete. Separate files created.")
    print(f"  - Training Boost: {len(available_neo)} samples")
    print(f"  - Validation Set: {len(available_valid)} samples")

if __name__ == "__main__":
    main()
