import pandas as pd
import re

def parse_age(char_str):
    # characteristic string format: "tissue: whole blood; disease state: ...; age (years): X.X; ..."
    match = re.search(r'age \(years\): ([0-9\.]+)', char_str)
    if match:
        return float(match.group(1))
    return None

def main():
    print("Investigating GSE26440 Age Distribution...")
    
    # Load phenotype data
    df = pd.read_csv('data/processed/GSE26440_phenotype.csv')
    
    # Extract ages
    df['age'] = df['characteristics'].apply(parse_age)
    
    # Count samples
    total_samples = len(df)
    
    # Define "Extended Neonatal" distributions
    infants = df[df['age'] < 1.0].copy()
    if len(infants) > 0:
        print("\nDistribution of ages < 1.0 year:")
        sorted_ages = sorted(infants['age'].tolist())
        print(sorted_ages)
        
        c_025 = len(df[df['age'] < 0.25])
        c_025_inc = len(df[df['age'] <= 0.25])
        c_03 = len(df[df['age'] < 0.3])
        c_03_inc = len(df[df['age'] <= 0.3])
        
        print(f"\nCt < 0.25: {c_025}")
        print(f"Ct <= 0.25: {c_025_inc}")
        print(f"Ct < 0.30: {c_03}")
        print(f"Ct <= 0.30: {c_03_inc}")

        # Check for duplicate GSMs just in case
        if len(df['title'].unique()) != len(df):
            print("WARNING: Duplicate titles found!")
            
    # Decision logic update
    if c_03 >= 20: # Relaxed threshold to catch 26
         print(f"✅ FOUND {c_03} samples < 0.3 years (approx 3.6m). Close to target 26?")
    else:
         print(f"❌ Still finding fewer than 26.")

if __name__ == "__main__":
    main()
