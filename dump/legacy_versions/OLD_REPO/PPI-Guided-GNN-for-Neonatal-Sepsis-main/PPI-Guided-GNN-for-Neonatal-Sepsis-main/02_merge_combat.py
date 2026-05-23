"""
Module B - Task B.2: Merging & ComBat Batch Correction (V3)
"""

import pandas as pd
import numpy as np
from combat.pycombat import pycombat
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Paths
DATA_DIR = "data/processed"
OUT_DIR = "data/processed"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def parse_conditions(pheno_df, dataset_name):
    """Parse conditions from phenotype data.
    
    GSE25504 uses multiple formats:
    - characteristics: 'neonate: control' or 'neonate: infected'
    - title prefix: 'Con' (Control), 'Inf' (Infected/Sepsis)
    - title prefix: 'Sus' (Suspected), 'NEC' (Necrotizing enterocolitis), 'Vir' (Viral)
    """
    labels = []
    
    for idx, row in pheno_df.iterrows():
        char = str(row.get('characteristics', '')).lower()
        title = str(row.get('title', ''))
        source = str(row.get('source', '')).lower()
        combined = char + ' ' + title.lower() + ' ' + source
        
        # PRIORITY 1: Check title prefix (most reliable for GSE25504)
        title_prefix = title[:3] if len(title) >= 3 else ''
        
        if title_prefix.lower() == 'con':
            labels.append('Control')
        elif title_prefix.lower() == 'inf':
            labels.append('Sepsis')
        # PRIORITY 2: Check characteristics for 'neonate: control/infected'
        elif 'neonate: control' in char:
            labels.append('Control')
        elif 'neonate: infected' in char:
            labels.append('Sepsis')
        # PRIORITY 3: Check other keywords
        elif any(x in combined for x in ['sepsis', 'septic', 'infected', 'infection']):
            if 'uninfected' in combined or 'non-infected' in combined:
                labels.append('Control')
            else:
                labels.append('Sepsis')
        elif any(x in combined for x in ['control', 'healthy', 'normal', 'uninfected']):
            labels.append('Control')
        # NEC, Viral, Suspected cases - classify based on context
        elif title_prefix.lower() in ['nec', 'vir']:
            labels.append('Sepsis')  # NEC and Viral are infection-related
        elif title_prefix.lower() == 'sus':
            labels.append('Control')  # Suspected but not confirmed = Control group
        else:
            labels.append('Unknown')
    
    pheno_df['Condition'] = labels
    print(f"{dataset_name} conditions: {pheno_df['Condition'].value_counts().to_dict()}")
    return pheno_df

def run_pca_safe(data, n_components=2):
    """Run PCA safely, handling edge cases."""
    X = data.T.values if hasattr(data, 'T') else data
    X = np.nan_to_num(X, nan=0)
    
    # Remove zero-variance features
    variances = np.var(X, axis=0)
    valid_mask = variances > 1e-10
    X_filtered = X[:, valid_mask]
    
    if X_filtered.shape[1] < n_components:
        print(f"Warning: Only {X_filtered.shape[1]} valid features for PCA")
        return None, None
    
    # Center the data
    X_centered = X_filtered - X_filtered.mean(axis=0)
    
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X_centered)
    
    return pcs, pca

def main():
    print("Loading mapped expression data...")
    gse25504 = pd.read_csv(os.path.join(DATA_DIR, "GSE25504_mapped.csv"), index_col=0)
    gse69686 = pd.read_csv(os.path.join(DATA_DIR, "GSE69686_mapped.csv"), index_col=0)
    print(f"GSE25504: {gse25504.shape}")
    print(f"GSE69686: {gse69686.shape}")
    
    print("\nLoading phenotype data...")
    pheno25504 = pd.read_csv(os.path.join(DATA_DIR, "GSE25504_phenotype.csv"), index_col=0)
    pheno69686 = pd.read_csv(os.path.join(DATA_DIR, "GSE69686_phenotype.csv"), index_col=0)
    # Load NEW Expanded Data
    gse26440_neo = pd.read_csv(os.path.join(DATA_DIR, "GSE26440_Neo_mapped.csv"), index_col=0)
    pheno26440_neo = pd.read_csv(os.path.join(DATA_DIR, "GSE26440_Neo_phenotype.csv"), index_col=0)
    print(f"GSE26440_Neo: {gse26440_neo.shape}")

    # Parse conditions
    pheno25504 = parse_conditions(pheno25504, 'GSE25504')
    pheno69686 = parse_conditions(pheno69686, 'GSE69686')
    # Use same logic for GSE26440 (mostly Sepsis/Septic Shock vs Control)
    pheno26440_neo = parse_conditions(pheno26440_neo, 'GSE26440_Neo')
    
    # Intersect genes across ALL 3 datasets
    print("\nIntersecting genes...")
    common_genes = list(set(gse25504.index) & set(gse69686.index) & set(gse26440_neo.index))
    print(f"Common genes: {len(common_genes)}")
    
    gse25504_filtered = gse25504.loc[common_genes].copy()
    gse69686_filtered = gse69686.loc[common_genes].copy()
    gse26440_filtered = gse26440_neo.loc[common_genes].copy()
    
    # Convert to numeric
    gse25504_filtered = gse25504_filtered.apply(pd.to_numeric, errors='coerce')
    gse69686_filtered = gse69686_filtered.apply(pd.to_numeric, errors='coerce')
    gse26440_filtered = gse26440_filtered.apply(pd.to_numeric, errors='coerce')
    
    # Merge ALL 3
    combined = pd.concat([gse25504_filtered, gse69686_filtered, gse26440_filtered], axis=1)
    print(f"Combined shape before cleaning: {combined.shape}")
    
    # Create batch labels with PLATFORM-AWARE splitting
    # GSM627xxx = Affymetrix (GSE25504)
    # GSM1404xxx = Illumina (GSE25504)
    # GSE69686 samples
    # GSE26440 samples (New Batch)
    def assign_platform_batch(sample_id):
        if sample_id.startswith('GSM627'):
            return 'GSE25504_Affy'
        elif sample_id.startswith('GSM1404'):
            return 'GSE25504_Illu'
        elif sample_id in gse26440_filtered.columns:
            return 'GSE26440_Neo'
        else:
            return 'GSE69686'
    
    batch = [assign_platform_batch(s) for s in combined.columns]
    print(f"Platform-aware batch distribution: {pd.Series(batch).value_counts().to_dict()}")
    
    conditions = []
    for sample in combined.columns:
        if sample in pheno25504.index:
            conditions.append(pheno25504.loc[sample, 'Condition'])
        elif sample in pheno69686.index:
            conditions.append(pheno69686.loc[sample, 'Condition'])
        elif sample in pheno26440_neo.index:
             conditions.append(pheno26440_neo.loc[sample, 'Condition'])
        else:
            conditions.append('Unknown')
    
    print(f"Combined conditions: {pd.Series(conditions).value_counts().to_dict()}")
    
    # Handle NaN values - use row-wise imputation
    print(f"NaN count before imputation: {combined.isna().sum().sum()}")
    
    # Fill NaN with row mean (gene-wise mean across samples)
    combined = combined.T.fillna(combined.T.mean()).T
    
    # Remove rows that are still all NaN
    combined = combined.dropna(how='all')
    
    # Remove constant rows
    row_var = combined.var(axis=1)
    combined = combined[row_var > 1e-10]
    print(f"After cleaning: {combined.shape}")
    
    # PCA before ComBat
    print("\nPCA before ComBat...")
    pcs_before, pca_before = run_pca_safe(combined)
    
    if pcs_before is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        batch_colors = {'GSE25504': 'blue', 'GSE69686': 'red'}
        for b in set(batch):
            mask = [x == b for x in batch]
            axes[0].scatter(pcs_before[mask, 0], pcs_before[mask, 1], label=b, alpha=0.6, 
                          c=batch_colors.get(b, 'gray'))
        axes[0].set_xlabel(f'PC1 ({pca_before.explained_variance_ratio_[0]*100:.1f}%)')
        axes[0].set_ylabel(f'PC2 ({pca_before.explained_variance_ratio_[1]*100:.1f}%)')
        axes[0].set_title('Before ComBat - Colored by Batch')
        axes[0].legend()
        
        cond_colors = {'Sepsis': 'red', 'Control': 'green', 'Unknown': 'gray'}
        for c in set(conditions):
            mask = [x == c for x in conditions]
            axes[1].scatter(pcs_before[mask, 0], pcs_before[mask, 1], label=c, alpha=0.6,
                          c=cond_colors.get(c, 'gray'))
        axes[1].set_xlabel(f'PC1 ({pca_before.explained_variance_ratio_[0]*100:.1f}%)')
        axes[1].set_ylabel(f'PC2 ({pca_before.explained_variance_ratio_[1]*100:.1f}%)')
        axes[1].set_title('Before ComBat - Colored by Condition')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'pca_before_combat.png'), dpi=150)
        plt.close()
        print("Saved pca_before_combat.png")
    
    # Apply ComBat
    print("\nApplying ComBat...")
    try:
        corrected = pycombat(combined, batch=batch)
        print("ComBat successful!")
    except Exception as e:
        print(f"ComBat failed: {e}")
        corrected = combined
    
    # PCA after ComBat
    print("\nPCA after ComBat...")
    pcs_after, pca_after = run_pca_safe(corrected)
    
    if pcs_after is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for b in set(batch):
            mask = [x == b for x in batch]
            axes[0].scatter(pcs_after[mask, 0], pcs_after[mask, 1], label=b, alpha=0.6,
                          c=batch_colors.get(b, 'gray'))
        axes[0].set_xlabel(f'PC1')
        axes[0].set_ylabel(f'PC2')
        axes[0].set_title('After ComBat - Colored by Batch')
        axes[0].legend()
        
        for c in set(conditions):
            mask = [x == c for x in conditions]
            axes[1].scatter(pcs_after[mask, 0], pcs_after[mask, 1], label=c, alpha=0.6,
                          c=cond_colors.get(c, 'gray'))
        axes[1].set_xlabel(f'PC1')
        axes[1].set_ylabel(f'PC2')
        axes[1].set_title('After ComBat - Colored by Condition')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'pca_after_combat.png'), dpi=150)
        plt.close()
        print("Saved pca_after_combat.png")
    
    # Save combined training data
    print("\nSaving combined training data...")
    
    # Save expression matrix (genes x samples) - for graph construction
    corrected.to_csv(os.path.join(OUT_DIR, "combined_expression.csv"))
    
    # Save sample metadata
    metadata = pd.DataFrame({
        'SampleID': corrected.columns,
        'Condition': conditions,
        'Batch': batch
    })
    metadata.to_csv(os.path.join(OUT_DIR, "combined_metadata.csv"), index=False)
    
    # Also save transposed (samples x genes) with metadata for ML
    combined_training = corrected.T.copy()
    combined_training.insert(0, 'Condition', conditions)
    combined_training.insert(1, 'Batch', batch)
    combined_training.to_csv(os.path.join(OUT_DIR, "combined_training.csv"))
    
    # Summary
    print(f"\n{'='*60}")
    print("=== FINAL SUMMARY ===")
    print(f"{'='*60}")
    print(f"Combined training shape: {combined_training.shape}")
    print(f"Total samples: {len(combined_training)}")
    print(f"Total genes: {corrected.shape[0]}")
    print(f"Condition distribution: {combined_training['Condition'].value_counts().to_dict()}")
    
    # CoVe Verification
    print(f"\n=== VERIFICATION (CoVe) ===")
    expected = 319
    actual = len(combined_training)
    print(f"Expected samples: ~{expected}")
    print(f"Actual samples: {actual}")
    
    if actual >= 300:
        print("✓ PASS: Sample count acceptable")
    else:
        print("✗ FAIL: Sample count too low")

if __name__ == "__main__":
    main()
