import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from combat.pycombat import pycombat

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_DIR = os.path.join(ROOT, 'data', 'processed')
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

EXPR_PATH = os.path.join(PROC_DIR, 'combined_expression_log2.csv')
META_PATH = os.path.join(PROC_DIR, 'combined_metadata.csv')


def plot_pca(expression, meta, title, out_path):
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    X = expression.T.values
    X = np.nan_to_num(X, nan=0)

    # Remove zero-variance genes
    var = np.var(X, axis=0)
    X = X[:, var > 1e-10]

    if X.shape[1] < 2:
        print('  [warn] Not enough variance for PCA')
        return

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X - X.mean(axis=0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # By batch
    for b in sorted(meta['Batch'].unique()):
        mask = meta['Batch'] == b
        axes[0].scatter(pcs[mask, 0], pcs[mask, 1], label=b, alpha=0.7, s=40)
    axes[0].set_title(f'{title} - Batch')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].legend(fontsize=8)

    # By condition
    for c in sorted(meta['Condition'].unique()):
        mask = meta['Condition'] == c
        axes[1].scatter(pcs[mask, 0], pcs[mask, 1], label=c, alpha=0.7, s=40)
    axes[1].set_title(f'{title} - Condition')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved PCA: {out_path}')


def main():
    if not os.path.exists(EXPR_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError('Missing processed files. Run 01_prepare_expression.py first.')

    expr = pd.read_csv(EXPR_PATH, index_col=0)
    meta = pd.read_csv(META_PATH)

    # Align columns
    meta = meta[meta['SampleID'].isin(expr.columns)].copy()
    expr = expr[meta['SampleID'].tolist()]

    # Filter unknowns
    meta = meta[meta['Condition'] != 'Unknown'].copy()
    expr = expr[meta['SampleID'].tolist()]

    # Clean
    expr = expr.apply(pd.to_numeric, errors='coerce')
    expr = expr.T.fillna(expr.T.mean()).T
    expr = expr.loc[expr.var(axis=1) > 1e-10]

    print(f'Expression: {expr.shape[0]} genes x {expr.shape[1]} samples')
    print(meta.groupby(['Batch', 'Condition']).size().unstack(fill_value=0))

    # PCA before
    plot_pca(expr, meta, 'Before ComBat', os.path.join(FIG_DIR, 'pca_before_combat.png'))

    # Build covariate matrix (Condition + Treatment)
    covars = meta[['Condition']].copy()
    mod = [covars['Condition'].tolist()]

    # Run ComBat (genes x samples)
    corrected = pycombat(expr, batch=meta['Batch'].tolist(), mod=mod)

    # PCA after
    plot_pca(corrected, meta, 'After ComBat', os.path.join(FIG_DIR, 'pca_after_combat.png'))

    # Save
    corrected.to_csv(os.path.join(PROC_DIR, 'expression_combat.csv'))
    meta.to_csv(os.path.join(PROC_DIR, 'metadata_combat.csv'), index=False)

    print('Saved: expression_combat.csv, metadata_combat.csv')


if __name__ == '__main__':
    main()
