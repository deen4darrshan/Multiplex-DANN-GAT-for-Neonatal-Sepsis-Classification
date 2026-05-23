# 05 Explainability Pipeline and Biomarker Extraction

Date: 2026-02-24

## Objective
Quantify gene-level attribution to class predictions while retaining compatibility with Windows execution constraints.

## Integrated Gradients (IG)
For input x, baseline x', model F, and gene index i:
IG_i(x) = (x_i - x'_i) * integral_{alpha=0..1} [dF(x' + alpha(x - x')) / dx_i] d(alpha)

Practical approximation used m steps:
IG_i(x) ~= (x_i - x'_i) * (1/m) * sum_{k=1..m} grad_i(F(x' + k/m * (x - x')))

## Implementation Notes
- Custom PyTorch IG implementation (no Captum dependency required).
- Combined IG signal with learned gene score mask statistics.
- Replaced fragile SciPy calls with NumPy/Pandas rank and MAD alternatives.
- STRING relation construction executed in chunked scan mode.

## Outputs
- `results/top_100_biomarkers.csv`
- `results/all_gene_attributions.csv`
- `figures/fig_biomarker_attributions.png`
- `figures/fig_biomarker_correlation_heatmap.png`
- `figures/fig_top_biomarker_pca.png`

## Top Biomarker Signals (Examples)
TNFAIP6, S100A12, RETN, HP, and ANXA3 showed strong positive sepsis direction in this run, while CD52 and OCIAD2 were strongly control-up in attribution space.
