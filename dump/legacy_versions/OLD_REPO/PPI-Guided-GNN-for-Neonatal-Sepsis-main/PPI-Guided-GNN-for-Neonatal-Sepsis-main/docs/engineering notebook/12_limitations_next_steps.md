# Limitations and Next Steps

Limitations
- Sample sizes are small and class imbalance remains a risk for variance in metrics.
- Cohort heterogeneity is large, even after batch correction.
- Some cohorts are processed from FPKM while others are counts, which can introduce scale differences.
- External validation results can be sensitive to the choice of top K genes.

Next steps
- Evaluate additional independent human cohorts when available.
- Explore multi task learning with phenotype severity if metadata becomes available.
- Investigate biological interpretability of attention scores and gene subnetworks.
- Increase robustness with nested cross validation for hyperparameter selection.
