# Methods Summary

Study design
- Public GEO datasets were curated into a multicohort human dataset with strict labeling and removal of intervention samples.
- A separate expanded dataset that includes a mouse cohort was constructed for sensitivity analyses.

Preprocessing
- Gene symbols were cleaned and duplicate symbols collapsed by mean.
- Expression values were log2 transformed and missing values imputed per gene.
- Batch effects were corrected with ComBat using Condition as a covariate.

Graph construction
- The STRING PPI network was filtered at confidence 700.
- Top K genes by median absolute deviation were selected per fold.
- Node features used expression, MAD rank, degree, and clustering coefficient.

Models
- OIGATv2 uses three TransformerConv layers with residuals, global mean and max pooling, and an MLP head.
- Baseline models include Logistic Regression, Random Forest, and SVM.

Evaluation
- Leave one dataset out external validation assesses real world generalization.
- Human only StratifiedGroupKFold enforces grouped 5 fold evaluation by family or subject.
- Metrics include accuracy, AUC, and F1 score.
