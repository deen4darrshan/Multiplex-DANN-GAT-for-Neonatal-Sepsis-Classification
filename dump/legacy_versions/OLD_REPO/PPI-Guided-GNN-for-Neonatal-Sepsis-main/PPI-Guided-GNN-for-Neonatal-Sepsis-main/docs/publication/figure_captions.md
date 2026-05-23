# Figure Captions

pipeline_overview.png
Overview of the OI analysis pipeline from raw GEO datasets through cleaning, ComBat correction, feature selection, graph construction, and the GAT classifier.

pca_before_combat.png
PCA of log2 transformed expression before ComBat, colored by batch and condition.

pca_after_combat.png
PCA after ComBat batch correction to show reduced dataset driven separation.

dataset_composition.png
Dataset composition for the human multicohort data showing OI and control counts per dataset.

real_external_accuracy_by_holdout.png
External accuracy for leave one dataset out validation, comparing GAT and Logistic Regression.

real_roc_GSE160207.png
External ROC for GSE160207 holdout with GAT and Logistic Regression.

real_roc_GSE163812.png
External ROC for GSE163812 holdout with GAT and Logistic Regression.

real_roc_GSE180838.png
External ROC for GSE180838 holdout with GAT and Logistic Regression.

real_roc_GSE186141.png
External ROC for GSE186141 holdout with GAT and Logistic Regression.

human_grouped5_accuracy.png
Human only grouped 5 fold accuracy comparing best tabular baseline and GAT.

human_grouped5_l2_lr_tuning_best.png
Best L2 Logistic Regression configuration from grouped 5 fold tuning.

human_grouped5_optimized_lr_roc.png
ROC curve for the tuned threshold L2 Logistic Regression in human grouped 5 fold evaluation.

metrics_summary_accuracy.png
Key evaluation accuracies across the external holdout and grouped 5 fold protocols.

architecture\oigatv2_architecture.png
OIGATv2 architecture showing three TransformerConv layers, global pooling, and MLP classification head.
