# Slide 04 - Methods (Architecture and Validation)
## Section
Methods

## Text to display
- Final architecture: Multiplex-Hypergraph-DANN-MLP.
- Relation channels: KEGG pathway hyperedges, STRING interaction graph, and dynamic co-expression edges.
- Core blocks: relation-specific hypergraph convolutions -> relation attention -> gene scoring mask -> MLP classifier.
- Domain robustness: adversarial domain head discourages batch-specific shortcuts during training.
- Validation protocols:
- 5-fold stratified CV for core model development.
- Strict leave-one-cohort/batch-out baseline protocols.
- Independent external validation on GSE26440.

## Image to display
- `ACSEF_Final_Submission/figures/fig_architecture_flowchart.png`

## Graphic credit (APA)
- Student Researcher. (2026). Multiplex-Hypergraph-DANN-MLP architecture flowchart [Figure]. Generated from final ACSEF model documentation.
