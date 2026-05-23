# 03 Architecture and Mathematical Formulation

Date: 2026-02-24

## Final Named Model
`Multiplex-Hypergraph-DANN-MLP`

## Components
1. Multiplex hypergraph convolution over three relations:
   - KEGG relation hyperedges.
   - STRING PPI relation hyperedges.
   - Co-expression relation hyperedges.
2. Relation attention fusion.
3. Gene scoring head (importance mask).
4. MLP classifier for disease prediction.
5. Domain-adversarial head for batch/domain invariance.

## Relation-Specific Encoding
For relation r in {kegg, string, coexpr}:
- H_r = HypergraphConv_r(X, E_r)
where X is sample-by-gene input and E_r is relation-specific hyperedge structure.

## Attention Fusion
- a_r = softmax(W_a [H_kegg, H_string, H_coexpr])
- H_fused = sum_r a_r * H_r

## Gene Scoring
- s = sigmoid(W_s H_fused + b_s)
- X_masked = X * s

## Classifier and Domain Head
- y_hat = MLP_class(X_masked)
- d_hat = MLP_domain(grad_reverse(H_fused))

## Training Objective
- L_total = L_cls(y_hat, y) + alpha * L_dom(d_hat, d) + beta * L_reg
where:
- L_cls: binary cross entropy for sepsis label.
- L_dom: domain classification loss through gradient reversal.
- L_reg: weight decay and optional sparsity terms.

## Why This Matters
Archived ablation showed that removing the MLP caused AUROC collapse toward random chance, indicating non-linear feature synthesis is essential after graph-based feature weighting.
