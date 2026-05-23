# 00 Project Origin and Scope

Date: 2026-02-24

## Problem Framing
The project began from a translational constraint: neonatal sepsis classifiers trained on one cohort often degrade on another cohort due to domain shift (batch, platform, and cohort composition changes). The design objective was to maximize external validity while preserving interpretability at the gene level.

## Development Lineage
1. Baseline graph models (GCN/GAT/HGCN) were used as reference performance anchors.
2. Multiplex graph/hypergraph variants were introduced to encode complementary biological relations.
3. Domain-adversarial learning was added to reduce spurious cohort-specific cues.
4. Explainability was integrated using gene scoring and Integrated Gradients.
5. Cross-disease transfer was explored in osteogenesis imperfecta to test generality.

## Final Scientific Claim
A multiplex, biology-informed, domain-aware architecture yields stronger cross-cohort sepsis performance and preserves actionable biomarker interpretation.

## Repository Exploration Summary
Reviewed directories included:
- `CH_DANN_Plan` for final sepsis architecture and external validation runs.
- `Sepsis_GNN_V2` for historical baseline results.
- `Osteogenesis imperfecta` for rare-disease scaling and grouped validation outputs.
- Root-level legacy scripts and data acquisition utilities for reproducibility context.
