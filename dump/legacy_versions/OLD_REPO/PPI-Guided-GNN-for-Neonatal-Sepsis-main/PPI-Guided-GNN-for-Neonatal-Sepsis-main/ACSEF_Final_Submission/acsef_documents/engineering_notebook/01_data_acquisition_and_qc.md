# 01 Data Acquisition and Quality Control

Date: 2026-02-24

## Data Sources
- Development cohorts: GEO series GSE25504 and GSE69686.
- External validation cohort: GEO series GSE26440.
- Biological priors:
  - KEGG pathway memberships.
  - STRING protein-protein interactions (v12 file scan).
  - Data-driven co-expression graph from expression correlations.

## Acquisition Procedure
1. Downloaded GEO series using GEOparse-compatible scripts.
2. Retrieved STRING interaction table (`9606.protein.links.v12.0.txt.gz`).
3. Recorded failed direct endpoint responses and switched to SOFT-based fallback where required.

## Quality Checks
- Confirmed per-sample label availability.
- Harmonized to common gene universe.
- Inspected missing values and removed degenerate features.
- Verified class counts before training splits.

## Engineering Constraints
Because STRING contains more than 11M interactions, scanning was chunked and filtered to top-gene proteins only. This prevented swap-thrashing and stabilized runtime on Windows.
