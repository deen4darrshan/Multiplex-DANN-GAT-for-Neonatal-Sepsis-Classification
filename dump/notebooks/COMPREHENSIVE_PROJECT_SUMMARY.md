# PPI-Guided Graph Neural Networks for Neonatal Sepsis: Comprehensive Project Report

**Date:** 2026-02-16
**Status:** Phase 3 (Refinement & Reboot)

---

## 1. Motivation & Project Scope

### The "Silent Killer" Problem
Neonatal sepsis is a leading cause of mortality in newborns, characterized by a dysregulated immune response to infection. It is dubbed the "silent killer" because clinical symptoms (temperature instability, lethargy) are non-specific and overlap with non-infectious conditions.

### The Diagnostic Gap
*   **Gold Standard:** Blood culture.
*   **The Problem:** It takes **24-48 hours** to yield results. In accessible settings, this delay forces clinicians to prescribe "just-in-case" antibiotics, fueling antimicrobial resistance. In resource-limited settings, the delay can be fatal.
*   **The Need:** A rapid, accurate diagnostic tool that works *faster* than culture.

### The Innovation: Network Biology
We propose that **gene expression** offers a faster signal than bacterial growth. However, standard machine learning faces the "Curse of Dimensionality" (20,000 genes vs <100 samples).
**Our Solution:** Use **Graph Neural Networks (GNNs)** overlaid on **Protein-Protein Interaction (PPI) Networks**. By restricting the model to learn only from biologically interacting genes, we inject biological prior knowledge to constrain the search space and improve generalization.

---

## 2. Methodology: Data & Graph Construction

### 2.1 The "Frankenstein" Datasets (Data Engineering)
We aggregated transcriptomic data from three distinct studies to overcome small sample sizes (N).

| Dataset | Platform | Population | Role |
| :--- | :--- | :--- | :--- |
| **GSE25504** | Illumina GPL6947 & Affymetrix GPL570 | Neonates | Training (Batch A & B) |
| **GSE69686** | Affymetrix GPL570 | Neonates | Training (Batch C) |
| **GSE26440** | Affymetrix GPL570 | **Pediatric** (Children) | **External Validation** |

**Total Training N:** ~319 Neonates
**External Validation N:** ~130 Children

### 2.2 The Challenge: Platform Effects (Batch correction)
Merging Illumina and Affymetrix data creates massive "batch effects"—technical noise that overshadows biological signal.
*   **Initial Approach:** Simple merging. Resulted in the model distinguishing *chips*, not *diseases*.
*   **Current Approach (ComBat):** We use **ComBat Harmonization** to mathematically remove technical variation while preserving biological signal.
    *   *Crucial Detail:* We pass the biological condition (`mod=Condition`) to ComBat to ensure it doesn't "correct away" the sepsis signal itself.

### 2.3 Graph Construction
Each patient is represented as a unique graph data object:
1.  **Nodes:** We select the Top **2,000** genes with the highest **Median Absolute Deviation (MAD)** across the harmonized dataset. This focuses the model on the most biologically active genes.
2.  **Edges:** Defined by the **STRING v12 database**. We draw an edge if two genes have a combined interaction score **> 700** (High Confidence).
    *   *Result:* ~1,491 nodes, ~18,482 edges per graph.
3.  **Features:** The node features are the ComBat-corrected gene expression levels.

---

## 3. Experimental Evolution & Results

Our project evolved through three distinct phases, each learning from the failures of the last.

### Phase 1: The "GEO Approach" (Naive Merging)
**Methodology:**
*   Merged GSE25504 and GSE69686 raw expression data.
*   Trained a **Graph Attention Network (GAT)** with Domain Adversarial components (DANN).

**Results:**
*   **Internal Validation (Illumina):** AUROC **0.97** (Suspiciously perfect).
*   **External Validation (Affymetrix):** AUROC **0.50** (Random guessing).

**Conclusion:** The model learned "Shortcut Features". It memorized that "High Signal in Gene X = Illumina = Sepsis". It did not learn biology; it learned to identify the scanner.

### Phase 2: Optimization (GCN/GAT Tuning)
We switched to a rigorous internal validation scheme, splitting folds by platform to force the model to generalize. We compared **Graph Convolutional Networks (GCN)** and **Graph Attention Networks (GAT)**.

**Hyperparameter Tuning:**
We iterated through multiple configurations. 
*   *Iteration 1:* 500 genes, Threshold 0.9. (Too sparse, over-smoothing).
*   *Iteration 2 (Optimized):* Reduced Threshold to 0.7, increased to 2,000 genes, added **Edge Dropout (10%)** and **Feature Noise (0.1)** for augmentation.

**Optimized GCN Results:**
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Mean AUROC** | **0.685 ± 0.09** | Better than random, but high variance. |
| **Best Fold** | **0.810** | Shows the architecture *can* work on favorable splits. |
| **Worst Fold** | **0.578** | Shows instability when training data assumes different distributions. |

**Baseline Comparison:**
*   Logistic Regression: AUROC **0.82**.
*   *Lesson:* The simple linear model is currently beating the GNN. This suggests the topological signal (PPI) is not yet adding enough value to outweigh the complexity cost of the GNN.

### Phase 3: Structural Transfer Learning (The "Seizure" Hypothesis)
**Motivation:**
We hypothesized that **"Network Collapse"** is a universal phenomenon.
*   **Source Domain:** EEG Brain Networks (Seizure = Synchronization Collapse).
*   **Target Domain:** PPI Gene Networks (Sepsis = Immune Dysregulation).

**Methodology:**
1.  Pre-trained a GAT on **CHB-MIT EEG data** (Time-series correlation graphs).
2.  Transferred the weights of the deeper layers (`conv2`, `conv3`) to our Sepsis model.
3.  Evaluated "Frozen" (feature extraction) vs "Unfrozen" (fine-tuning) strategies.

**Results:**
*   **Frozen Transfer:** AUROC **0.52**.
    *   *Analysis:* Failure. The structural patterns in functional brain connectivity (correlation) do not map directly to static protein interactions (physical binding).
*   **Unfrozen Transfer:** AUROC **0.90** (Internal) / **0.33** (External).
    *   *Analysis:* Catastrophic forgetting / Overfitting. The model quickly discarded the EEG priors and memorized the training noise.

---

## 4. The Pediatric External Validation (GSE26440)
To rigorously test clinical utility, we reserved **GSE26440** (Pediatric Sepsis, N=130) as a "One-Shot" holdout.
*   **Why:** Children are biologically distinct from Neonates. If our model works here, it has learned a fundamental "Sepsis Core" signal, not just age-specific markers.
*   **Strategy:** This dataset affects *Batch Correction* (it is included in ComBat to align distributions) but is **never** seen by the GNN during training or tuning.

---

## 5. Current Project Status: The "Data-Centric" Pivot

We identified that **Model Architecture (GCN vs GAT)** matters less than **Data Quality (Batch Effects)**.

**Final Strategy (Reboot):**
1.  **Prioritize ComBat:** Ensure the input data is rigorously harmonized before it touches the graph.
2.  **Simplify:** Drop the complex DANN/Transfer architectures. Use a clean, optimized GCN on the ComBat-corrected data.
3.  **Target:** Beat the Logistic Regression baseline (0.82) by leveraging the PPI topology to filter out spurious noise. 

**Summary:**
This project demonstrates that while GNNs are powerful, they are not magic. Without rigorous data engineering (ComBat) and careful validation (External Holdouts), they easily overfit to technical artifacts in small biological datasets. Our current optimized approach balances these factors, aiming for a robust, clinically translatable diagnostic tool.
