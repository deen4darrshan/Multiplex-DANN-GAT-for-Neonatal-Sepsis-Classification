# MASTER DIRECTIVE: ACSEF Science Fair Completion & Deliverables Generation

You are an expert AI Data Scientist and Bioinformatics Researcher completing a high-stakes competitive project for the ACSEF Science Fair. You are working in the `ppi_gnn_combined_dataset` repository. 

**Your ultimate responsibility** is to finish the implementation of this project and record the final documentation. This prompt is a guide—you must go above and beyond these specifications to build a winning presentation.

---

## 🛑 Rules of Engagement & Directory Structure
1. **Explore Everything:** You MUST look at the ENTIRE repository. Explore older files, original approaches, and unused scripts so you don't miss anything that could strengthen the project's narrative.
2. **Workspace:** You must perform all of your final compilation work in a dedicated, newly created subdirectory named `ACSEF_Final_Submission`. 
3. **Organization:** Maintain a meticulously clean folder structure inside `ACSEF_Final_Submission` (`/data`, `/scripts`, `/models`, `/results`, `/figures`, `/acsef_documents`).
4. **Dynamic Naming Scheme:** "V11" or "V8" are internal development names. **Develop a new, descriptive, and scientific naming scheme** for the models (e.g., "Multiplex-GNN-DANN", "Baseline-HGCN") and apply it to all generated reports and figures.
5. **Logs & Notebooks:** All execution logs and created notebooks must be explicitly dated and highly descriptive. They must detail exactly what was done, what failures occurred, and how they were solved.
6. **Explain Failures:** If a script fails (e.g., OOM errors, deadlocks on Windows with SciPy/Pandas during XAI), explain *why* it failed in a titled `failure_analysis_log.md` file, date it, and document how you engineered around it.

---

## Phase 1: Finalize XAI & Biomarker Discovery
The XAI pipeline using Integrated Gradients (IG) on the V11 model was previously aborted due to severe PyTorch/CUDA environment deadlocks and memory swapping on Windows.
* **Task:** Re-develop, robustify, and fully execute the biomarker extraction scripts.
* **Requirements:**
  * Avoid `scipy.stats.rankdata` or `median_abs_deviation` if they natively deadlock the Python 3.13 interpreter; implement pure NumPy/Pandas equivalents. Load the 11M row STRING database in chunks to prevent swap-thrashing.
  * Extract the `gene_scores` (GNN attention mask) and use a custom PyTorch Integrated Gradients implementation.
  * **Output:** Generate `top_100_biomarkers.csv`.
  * **Visuals:** Generate diverging bar charts for the top 20 biomarkers. **Go beyond the prompt:** generate extra figures like PCA projections of the top biomarkers, data normalization distribution curves, or correlation heatmaps.

---

## Phase 2: Results Aggregation, Baseline Justification & External Validation
* **Task:** Compile all benchmarking results from previous models to justify the final architecture.
* **Requirements:**
  * Gather results from baseline models (HGCN, GCN, GAT). **Note it is HGCN, not NGCN.**
  * Compile all metrics into a single JSON file (`compiled_model_metrics.json`) inside the `/results/` directory.
  * **Justification Document:** Write a `model_justification_and_architecture.md` file that:
    1. Compares the final model against the HGCN, GCN, and GAT baselines.
    2. Explains *why* the MLP integration was critical (i.e., pure V12 HGCN models collapsed to random chance without the MLP extracting non-linear features).
    3. Details the architecture (Multiplex Hypergraph Conv on 3 relations -> Relation Attention -> Gene Scorer -> MLP classifier + Domain Adversarial head).
  * **External Validation:** You MUST explicitly document and interpret the model's performance on the external validation dataset (GSE26440).

---

## Phase 3: Scaling to Rare Diseases (Osteogenesis Imperfecta)
* **Task:** A new folder named `osteogenesis_imperfecta` exists where the architecture was scaled to another rare disease.
* **Requirements:**
  * Read the files and results in that directory.
  * Create a dedicated section/document proving that our architecture can be generalized to other rare biological conditions.
  * Interpret its results, findings, and methodology, and incorporate this highly impressive scaling ability into the final ACSEF narrative.

---

## Phase 4: ACSEF Deliverable Generation
Generate the text and layout instructions for the required ACSEF deliverables strictly adhering to the formatting rules (no active links, APA citations).

### 1. Official Abstract (Max 250 words)
* Create `acsef_official_abstract.txt`.
* Must include: (a) research problem, (b) procedures, (c) hard data/AUROC numbers, (d) interpretation, (e) conclusions.

### 2. Quad Chart (Visual Summary)
* Create `acsef_quad_chart_content.md`. (Max 75 words per quadrant).
* **Quadrants:** Q1: Scientific Question, Q2: Methodology (include architecture visual), Q3: Data Analysis & Results (include ROC curves/External Validation), Q4: Interpretation & Conclusions (Top biomarkers & Osteogenesis Imperfecta scaling).

### 3. Virtual Display / Project Presentation (12 Slides Max)
* Create `acsef_project_presentation.md` documenting out the 12 slides.
* **Rules:** APA Citations required for EVERY graphic (e.g., "Graph generated by [Student Name] on [Date]"). Standard fonts.
* **Sections:** Title, Introduction, Methods, Results (include data tables), Discussion (interpret failures like OOM crashes), Conclusions (emphasize generalizability), References (APA, acknowledge AI).

### 4. Required Visuals & Figures (Proactive Generation)
Generate Python scripts to build these required graphics, execute them, and save them to `/figures/`:
* `fig_roc_comparisons.png` (Final Model vs. GCN/GAT/HGCN).
* `fig_architecture_flowchart.png`.
* `fig_biomarker_attributions.png`.
* **Proactive:** Generate any other compelling scientific figures you can think of (e.g., batch-correction PCA, attention distribution heatmaps, data normalization steps). 

---
**EXECUTE ALL OF THE ABOVE TASKS COMPLETELY.** You are the master agent responsible for the final polish. Do not stop until the `ACSEF_Final_Submission` folder represents a winning science fair entry.
