"""
Phase 5: V11 Explainable AI (XAI) & Biomarker Extraction
====================================================================
Uses Captum Integrated Gradients and the internal GNN `gene_scores`
mask to extract the top genes (features) driving Sepsis classification 
in the V11 Multiplex DANN model.
"""

import os, sys, json, warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

# Import V11 architecture and functions
sys.path.insert(0, os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "scripts"))
import importlib
train_module = importlib.import_module('19_train_v11_multiplex_dann')
MultiplexGNNGuidedDANN = train_module.MultiplexGNNGuidedDANN
make_data_list = train_module.make_data_list
collate_multiplex = train_module.collate_multiplex

# ============================================================================
# CONFIGURATION
# ============================================================================
OUT_DIR     = os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "results")
MODEL_DIR   = os.path.join(PROJECT_ROOT, "CH_DANN_Plan", "models")
PROC_DIR    = os.path.join(PROJECT_ROOT, "data", "processed")
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TOP_K = 2000
H_DIM = 64
DROPOUT = 0.3
STRING_THR = 700

def build_string_hyperedges(gene_list):
    gene_set = set(gene_list)
    g2i = {g: i for i, g in enumerate(gene_list)}
    ppi_path = os.path.join(PROC_DIR, "ppi_network.csv")
    ni, hi, hid = [], [], 0

    if os.path.exists(ppi_path):
        print("    Loading STRING PPI edges in chunks...")
        for chunk in pd.read_csv(ppi_path, chunksize=250000):
            pf = chunk[(chunk['source'].isin(gene_set)) &
                      (chunk['target'].isin(gene_set)) &
                      (chunk['score'] >= STRING_THR)]
            
            if len(pf) == 0: continue
            
            sources = pf['source'].values
            targets = pf['target'].values
            for s, t in zip(sources, targets):
                if s in g2i and t in g2i:
                    ni.append(g2i[s]); hi.append(hid)
                    ni.append(g2i[t]); hi.append(hid)
                    hid += 1

    if ni:
        return torch.tensor([ni, hi], dtype=torch.long), hid
    return torch.zeros(2, 0, dtype=torch.long), 0

def build_coexpr_hyperedges(expr_f, gene_list, sample_ids):
    g2i = {g: i for i, g in enumerate(gene_list)}
    sub_expr = expr_f[sample_ids]
    
    # Use Pandas rank to completely avoid SciPy C-extension deadlocks on Windows
    ranked_df = sub_expr.rank(axis=1)
    ranked = ranked_df.values
    
    ranked = (ranked - ranked.mean(axis=1, keepdims=True)) / (ranked.std(axis=1, keepdims=True) + 1e-8)
    corr = ranked @ ranked.T / ranked.shape[1]
    
    np.fill_diagonal(corr, 0)
    pairs = np.argwhere(np.abs(corr) > 0.7)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    
    ni, hi, hid = [], [], 0
    for (i, j) in pairs:
        ni.append(i); hi.append(hid)
        ni.append(j); hi.append(hid)
        hid += 1
        
    if ni:
        return torch.tensor([ni, hi], dtype=torch.long), hid
    return torch.zeros(2, 0, dtype=torch.long), 0

def build_kegg_hyperedges(gene_list):
    gene_set = set(gene_list)
    g2i = {g: i for i, g in enumerate(gene_list)}
    pw = {}
    
    # Force use of cached file instead of hanging on Enrichr API via gseapy
    info_path = os.path.join(OUT_DIR, "pathway_info_v2.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            kegg = json.load(f)
            for p, dict_val in kegg.items():
                genes = dict_val.get('genes', []) if isinstance(dict_val, dict) else dict_val
                ol = list(set(genes) & gene_set)
                if len(ol) >= 3:
                    pw[p] = ol
                    
    ni, hi, hid = [], [], 0
    for genes in pw.values():
        for g in genes:
            if g in g2i:
                ni.append(g2i[g]); hi.append(hid)
        hid += 1

    if ni:
        return torch.tensor([ni, hi], dtype=torch.long), len(pw)
    return torch.zeros(2, 0, dtype=torch.long), 0

def extract_biomarkers():
    print("=" * 80)
    print("  EXPLAINABLE AI: Extracting V11 Sepsis Biomarkers")
    print("=" * 80)

    # 1. Load Data & Identify Top 2000 Genes (matches training)
    print("\n[1/5] Extracting Shared Genes and Graph Topology...")
    train_expr_path = os.path.join(OUT_DIR, "expression_combat_v2.csv")
    train_meta_path = os.path.join(OUT_DIR, "metadata_v2.csv")
    gene_list_path = os.path.join(OUT_DIR, "gene_list_v2.json")
    
    expr_train = pd.read_csv(train_expr_path, index_col=0)
    meta_train = pd.read_csv(train_meta_path)
    
    if os.path.exists(gene_list_path):
        with open(gene_list_path, 'r') as f:
            top_genes = json.load(f)
    else:
        mad = expr_train.apply(median_abs_deviation, axis=1)
        top_genes = mad.sort_values(ascending=False).head(TOP_K).index.tolist()
    
    kegg_hei, _ = build_kegg_hyperedges(top_genes)
    string_hei, _ = build_string_hyperedges(top_genes)
    
    # Co-Expression uses training set only
    train_sids = expr_train.columns.tolist()
    coexpr_hei, _ = build_coexpr_hyperedges(expr_train.loc[top_genes], top_genes, train_sids)

    # 2. Prepare the Evaluation Dataset (We will use the Internal Training set for vast feature coverage, 
    # but we only explain the SEPSIS subset to find Sepsis biomarkers)
    print("\n[2/5] Preparing Cohort for Explanation...")
    data_list = make_data_list(expr_train.loc[top_genes], meta_train, top_genes, kegg_hei, string_hei)
    
    # Keep only Sepsis patients (Label = 1)
    sepsis_data = [d for d in data_list if d.y.item() == 1]
    n_sepsis = len(sepsis_data)
    print(f"  Explaining over {n_sepsis} Sepsis patients.")

    # 3. Load Model
    print("\n[3/5] Loading V11 Best Model Weights...")
    model_path = os.path.join(MODEL_DIR, "v11_multiplex_dann_best.pt")
    model = MultiplexGNNGuidedDANN(n_genes=len(top_genes), h_dim=H_DIM, dropout=DROPOUT).to(DEVICE)
    
    # Handle DANN head compatibility
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    dom_weight = state_dict.get('domain_discriminator.4.weight')
    if dom_weight is not None:
        model.domain_discriminator[4] = torch.nn.Linear(H_DIM, dom_weight.shape[0]).to(DEVICE)
    
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Global Gene Importance Mask Extraction (GNN Attention)
    print("\n[4/5] Extracting Graph-Guided Gene Attention Masks...")
    # The pure GNN mechanism provides a 0-1 score for every gene. We need to spoof the forward 
    # pass to return these intermediate scores.
    def forward_with_mask(model, x, hedge_indices, batch, global_feat=None):
        n_nodes_per_graph = model.n_genes
        batch_size = batch.max().item() + 1
        g = model.gene_embed(x)
        rel_outputs = []
        for i in range(model.n_relations):
            hei = hedge_indices[i]
            if hei is not None and hei.size(1) > 0:
                h = model.convs1[i](g, hei)
                h = model.lns1[i](h); h = F.gelu(h)
                r = g + h
                h = model.convs2[i](r, hei)
                h = model.lns2[i](h); h = F.gelu(h)
                r = r + h
                rel_outputs.append(r)
            else:
                rel_outputs.append(g)
        stacked = torch.stack(rel_outputs, dim=1)
        concat = torch.cat(rel_outputs, dim=1)
        attn_logits = model.relation_attn(concat)
        attn_weights = F.softmax(attn_logits, dim=1)
        h_multi = (stacked * attn_weights.unsqueeze(2)).sum(dim=1)
        gene_scores = torch.sigmoid(model.gene_scorer(h_multi))
        
        scores_per_graph = gene_scores.view(batch_size, n_nodes_per_graph)
        weighted_expr = global_feat * scores_per_graph if global_feat is not None else None
        
        return model.classifier(model.mlp(weighted_expr)), scores_per_graph

    # Run batched inference to extract average GNN masks
    batch_size_gnn = 16
    all_gene_scores = []
    
    with torch.no_grad():
        for i in range(0, len(sepsis_data), batch_size_gnn):
            print(f"    GNN Batch {i//batch_size_gnn + 1}/{(len(sepsis_data)-1)//batch_size_gnn + 1}...")
            batch_data = sepsis_data[i:i+batch_size_gnn]
            mbatch = collate_multiplex(batch_data, coexpr_hei).to(DEVICE)
            global_expr_batch_sub = mbatch.x.squeeze(1).view(-1, len(top_genes))
            _, gene_scores_batch = forward_with_mask(model, mbatch.x, mbatch.hedge_indices, mbatch.batch, global_expr_batch_sub)
            all_gene_scores.append(gene_scores_batch)
            
        gene_scores_batch_all = torch.cat(all_gene_scores, dim=0)
        mean_gene_scores = gene_scores_batch_all.mean(dim=0).cpu().numpy()
        
    global_expr_batch = torch.cat([d.x.squeeze(1).unsqueeze(0) for d in sepsis_data], dim=0).to(DEVICE)

    # 5. Integrated Gradients (Custom PyTorch Implementation)
    print("\n[5/5] Computing Network Path Attributions (Integrated Gradients)...")
    
    # Custom IG Implementation
    def integrated_gradients(inputs, baseline, fixed_scores, target_class_idx, steps=50):
        # We only attribute the MLP since the GNN mask is static with respect to the input
        class CustomWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, expr_input, scores):
                weighted_expr = expr_input * scores
                return self.model.classifier(self.model.mlp(weighted_expr))
                
        wrapper = CustomWrapper(model)
        wrapper.eval()
        
        # Scale inputs (Shape: [steps+1, batch_size, n_genes])
        scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
        scaled_inputs = torch.cat(scaled_inputs, dim=0).requires_grad_()
        
        # Repeat fixed_scores to match scaled_inputs
        repeated_scores = fixed_scores.repeat(steps + 1, 1)
        
        preds = wrapper(scaled_inputs, repeated_scores)
        target_preds = preds[:, target_class_idx]
        
        wrapper.zero_grad()
        target_preds.sum().backward()
        
        grads = scaled_inputs.grad
        num_examples = inputs.shape[0]
        
        # grads is (steps+1 * batch_size, n_genes), sequenced by step then batch
        grads = grads.view(steps + 1, num_examples, -1).transpose(0, 1) # now (batch_size, steps+1, n_genes)
        
        # Trapezoidal rule per patient
        avg_grads = (grads[:, :-1, :] + grads[:, 1:, :]) / 2.0  # (batch_size, steps, n_genes)
        avg_grads = avg_grads.mean(dim=1)  # (batch_size, n_genes)
        
        return (inputs - baseline) * avg_grads

    baseline = torch.zeros_like(global_expr_batch).to(DEVICE)
    

    batch_size_ig = 16
    all_attributions = []
    
    for i in range(0, global_expr_batch.shape[0], batch_size_ig):
        print(f"    IG Batch {i//batch_size_ig + 1}/{(global_expr_batch.shape[0]-1)//batch_size_ig + 1}...")
        end_idx = min(i + batch_size_ig, global_expr_batch.shape[0])
        batch_inputs = global_expr_batch[i:end_idx]
        batch_baseline = baseline[i:end_idx]
        batch_fixed_scores = gene_scores_batch_all[i:end_idx]
        
        batch_attr = integrated_gradients(batch_inputs, batch_baseline, batch_fixed_scores, target_class_idx=1, steps=50)
        all_attributions.append(batch_attr)
        
    attributions = torch.cat(all_attributions, dim=0)
    mean_attributions = attributions.mean(dim=0).detach().cpu().numpy()


    # 6. Aggregate XAI Results
    results_df = pd.DataFrame({
        'Gene': top_genes,
        'GNN_Attention_Mask': mean_gene_scores,
        'MLP_IG_Attribution': mean_attributions
    })
    
    # Absolute attribution tells us overall importance magnitude
    results_df['Absolute_IG_Attribution'] = results_df['MLP_IG_Attribution'].abs()
    
    # Sort by absolute attribution for topmost important driving genes
    results_df = results_df.sort_values(by='Absolute_IG_Attribution', ascending=False)
    
    # Save raw CSV
    csv_path = os.path.join(OUT_DIR, "v11_biomarkers.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Extracted biomarkers saved to {csv_path}")

    # 7. Generate Visualizations (CoVe)
    print("  Generating Visualization plots...")
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    
    # Plot top 20 positive and negative attributes
    top_20 = results_df.head(20)
    top_20 = top_20.sort_values('MLP_IG_Attribution') # Sort for visually diverging bar plot
    
    colors = ['#d62728' if x > 0 else '#1f77b4' for x in top_20['MLP_IG_Attribution']]
    
    plt.barh(top_20['Gene'], top_20['MLP_IG_Attribution'], color=colors)
    plt.axvline(0, color='black', linewidth=1.2)
    plt.xlabel('Integrated Gradient Attribution Score (Contribution to Sepsis Diagnosis)', fontsize=12)
    plt.title('Top 20 Transciptomic Biomarkers Driving Neonatal Sepsis (V11 XAI)', fontsize=14, pad=15)
    
    # Save Plot
    plot_path = os.path.join(OUT_DIR, "v11_biomarkers_barplot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"✅ V11 Biomarker plot saved to {plot_path}")

if __name__ == "__main__":
    extract_biomarkers()
