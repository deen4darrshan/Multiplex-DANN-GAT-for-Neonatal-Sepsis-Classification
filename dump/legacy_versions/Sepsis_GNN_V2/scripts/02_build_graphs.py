"""
Phase 2: Variance Filtering + Graph Construction
=================================================
This script:
1. Loads ComBat-corrected expression data
2. Selects Top-K most variable genes (MAD-based)
3. Loads STRING v12 PPI network
4. Filters to genes with PPI edges
5. Constructs PyTorch Geometric Data objects per patient
6. Saves training and external graph datasets

CoV Verification:
- Print gene counts at each filtering stage
- Assert graph statistics within expected ranges
- Print edge count, avg degree, connected components
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import gzip
from scipy.stats import median_abs_deviation
import torch
from torch_geometric.data import Data

# ---------- Config ----------
TOP_K = 1000             # Variance-filtered gene count
STRING_THRESH = 700      # STRING confidence threshold (0-1000)
V2_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
V1_ROOT = os.path.abspath(os.path.join(V2_ROOT, '..'))
PROCESSED_DIR = os.path.join(V2_ROOT, 'data', 'processed')
RAW_DIR = os.path.join(V1_ROOT, 'data', 'raw')
STRING_FILE = os.path.join(RAW_DIR, '9606.protein.links.v12.0.txt.gz')

# ============================================================
# STEP 1: Load ComBat-Corrected Data
# ============================================================
def load_combat_data():
    """Load combat-corrected expression and metadata."""
    train_expr = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_expression_combat.csv'), index_col=0)
    ext_expr = pd.read_csv(os.path.join(PROCESSED_DIR, 'external_expression_combat.csv'), index_col=0)
    train_meta = pd.read_csv(os.path.join(PROCESSED_DIR, 'train_metadata.csv'))
    ext_meta = pd.read_csv(os.path.join(PROCESSED_DIR, 'external_metadata.csv'))
    
    print(f"  Training: {train_expr.shape[0]} genes × {train_expr.shape[1]} samples")
    print(f"  External: {ext_expr.shape[0]} genes × {ext_expr.shape[1]} samples")
    
    return train_expr, ext_expr, train_meta, ext_meta

# ============================================================
# STEP 2: MAD-Based Variance Filtering
# ============================================================
def variance_filter(expression, k=TOP_K):
    """Select top-K most variable genes using Median Absolute Deviation."""
    mad_scores = expression.apply(median_abs_deviation, axis=1)
    mad_scores = mad_scores.sort_values(ascending=False)
    
    top_genes = mad_scores.head(k).index.tolist()
    
    print(f"  MAD filtering: {expression.shape[0]} → {len(top_genes)} genes")
    print(f"  MAD range: {mad_scores.iloc[0]:.4f} (max) → {mad_scores.iloc[k-1]:.4f} (cutoff)")
    
    return top_genes, mad_scores

# ============================================================
# STEP 3: Load STRING PPI Network
# ============================================================
def load_string_network(gene_list, threshold=STRING_THRESH):
    """Load STRING v12 and filter to our gene set."""
    
    # First, build ENSP → Gene Symbol mapping
    # We'll use the STRING aliases file or a simple approach
    print(f"  Loading STRING network (threshold >= {threshold})...")
    
    # Load STRING links
    edges = []
    all_proteins = set()
    
    with gzip.open(STRING_FILE, 'rt') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            p1, p2, score = parts[0], parts[1], int(parts[2])
            if score >= threshold:
                # Strip species prefix: '9606.ENSP00000...' -> 'ENSP00000...'
                p1 = p1.replace('9606.', '')
                p2 = p2.replace('9606.', '')
                edges.append((p1, p2, score))
                all_proteins.add(p1)
                all_proteins.add(p2)
    
    print(f"  STRING total edges (>= {threshold}): {len(edges)}")
    print(f"  STRING total proteins: {len(all_proteins)}")
    
    return edges, all_proteins

def map_ensp_to_genes(proteins, gene_list):
    """Map Ensembl protein IDs to gene symbols using mygene."""
    import mygene
    mg = mygene.MyGeneInfo()
    
    protein_list = list(proteins)
    print(f"  Mapping {len(protein_list)} ENSP IDs to gene symbols...")
    
    # Query in batches
    batch_size = 1000
    ensp_to_gene = {}
    
    for i in range(0, len(protein_list), batch_size):
        batch = protein_list[i:i+batch_size]
        results = mg.querymany(batch, scopes='ensembl.protein', 
                               fields='symbol', species='human',
                               returnall=False, verbose=False)
        for r in results:
            if 'symbol' in r:
                ensp_to_gene[r['query']] = r['symbol']
    
    # Filter to our gene list
    gene_set = set(gene_list)
    mapped = {k: v for k, v in ensp_to_gene.items() if v in gene_set}
    print(f"  Mapped to our gene set: {len(mapped)} proteins → genes")
    
    return ensp_to_gene, mapped

def build_gene_edges(string_edges, ensp_to_gene, gene_list):
    """Convert protein edges to gene-level edges."""
    gene_set = set(gene_list)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    
    edges_src, edges_dst = [], []
    seen = set()
    
    for p1, p2, score in string_edges:
        g1 = ensp_to_gene.get(p1)
        g2 = ensp_to_gene.get(p2)
        
        if g1 and g2 and g1 in gene_set and g2 in gene_set and g1 != g2:
            edge_key = tuple(sorted([g1, g2]))
            if edge_key not in seen:
                seen.add(edge_key)
                i, j = gene_to_idx[g1], gene_to_idx[g2]
                edges_src.extend([i, j])  # undirected
                edges_dst.extend([j, i])
    
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    
    return edge_index, seen

# ============================================================
# STEP 4: Graph Construction (Multi-Feature)
# ============================================================
def compute_graph_features(edge_index, n_nodes):
    """Compute structural features: degree and clustering coefficient."""
    # Degree
    degree = torch.zeros(n_nodes, dtype=torch.float)
    for i in range(edge_index.shape[1]):
        degree[edge_index[0, i]] += 1
    
    # Clustering coefficient (fraction of triangles)
    adj = {}
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj.setdefault(src, set()).add(dst)
    
    clustering = torch.zeros(n_nodes, dtype=torch.float)
    for node in range(n_nodes):
        neighbors = adj.get(node, set())
        k = len(neighbors)
        if k < 2:
            continue
        # Count edges between neighbors
        triangles = 0
        neighbors_list = list(neighbors)
        for i_n in range(len(neighbors_list)):
            for j_n in range(i_n + 1, len(neighbors_list)):
                if neighbors_list[j_n] in adj.get(neighbors_list[i_n], set()):
                    triangles += 1
        clustering[node] = 2.0 * triangles / (k * (k - 1))
    
    return degree, clustering


def create_patient_graphs(expression, metadata, gene_list, edge_index, 
                          mad_scores, degree, clustering):
    """Create PyG Data objects with 4D node features per patient.
    
    Features per node: [expression, MAD_rank, degree, clustering_coeff]
    - expression: z-scored per sample (patient-specific)
    - MAD_rank: normalized rank by variance (static across patients)
    - degree: PPI node degree (static across patients)
    - clustering: PPI clustering coefficient (static across patients)
    """
    n_nodes = len(gene_list)
    
    # Static features (same for all patients)
    # MAD rank: normalize to [0, 1] by rank position
    gene_mads = np.array([mad_scores.get(g, 0.0) for g in gene_list], dtype=np.float32)
    mad_ranks = np.argsort(np.argsort(-gene_mads)).astype(np.float32) / max(n_nodes - 1, 1)
    
    # Normalize degree and clustering to [0, 1]
    deg_np = degree.numpy()
    deg_norm = deg_np / max(deg_np.max(), 1.0)
    clust_np = clustering.numpy()
    
    graphs = []
    
    for _, row in metadata.iterrows():
        sample_id = row['SampleID']
        label = int(row['Label'])
        
        # Node feature 1: expression (z-scored per sample)
        expr_vals = expression.loc[gene_list, sample_id].values.astype(np.float32)
        mean = np.mean(expr_vals)
        std = np.std(expr_vals)
        if std > 1e-8:
            expr_vals = (expr_vals - mean) / std
        
        # Stack 4 features: [expression, mad_rank, degree, clustering]
        features = np.stack([expr_vals, mad_ranks, deg_norm, clust_np], axis=1)  # [N, 4]
        
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data.sample_id = sample_id
        data.batch_label = row.get('Batch', 'Unknown')
        
        graphs.append(data)
    
    return graphs

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print("=" * 70)
    print("PHASE 2: Variance Filtering + Graph Construction (Multi-Feature)")
    print("=" * 70)
    
    # --- Load ---
    print("\n[1/5] Loading ComBat-corrected data...")
    train_expr, ext_expr, train_meta, ext_meta = load_combat_data()
    
    # --- Variance Filter ---
    print("\n[2/5] MAD-based variance filtering...")
    top_genes, mad_scores = variance_filter(train_expr, k=TOP_K)
    
    # Save gene list
    gene_list_path = os.path.join(PROCESSED_DIR, 'top_genes.txt')
    with open(gene_list_path, 'w') as f:
        for g in top_genes:
            f.write(g + '\n')
    print(f"  Saved gene list: {gene_list_path}")
    
    # --- Load STRING ---
    print("\n[3/5] Loading STRING PPI network...")
    string_edges, all_proteins = load_string_network(top_genes)
    
    # Map ENSP → Gene
    ensp_to_gene_all, ensp_to_gene_filtered = map_ensp_to_genes(all_proteins, top_genes)
    
    # Build gene-level edges
    edge_index, unique_edges = build_gene_edges(string_edges, ensp_to_gene_all, top_genes)
    
    # --- Filter genes to those WITH edges ---
    nodes_with_edges = set()
    for g1, g2 in unique_edges:
        nodes_with_edges.add(g1)
        nodes_with_edges.add(g2)
    
    connected_genes = [g for g in top_genes if g in nodes_with_edges]
    print(f"\n  === CoV: Graph Statistics ===")
    print(f"  Top-K genes: {len(top_genes)}")
    print(f"  Genes with PPI edges: {len(connected_genes)}")
    print(f"  Gene edges (unique undirected): {len(unique_edges)}")
    print(f"  Edge index tensor shape: {edge_index.shape}")
    
    if len(connected_genes) < 100:
        print(f"  WARNING: Only {len(connected_genes)} connected genes. Using all Top-K genes instead.")
        connected_genes = top_genes
    
    # Rebuild edges for the final gene list
    final_gene_list = connected_genes
    gene_to_idx = {g: i for i, g in enumerate(final_gene_list)}
    
    # Rebuild edge_index for the potentially smaller gene list
    edges_src, edges_dst = [], []
    for g1, g2 in unique_edges:
        if g1 in gene_to_idx and g2 in gene_to_idx:
            i, j = gene_to_idx[g1], gene_to_idx[g2]
            edges_src.extend([i, j])
            edges_dst.extend([j, i])
    
    final_edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    
    n_nodes = len(final_gene_list)
    n_edges = final_edge_index.shape[1] // 2  # undirected
    avg_degree = final_edge_index.shape[1] / max(n_nodes, 1)
    
    print(f"\n  Final graph topology:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges (undirected): {n_edges}")
    print(f"  Avg degree: {avg_degree:.1f}")
    
    # Verification
    assert n_nodes >= 100, f"FAIL: Too few nodes ({n_nodes}). Check STRING mapping."
    assert n_edges >= 100, f"FAIL: Too few edges ({n_edges}). Check STRING threshold."
    print(f"  ✓ PASS: Graph topology within expected range")
    
    # --- Compute structural features ---
    print("\n  Computing structural node features...")
    degree, clustering = compute_graph_features(final_edge_index, n_nodes)
    print(f"  Degree: mean={degree.mean():.2f}, max={degree.max():.0f}")
    print(f"  Clustering: mean={clustering.mean():.4f}, max={clustering.max():.4f}")
    
    # --- Create Graphs ---
    print("\n[4/5] Creating patient graphs (4D features)...")
    
    # Save final gene list
    with open(os.path.join(PROCESSED_DIR, 'final_genes.txt'), 'w') as f:
        for g in final_gene_list:
            f.write(g + '\n')
    
    # Create MAD score dict for the function
    mad_dict = mad_scores.to_dict()
    
    train_graphs = create_patient_graphs(train_expr, train_meta, final_gene_list, 
                                          final_edge_index, mad_dict, degree, clustering)
    ext_graphs = create_patient_graphs(ext_expr, ext_meta, final_gene_list,
                                        final_edge_index, mad_dict, degree, clustering)
    
    print(f"  Training graphs: {len(train_graphs)}")
    print(f"  External graphs: {len(ext_graphs)}")
    
    # Verification
    assert len(train_graphs) == len(train_meta), "FAIL: Graph count mismatch!"
    assert len(ext_graphs) == len(ext_meta), "FAIL: Graph count mismatch!"
    
    N_FEATURES = 4
    g0 = train_graphs[0]
    print(f"\n  === CoV: Sample Graph ===")
    print(f"  x shape: {g0.x.shape} (expected: [{n_nodes}, {N_FEATURES}])")
    print(f"  edge_index shape: {g0.edge_index.shape}")
    print(f"  label: {g0.y.item()}")
    print(f"  sample_id: {g0.sample_id}")
    print(f"  Feature names: [expression, MAD_rank, degree, clustering]")
    
    assert g0.x.shape == torch.Size([n_nodes, N_FEATURES]), f"FAIL: Wrong feature shape {g0.x.shape}"
    print(f"  ✓ PASS: Graph structure verified (4D features)")
    
    # Label distribution
    train_labels = [g.y.item() for g in train_graphs]
    ext_labels = [g.y.item() for g in ext_graphs]
    print(f"\n  Training labels: {sum(train_labels)} Sepsis, {len(train_labels)-sum(train_labels)} Control")
    print(f"  External labels: {sum(ext_labels)} Sepsis, {len(ext_labels)-sum(ext_labels)} Control")
    
    # --- Save ---
    print("\n[5/5] Saving graph datasets...")
    
    torch.save(train_graphs, os.path.join(PROCESSED_DIR, 'train_graphs.pt'))
    torch.save(ext_graphs, os.path.join(PROCESSED_DIR, 'external_graphs.pt'))
    
    # Save graph metadata
    graph_meta = {
        'gene_list': final_gene_list,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'edge_index': final_edge_index,
        'avg_degree': avg_degree,
        'string_threshold': STRING_THRESH,
        'top_k': TOP_K,
    }
    with open(os.path.join(PROCESSED_DIR, 'graph_metadata.pkl'), 'wb') as f:
        pickle.dump(graph_meta, f)
    
    # --- Final Summary ---
    print(f"\n{'=' * 70}")
    print("PHASE 2 COMPLETE: Graph Construction")
    print(f"{'=' * 70}")
    print(f"  Gene filtering:    {train_expr.shape[0]} → {TOP_K} (MAD) → {n_nodes} (PPI connected)")
    print(f"  Graph topology:    {n_nodes} nodes, {n_edges} edges, avg degree {avg_degree:.1f}")
    print(f"  Training graphs:   {len(train_graphs)}")
    print(f"  External graphs:   {len(ext_graphs)}")
    print(f"  Files saved to:    {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
