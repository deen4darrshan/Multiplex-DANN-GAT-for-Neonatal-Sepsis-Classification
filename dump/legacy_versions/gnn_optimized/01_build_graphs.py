"""
GNN Optimized - Step 1: Graph Construction

Creates PPI network and patient graphs with optimized parameters:
- STRING threshold: 0.7 (700)
- Top 2000 variance genes
- 3D node features (Expression, Degree Centrality, Gene Variance)
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import os
import pickle
import gzip
import networkx as nx

# ============================================================
# CONFIGURATION
# ============================================================
NUM_GENES = 2000           # Top variance genes to select
STRING_THRESHOLD = 700     # Confidence score threshold (0.7)
MIN_AVG_DEGREE = 5.0       # Minimum average degree for adaptive threshold

# Paths
DATA_DIR = "../data/processed"
RAW_DIR = "../data/raw"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("GNN OPTIMIZED - GRAPH CONSTRUCTION")
print("=" * 60)
print(f"Configuration:")
print(f"  - Top variance genes: {NUM_GENES}")
print(f"  - STRING threshold: {STRING_THRESHOLD} (confidence > {STRING_THRESHOLD/1000:.1f})")
print(f"  - Min average degree: {MIN_AVG_DEGREE}")
print("=" * 60)


def load_string_network(filepath, score_threshold):
    """Load and filter STRING network."""
    print(f"\n[1/6] Loading STRING network...")
    
    edges = []
    with gzip.open(filepath, 'rt') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                protein1 = parts[0]
                protein2 = parts[1]
                score = int(parts[2])
                
                if score >= score_threshold:
                    edges.append((protein1, protein2, score))
    
    print(f"  Loaded {len(edges):,} edges with score >= {score_threshold}")
    return edges


def map_proteins_to_genes(edges):
    """Map STRING protein IDs to gene symbols."""
    import mygene
    
    print(f"\n[2/6] Mapping proteins to gene symbols...")
    
    proteins = set()
    for p1, p2, _ in edges:
        proteins.add(p1.replace('9606.', ''))
        proteins.add(p2.replace('9606.', ''))
    
    print(f"  Unique proteins: {len(proteins):,}")
    
    mg = mygene.MyGeneInfo()
    results = mg.querymany(list(proteins), scopes='ensembl.protein', 
                          fields='symbol', species='human', returnall=True, verbose=False)
    
    ensembl_to_symbol = {}
    for item in results['out']:
        if 'symbol' in item and 'query' in item:
            ensembl_to_symbol[item['query']] = item['symbol']
    
    print(f"  Mapped {len(ensembl_to_symbol):,} proteins to symbols")
    return ensembl_to_symbol


def select_top_variance_genes(expression, top_k):
    """Select top-K genes by variance."""
    print(f"\n[3/6] Selecting top {top_k} variance genes...")
    
    gene_variance = expression.var(axis=1)
    top_genes = gene_variance.nlargest(top_k).index.tolist()
    
    print(f"  Total genes: {len(expression):,}")
    print(f"  Selected: {len(top_genes):,}")
    print(f"  Variance range: {gene_variance[top_genes].min():.4f} - {gene_variance[top_genes].max():.4f}")
    
    return set(top_genes), gene_variance


def build_filtered_network(edges, ensembl_to_symbol, gene_set):
    """Build network filtered to selected genes."""
    print(f"\n[4/6] Building filtered network...")
    
    filtered_edges = []
    for p1, p2, score in edges:
        p1_clean = p1.replace('9606.', '')
        p2_clean = p2.replace('9606.', '')
        symbol1 = ensembl_to_symbol.get(p1_clean)
        symbol2 = ensembl_to_symbol.get(p2_clean)
        
        if symbol1 and symbol2 and symbol1 in gene_set and symbol2 in gene_set:
            filtered_edges.append((symbol1, symbol2, score))
    
    # Get genes that have at least one edge
    network_genes = set()
    for g1, g2, _ in filtered_edges:
        network_genes.add(g1)
        network_genes.add(g2)
    
    print(f"  Edges: {len(filtered_edges):,}")
    print(f"  Genes with edges: {len(network_genes):,}")
    
    # Calculate average degree
    if len(network_genes) > 0:
        avg_degree = 2 * len(filtered_edges) / len(network_genes)
        print(f"  Average degree: {avg_degree:.2f}")
    
    return filtered_edges, network_genes


def create_patient_graphs(expression, metadata, gene_list, gene_to_idx, edge_index,
                          degree_centrality, gene_variance_values):
    """Create patient-specific graphs with 3D node features."""
    print(f"\n[5/6] Creating patient graphs...")
    
    data_list = []
    label_map = {'Control': 0, 'Sepsis': 1}
    
    # Normalize global features
    dc_normalized = (degree_centrality - degree_centrality.mean()) / (degree_centrality.std() + 1e-8)
    var_normalized = (gene_variance_values - gene_variance_values.mean()) / (gene_variance_values.std() + 1e-8)
    
    skipped = 0
    for _, row in metadata.iterrows():
        sample_id = row['SampleID']
        condition = row['Condition']
        
        if condition not in label_map:
            skipped += 1
            continue
        
        if sample_id in expression.columns:
            sample_expr = expression.loc[gene_list, sample_id].values
            
            # Feature 1: Z-score normalized expression (per sample)
            expr_normalized = (sample_expr - sample_expr.mean()) / (sample_expr.std() + 1e-8)
            
            # Stack 3D features
            x = np.stack([expr_normalized, dc_normalized, var_normalized], axis=1)
            x = torch.tensor(x, dtype=torch.float32)
            
            y = torch.tensor([label_map[condition]], dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, y=y)
            data.sample_id = sample_id
            data.condition = condition
            data_list.append(data)
    
    print(f"  Created {len(data_list)} patient graphs (skipped {skipped} unknown)")
    return data_list


def main():
    # Load expression data
    print("\nLoading expression data...")
    expression = pd.read_csv(os.path.join(DATA_DIR, "combined_expression.csv"), index_col=0)
    metadata = pd.read_csv(os.path.join(DATA_DIR, "combined_metadata.csv"))
    print(f"Expression: {expression.shape[0]:,} genes x {expression.shape[1]} samples")
    
    # Load STRING network
    string_path = os.path.join(RAW_DIR, "9606.protein.links.v12.0.txt.gz")
    raw_edges = load_string_network(string_path, STRING_THRESHOLD)
    
    # Map proteins to genes
    ensembl_to_symbol = map_proteins_to_genes(raw_edges)
    
    # Select top variance genes
    top_genes, full_variance = select_top_variance_genes(expression, NUM_GENES)
    
    # Build filtered network
    filtered_edges, network_genes = build_filtered_network(raw_edges, ensembl_to_symbol, top_genes)
    
    # Final gene set: intersection of top variance and network genes
    final_genes = top_genes & network_genes
    print(f"\n  Final gene set (variance ∩ network): {len(final_genes)}")
    
    # Create gene mapping
    gene_list = sorted(list(final_genes))
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_list)}
    
    # Create edge index
    edges = []
    for g1, g2, _ in filtered_edges:
        if g1 in gene_to_idx and g2 in gene_to_idx:
            src, tgt = gene_to_idx[g1], gene_to_idx[g2]
            edges.append([src, tgt])
            edges.append([tgt, src])  # Undirected
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"  Edge index shape: {edge_index.shape}")
    
    # Compute degree centrality
    G = nx.Graph()
    G.add_nodes_from(range(len(gene_list)))
    G.add_edges_from(edge_index.t().numpy())
    degree_cent = nx.degree_centrality(G)
    degree_centrality = np.array([degree_cent.get(i, 0) for i in range(len(gene_list))])
    
    # Get gene variance values
    gene_variance_values = np.array([full_variance[g] for g in gene_list])
    
    # Create patient graphs
    data_list = create_patient_graphs(
        expression, metadata, gene_list, gene_to_idx, edge_index,
        degree_centrality, gene_variance_values
    )
    
    # Save
    print(f"\n[6/6] Saving...")
    output = {
        'data_list': data_list,
        'gene_list': gene_list,
        'gene_to_idx': gene_to_idx,
        'edge_index': edge_index,
        'degree_centrality': degree_centrality,
        'gene_variance': gene_variance_values,
        'config': {
            'num_genes': NUM_GENES,
            'string_threshold': STRING_THRESHOLD,
            'final_nodes': len(gene_list),
            'final_edges': edge_index.shape[1] // 2
        }
    }
    
    with open(os.path.join(OUT_DIR, "patient_graphs_optimized.pkl"), 'wb') as f:
        pickle.dump(output, f)
    
    # Verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    sample = data_list[0]
    print(f"✓ Total graphs: {len(data_list)}")
    print(f"✓ Nodes per graph: {sample.x.shape[0]}")
    print(f"✓ Node features: {sample.x.shape[1]}")
    print(f"✓ Edges: {edge_index.shape[1] // 2}")
    
    labels = [d.y.item() for d in data_list]
    from collections import Counter
    print(f"✓ Labels: {dict(Counter(labels))} (0=Control, 1=Sepsis)")
    
    ratio = len(data_list) / sample.x.shape[0]
    print(f"✓ Sample:Feature ratio: 1:{1/ratio:.1f}")
    
    print(f"\nSaved to: {OUT_DIR}/patient_graphs_optimized.pkl")


if __name__ == "__main__":
    main()
