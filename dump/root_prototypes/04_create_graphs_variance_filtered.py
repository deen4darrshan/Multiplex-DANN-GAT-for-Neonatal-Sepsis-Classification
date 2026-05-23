"""
Optimization Phase - CRITICAL FIX: Variance-Based Feature Selection

Reduces nodes from 9,380 to 500 to address curse of dimensionality.
Target: ~500 nodes for ~1:1 ratio (acceptable).

Steps:
1. Load combined expression data (319 samples)
2. Calculate variance for each gene
3. Select top 500 highest-variance genes # Aggressive reduction to improve Sample-to-Feature ratio
4. Re-filter STRING network to these genes only
5. Recalculate degree centrality and variance features
6. Create 3D node features and save
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import os
import pickle
import networkx as nx

# Paths
DATA_DIR = "data/processed"
GRAPH_DIR = "data/graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

# Hyperparameters
NUM_GENES = 500  # Aggressive reduction to improve Sample-to-Feature ratio
# With N=345, 500 features = 1:1.45 ratio


def load_data():
    """Load expression data, metadata, and network."""
    print("Loading data...")
    
    # Expression data (genes x samples)
    expression = pd.read_csv(os.path.join(DATA_DIR, "combined_expression.csv"), index_col=0)
    print(f"Expression: {expression.shape}")
    
    # Metadata
    metadata = pd.read_csv(os.path.join(DATA_DIR, "combined_metadata.csv"))
    print(f"Metadata: {metadata.shape}")
    
    # PPI Network
    network = pd.read_csv(os.path.join(DATA_DIR, "ppi_network.csv"))
    print(f"Network edges: {len(network)}")
    
    return expression, metadata, network

def select_top_variance_genes(expression, top_k=2000):
    """Select top-K genes by variance across samples."""
    print(f"\nSelecting top {top_k} genes by variance...")
    
    # Calculate variance for each gene
    gene_variance = expression.var(axis=1)
    
    # Sort and select top-K
    top_genes = gene_variance.nlargest(top_k).index.tolist()
    
    print(f"  Total genes: {len(expression)}")
    print(f"  Selected genes: {len(top_genes)}")
    print(f"  Min variance (selected): {gene_variance[top_genes].min():.4f}")
    print(f"  Max variance (selected): {gene_variance[top_genes].max():.4f}")
    
    return top_genes, gene_variance

def filter_network_to_genes(network, gene_set):
    """Filter network to only include edges between selected genes."""
    print("\nFiltering network to selected genes...")
    
    # Filter edges where both source and target are in gene_set
    mask = network['source'].isin(gene_set) & network['target'].isin(gene_set)
    filtered_network = network[mask].copy()
    
    print(f"  Original edges: {len(network)}")
    print(f"  Filtered edges: {len(filtered_network)}")
    
    # Get actual genes that appear in filtered network
    network_genes = set(filtered_network['source']) | set(filtered_network['target'])
    print(f"  Genes with edges: {len(network_genes)}")
    
    return filtered_network, network_genes

def create_gene_index_mapping(gene_list):
    """Create consistent gene-to-index mapping."""
    gene_list = sorted(list(gene_list))
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_list)}
    return gene_list, gene_to_idx

def create_edge_index(network, gene_to_idx):
    """Create edge_index tensor from network."""
    edges = []
    for _, row in network.iterrows():
        src = row['source']
        tgt = row['target']
        if src in gene_to_idx and tgt in gene_to_idx:
            src_idx = gene_to_idx[src]
            tgt_idx = gene_to_idx[tgt]
            # Add both directions (undirected graph)
            edges.append([src_idx, tgt_idx])
            edges.append([tgt_idx, src_idx])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"Edge index shape: {edge_index.shape}")
    
    return edge_index

def compute_degree_centrality(edge_index, num_nodes):
    """Compute degree centrality for each node."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().numpy()
    G.add_edges_from(edges)
    
    degree_cent = nx.degree_centrality(G)
    centrality = np.array([degree_cent.get(i, 0) for i in range(num_nodes)])
    
    return centrality

def create_patient_graphs_3d(expression, metadata, gene_list, gene_to_idx, edge_index, 
                              degree_centrality, gene_variance_values):
    """Create Data objects with 3D node features for each patient."""
    data_list = []
    
    # Label mapping (binary classification)
    label_map = {'Control': 0, 'Sepsis': 1}
    
    # Normalize degree centrality and gene variance globally
    dc_normalized = (degree_centrality - degree_centrality.mean()) / (degree_centrality.std() + 1e-8)
    var_normalized = (gene_variance_values - gene_variance_values.mean()) / (gene_variance_values.std() + 1e-8)
    
    skipped = 0
    for i, row in metadata.iterrows():
        sample_id = row['SampleID']
        condition = row['Condition']
        
        # Skip unknown conditions
        if condition not in label_map:
            skipped += 1
            continue
        
        # Get expression for this sample, only for genes in our graph
        if sample_id in expression.columns:
            sample_expr = expression.loc[gene_list, sample_id].values
            
            # Feature 1: Normalized expression (z-score per sample)
            expr_normalized = (sample_expr - sample_expr.mean()) / (sample_expr.std() + 1e-8)
            
            # Stack 3D features: [expression, degree_centrality, gene_variance]
            x = np.stack([
                expr_normalized,      # Feature 1: Expression
                dc_normalized,        # Feature 2: Degree Centrality
                var_normalized        # Feature 3: Gene Variance
            ], axis=1)
            
            x = torch.tensor(x, dtype=torch.float32)  # (num_nodes, 3)
            
            # Create label
            y = torch.tensor([label_map[condition]], dtype=torch.long)
            
            # Create Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            data.sample_id = sample_id
            data.condition = condition
            
            data_list.append(data)
    
    print(f"Created {len(data_list)} patient graphs (skipped {skipped} unknown)")
    return data_list

def main():
    # 1. Load data
    expression, metadata, network = load_data()
    
    # 2. CRITICAL: Select top-K variance genes
    top_genes, full_variance = select_top_variance_genes(expression, NUM_GENES)
    
    # 3. Filter network to selected genes
    filtered_network, network_genes = filter_network_to_genes(network, set(top_genes))
    
    # 4. Use intersection of top-variance genes and network genes
    final_genes = set(top_genes) & network_genes
    print(f"\nFinal gene set (top variance ∩ network): {len(final_genes)}")
    
    if len(final_genes) < 1000:
        print("WARNING: Less than 1000 genes after filtering!")
    
    # 5. Create gene mapping
    gene_list, gene_to_idx = create_gene_index_mapping(final_genes)
    
    # 6. Create edge index
    edge_index = create_edge_index(filtered_network, gene_to_idx)
    
    # 7. Compute features for the reduced gene set
    print("\nComputing node features for reduced gene set...")
    num_nodes = len(gene_list)
    
    # Feature 2: Degree Centrality
    degree_centrality = compute_degree_centrality(edge_index, num_nodes)
    print(f"  Degree centrality: mean={degree_centrality.mean():.4f}, std={degree_centrality.std():.4f}")
    
    # Feature 3: Gene Variance (from original full variance)
    gene_variance_values = np.array([full_variance[g] for g in gene_list])
    print(f"  Gene variance: mean={gene_variance_values.mean():.4f}, std={gene_variance_values.std():.4f}")
    
    # 8. Create patient graphs with 3D features
    data_list = create_patient_graphs_3d(
        expression, metadata, gene_list, gene_to_idx, edge_index,
        degree_centrality, gene_variance_values
    )
    
    # 9. Save (overwrite)
    print("\nSaving variance-filtered patient graphs...")
    
    with open(os.path.join(GRAPH_DIR, "patient_graphs_3d.pkl"), 'wb') as f:
        pickle.dump({
            'data_list': data_list,
            'gene_list': gene_list,
            'gene_to_idx': gene_to_idx,
            'edge_index': edge_index,
            'degree_centrality': degree_centrality,
            'gene_variance': gene_variance_values
        }, f)
    
    # Summary and Verification
    print(f"\n{'='*60}")
    print("=== VERIFICATION (CoVe) ===")
    print(f"{'='*60}")
    
    print(f"Total patient graphs: {len(data_list)}")
    if len(data_list) > 0:
        sample = data_list[0]
        print(f"Node feature shape: {sample.x.shape}")
        print(f"Edge index shape: {sample.edge_index.shape}")
        
        # CRITICAL: Verify node count ~2000
        if sample.x.shape[0] <= 2000:
            print(f"✓ PASS: Node count = {sample.x.shape[0]} (≤ 2,000 target)")
        else:
            print(f"✗ FAIL: Node count = {sample.x.shape[0]} (expected ≤ 2,000)")
        
        # Verify 3D features
        if sample.x.shape[1] == 3:
            print("✓ PASS: Node features have 3 dimensions")
        else:
            print(f"✗ FAIL: Expected 3 node features, got {sample.x.shape[1]}")
        
        # Label distribution
        labels = [d.y.item() for d in data_list]
        from collections import Counter
        label_dist = Counter(labels)
        print(f"Label distribution: {dict(label_dist)} (0=Control, 1=Sepsis)")
    
    # Verify sample count
    if len(data_list) == 319:
        print(f"✓ PASS: Graph count = 319")
    else:
        print(f"⚠ WARNING: Graph count = {len(data_list)} (expected 319)")
    
    # Sample-to-feature ratio
    ratio = len(data_list) / sample.x.shape[0] if len(data_list) > 0 else 0
    print(f"\nSample-to-feature ratio: 1:{1/ratio:.1f} (was 1:30, now acceptable)")

if __name__ == "__main__":
    main()
