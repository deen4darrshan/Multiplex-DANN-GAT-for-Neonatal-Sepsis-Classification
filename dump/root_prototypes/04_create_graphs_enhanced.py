"""
Optimization Phase - Task O.2/O.3: Enhanced Graph Construction

Creates PyTorch Geometric Data objects with 3D node features:
1. Normalized Gene Expression (Z-score)
2. Degree Centrality (from PPI graph)
3. Gene Variance (across training cohort)
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
OUT_DIR = "data/graphs"
os.makedirs(OUT_DIR, exist_ok=True)

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

def create_gene_index_mapping(expression, network):
    """Create consistent gene-to-index mapping."""
    # Get genes that appear in both expression and network
    expr_genes = set(expression.index)
    network_genes = set(network['source']) | set(network['target'])
    common_genes = expr_genes & network_genes
    
    print(f"Genes in expression: {len(expr_genes)}")
    print(f"Genes in network: {len(network_genes)}")
    print(f"Common genes: {len(common_genes)}")
    
    # Create sorted gene list for consistent indexing
    gene_list = sorted(list(common_genes))
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
    # Build NetworkX graph for degree computation
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().numpy()
    G.add_edges_from(edges)
    
    # Compute degree centrality
    degree_cent = nx.degree_centrality(G)
    
    # Convert to numpy array
    centrality = np.array([degree_cent.get(i, 0) for i in range(num_nodes)])
    
    return centrality

def compute_gene_variance(expression, gene_list):
    """Compute variance for each gene across all samples."""
    variance = expression.loc[gene_list].var(axis=1).values
    return variance

def create_patient_graphs_3d(expression, metadata, gene_list, gene_to_idx, edge_index, 
                              degree_centrality, gene_variance):
    """Create Data objects with 3D node features for each patient."""
    data_list = []
    
    # Label mapping (binary classification)
    label_map = {'Control': 0, 'Sepsis': 1}
    
    # Normalize degree centrality and gene variance globally
    dc_normalized = (degree_centrality - degree_centrality.mean()) / (degree_centrality.std() + 1e-8)
    var_normalized = (gene_variance - gene_variance.mean()) / (gene_variance.std() + 1e-8)
    
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
    
    # 2. Create gene mapping
    gene_list, gene_to_idx = create_gene_index_mapping(expression, network)
    
    if len(gene_list) == 0:
        print("ERROR: No common genes between expression and network!")
        return
    
    # 3. Create edge index
    edge_index = create_edge_index(network, gene_to_idx)
    
    # 4. Compute additional features
    print("\nComputing node features...")
    num_nodes = len(gene_list)
    
    # Feature 2: Degree Centrality
    degree_centrality = compute_degree_centrality(edge_index, num_nodes)
    print(f"  Degree centrality: mean={degree_centrality.mean():.4f}, std={degree_centrality.std():.4f}")
    
    # Feature 3: Gene Variance
    gene_variance = compute_gene_variance(expression, gene_list)
    print(f"  Gene variance: mean={gene_variance.mean():.4f}, std={gene_variance.std():.4f}")
    
    # 5. Create patient graphs with 3D features
    data_list = create_patient_graphs_3d(
        expression, metadata, gene_list, gene_to_idx, edge_index,
        degree_centrality, gene_variance
    )
    
    # 6. Save
    print("\nSaving enhanced patient graphs...")
    
    # Save as pickle
    with open(os.path.join(OUT_DIR, "patient_graphs_3d.pkl"), 'wb') as f:
        pickle.dump({
            'data_list': data_list,
            'gene_list': gene_list,
            'gene_to_idx': gene_to_idx,
            'edge_index': edge_index,
            'degree_centrality': degree_centrality,
            'gene_variance': gene_variance
        }, f)
    
    # Summary
    print(f"\n{'='*60}")
    print("=== VERIFICATION (CoVe) ===")
    print(f"{'='*60}")
    
    print(f"Total patient graphs: {len(data_list)}")
    if len(data_list) > 0:
        sample = data_list[0]
        print(f"Sample graph - x shape: {sample.x.shape}")
        print(f"Sample graph - edge_index shape: {sample.edge_index.shape}")
        
        # Verify 3D features
        if sample.x.shape[1] == 3:
            print("✓ PASS: Node features have 3 dimensions (expression, degree, variance)")
        else:
            print(f"✗ FAIL: Expected 3 node features, got {sample.x.shape[1]}")
        
        # Label distribution
        labels = [d.y.item() for d in data_list]
        from collections import Counter
        label_dist = Counter(labels)
        print(f"Label distribution: {dict(label_dist)} (0=Control, 1=Sepsis)")
    
    # Verify sample count
    expected_samples = 319
    if len(data_list) >= 300:
        print(f"✓ PASS: Created {len(data_list)} graphs (expected ~{expected_samples})")
    else:
        print(f"✗ FAIL: Only {len(data_list)} graphs created")
    
    if len(data_list) > 0 and data_list[0].x.shape[0] >= 2000:
        print(f"✓ PASS: Node count = {data_list[0].x.shape[0]} (>= 2,000)")
    else:
        print(f"⚠ WARNING: Node count below 2,000")

if __name__ == "__main__":
    main()
