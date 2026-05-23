"""
Module C - Task C.2: Patient-Specific Graph Object Creation

Creates PyTorch Geometric Data objects for each patient.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
import os
import pickle

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

def create_patient_graphs(expression, metadata, gene_list, gene_to_idx, edge_index):
    """Create Data objects for each patient."""
    data_list = []
    
    # Label mapping
    label_map = {'Control': 0, 'Sepsis': 1, 'Unknown': 2}
    
    for i, row in metadata.iterrows():
        sample_id = row['SampleID']
        condition = row['Condition']
        
        # Get expression for this sample, only for genes in our graph
        if sample_id in expression.columns:
            sample_expr = expression.loc[gene_list, sample_id].values
            
            # Convert to tensor (normalize per sample)
            x = torch.tensor(sample_expr, dtype=torch.float32).unsqueeze(1)  # (num_nodes, 1)
            
            # Normalize (z-score)
            x = (x - x.mean()) / (x.std() + 1e-8)
            
            # Create label
            y = torch.tensor([label_map.get(condition, 2)], dtype=torch.long)
            
            # Create Data object
            data = Data(x=x, edge_index=edge_index, y=y)
            data.sample_id = sample_id
            data.condition = condition
            
            data_list.append(data)
    
    print(f"Created {len(data_list)} patient graphs")
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
    
    # 4. Create patient graphs
    data_list = create_patient_graphs(expression, metadata, gene_list, gene_to_idx, edge_index)
    
    # 5. Save
    print("\nSaving patient graphs...")
    
    # Save as pickle
    with open(os.path.join(OUT_DIR, "patient_graphs.pkl"), 'wb') as f:
        pickle.dump({
            'data_list': data_list,
            'gene_list': gene_list,
            'gene_to_idx': gene_to_idx,
            'edge_index': edge_index
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
        
        # Label distribution
        labels = [d.y.item() for d in data_list]
        from collections import Counter
        label_dist = Counter(labels)
        print(f"Label distribution: {dict(label_dist)}")
    
    if len(data_list) >= 300:
        print("✓ PASS: Created graphs for all patients")
    else:
        print(f"⚠ WARNING: Only {len(data_list)} graphs created")
    
    if len(data_list) > 0 and data_list[0].x.shape[0] >= 2000:
        print("✓ PASS: Node count >= 2,000")
    else:
        print(f"⚠ WARNING: Node count below 2,000")

if __name__ == "__main__":
    main()
