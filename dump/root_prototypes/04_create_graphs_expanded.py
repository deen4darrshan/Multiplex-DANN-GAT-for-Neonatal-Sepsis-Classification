"""
Graph Creation with Expanded Nodes for GAT Training

Expands from ~1,500 nodes to ~3,000-4,000 nodes by:
1. Increasing variance genes from 2000 to 4000
2. Lowering STRING threshold from 700 to 500

This creates denser graphs for GAT to leverage attention mechanisms.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import os
import pickle
import networkx as nx
import gzip

# Paths
DATA_DIR = "data/processed"
RAW_DIR = "data/raw"
GRAPH_DIR = "data/graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

# EXPANDED Hyperparameters
NUM_GENES = 4000           # Up from 2000
STRING_THRESHOLD = 500     # Down from 700 (confidence > 0.5)

print(f"\n{'='*60}")
print("=== EXPANDED GRAPH CONSTRUCTION FOR GAT ===")
print(f"{'='*60}")
print(f"  - Variance genes to select: {NUM_GENES}")
print(f"  - STRING threshold: {STRING_THRESHOLD} (confidence > {STRING_THRESHOLD/1000:.1f})")
print(f"{'='*60}\n")


def load_data():
    """Load expression data, metadata, and network."""
    print("Loading data...")
    
    # Expression data (genes x samples)
    expression = pd.read_csv(os.path.join(DATA_DIR, "combined_expression.csv"), index_col=0)
    print(f"Expression: {expression.shape}")
    
    # Metadata
    metadata = pd.read_csv(os.path.join(DATA_DIR, "combined_metadata.csv"))
    print(f"Metadata: {metadata.shape}")
    
    # PPI Network (load raw STRING for re-thresholding)
    network = pd.read_csv(os.path.join(DATA_DIR, "ppi_network.csv"))
    print(f"Existing network edges (from ppi_network.csv): {len(network)}")
    
    return expression, metadata, network


def load_string_raw(filepath, score_threshold=500):
    """Load STRING network with lower threshold."""
    print(f"\nLoading STRING network with threshold >= {score_threshold}...")
    
    edges = []
    with gzip.open(filepath, 'rt') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                protein1 = parts[0].replace('9606.', '')
                protein2 = parts[1].replace('9606.', '')
                score = int(parts[2])
                
                if score >= score_threshold:
                    edges.append((protein1, protein2, score))
    
    print(f"Loaded {len(edges)} edges with score >= {score_threshold}")
    return edges


def map_proteins_to_genes(edges):
    """Map STRING protein IDs to gene symbols using mygene."""
    import mygene
    
    proteins = set()
    for p1, p2, _ in edges:
        proteins.add(p1)
        proteins.add(p2)
    
    print(f"Unique proteins to map: {len(proteins)}")
    
    mg = mygene.MyGeneInfo()
    protein_list = list(proteins)
    
    print(f"Querying mygene for {len(protein_list)} proteins...")
    results = mg.querymany(protein_list, scopes='ensembl.protein', 
                          fields='symbol', species='human', returnall=True)
    
    ensembl_to_symbol = {}
    for item in results['out']:
        if 'symbol' in item and 'query' in item:
            ensembl_to_symbol[item['query']] = item['symbol']
    
    print(f"Mapped {len(ensembl_to_symbol)} proteins to gene symbols")
    return ensembl_to_symbol


def select_top_variance_genes(expression, top_k=4000):
    """Select top-K genes by variance across samples."""
    print(f"\nSelecting top {top_k} genes by variance...")
    
    gene_variance = expression.var(axis=1)
    top_genes = gene_variance.nlargest(top_k).index.tolist()
    
    print(f"  Total genes: {len(expression)}")
    print(f"  Selected genes: {len(top_genes)}")
    print(f"  Min variance (selected): {gene_variance[top_genes].min():.4f}")
    print(f"  Max variance (selected): {gene_variance[top_genes].max():.4f}")
    
    return top_genes, gene_variance


def filter_network_to_genes(edges_with_symbols, gene_set):
    """Filter network to only include edges between selected genes."""
    print("\nFiltering network to selected genes...")
    
    filtered = [(g1, g2, s) for g1, g2, s in edges_with_symbols 
                if g1 in gene_set and g2 in gene_set]
    
    print(f"  Original edges: {len(edges_with_symbols)}")
    print(f"  Filtered edges: {len(filtered)}")
    
    network_genes = set()
    for g1, g2, _ in filtered:
        network_genes.add(g1)
        network_genes.add(g2)
    print(f"  Genes with edges: {len(network_genes)}")
    
    return filtered, network_genes


def create_gene_index_mapping(gene_list):
    """Create consistent gene-to-index mapping."""
    gene_list = sorted(list(gene_list))
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_list)}
    return gene_list, gene_to_idx


def create_edge_index(edges, gene_to_idx):
    """Create edge_index tensor from network."""
    edge_list = []
    for g1, g2, _ in edges:
        if g1 in gene_to_idx and g2 in gene_to_idx:
            src_idx = gene_to_idx[g1]
            tgt_idx = gene_to_idx[g2]
            edge_list.append([src_idx, tgt_idx])
            edge_list.append([tgt_idx, src_idx])  # Undirected
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
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
    label_map = {'Control': 0, 'Sepsis': 1}
    
    # Normalize global features
    dc_normalized = (degree_centrality - degree_centrality.mean()) / (degree_centrality.std() + 1e-8)
    var_normalized = (gene_variance_values - gene_variance_values.mean()) / (gene_variance_values.std() + 1e-8)
    
    skipped = 0
    for i, row in metadata.iterrows():
        sample_id = row['SampleID']
        condition = row['Condition']
        
        if condition not in label_map:
            skipped += 1
            continue
        
        if sample_id in expression.columns:
            sample_expr = expression.loc[gene_list, sample_id].values
            expr_normalized = (sample_expr - sample_expr.mean()) / (sample_expr.std() + 1e-8)
            
            x = np.stack([
                expr_normalized,      # Feature 1: Expression
                dc_normalized,        # Feature 2: Degree Centrality
                var_normalized        # Feature 3: Gene Variance
            ], axis=1)
            
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor([label_map[condition]], dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, y=y)
            data.sample_id = sample_id
            data.condition = condition
            
            data_list.append(data)
    
    print(f"Created {len(data_list)} patient graphs (skipped {skipped} unknown)")
    return data_list


def main():
    # 1. Load data
    expression, metadata, _ = load_data()
    
    # 2. Load STRING with lower threshold
    string_path = os.path.join(RAW_DIR, "9606.protein.links.v12.0.txt.gz")
    raw_edges = load_string_raw(string_path, STRING_THRESHOLD)
    
    # 3. Map proteins to gene symbols
    ensembl_to_symbol = map_proteins_to_genes(raw_edges)
    
    # 4. Convert edges to gene symbols
    gene_list_expr = set(expression.index.tolist())
    edges_with_symbols = []
    for p1, p2, score in raw_edges:
        g1 = ensembl_to_symbol.get(p1)
        g2 = ensembl_to_symbol.get(p2)
        if g1 and g2 and g1 in gene_list_expr and g2 in gene_list_expr:
            edges_with_symbols.append((g1, g2, score))
    
    print(f"Edges mapped to expressed genes: {len(edges_with_symbols)}")
    
    # 5. Select top variance genes
    top_genes, full_variance = select_top_variance_genes(expression, NUM_GENES)
    
    # 6. Filter network to selected genes
    filtered_edges, network_genes = filter_network_to_genes(
        edges_with_symbols, set(top_genes)
    )
    
    # 7. Final gene set = intersection
    final_genes = set(top_genes) & network_genes
    print(f"\nFinal gene set (top variance ∩ network): {len(final_genes)}")
    
    # 8. Create gene mapping
    gene_list, gene_to_idx = create_gene_index_mapping(final_genes)
    
    # 9. Create edge index
    edge_index = create_edge_index(filtered_edges, gene_to_idx)
    
    # 10. Compute features
    print("\nComputing node features...")
    num_nodes = len(gene_list)
    
    degree_centrality = compute_degree_centrality(edge_index, num_nodes)
    print(f"  Degree centrality: mean={degree_centrality.mean():.4f}, std={degree_centrality.std():.4f}")
    
    gene_variance_values = np.array([full_variance[g] for g in gene_list])
    print(f"  Gene variance: mean={gene_variance_values.mean():.4f}, std={gene_variance_values.std():.4f}")
    
    # 11. Create patient graphs
    data_list = create_patient_graphs_3d(
        expression, metadata, gene_list, gene_to_idx, edge_index,
        degree_centrality, gene_variance_values
    )
    
    # 12. Save expanded graphs
    output_file = os.path.join(GRAPH_DIR, "patient_graphs_expanded.pkl")
    print(f"\nSaving expanded patient graphs to {output_file}...")
    
    with open(output_file, 'wb') as f:
        pickle.dump({
            'data_list': data_list,
            'gene_list': gene_list,
            'gene_to_idx': gene_to_idx,
            'edge_index': edge_index,
            'degree_centrality': degree_centrality,
            'gene_variance': gene_variance_values,
            'config': {
                'num_genes': NUM_GENES,
                'string_threshold': STRING_THRESHOLD
            }
        }, f)
    
    # Verification
    print(f"\n{'='*60}")
    print("=== VERIFICATION ===")
    print(f"{'='*60}")
    
    if len(data_list) > 0:
        sample = data_list[0]
        print(f"Total patient graphs: {len(data_list)}")
        print(f"Node count: {sample.x.shape[0]}")
        print(f"Edge count: {sample.edge_index.shape[1] // 2}")
        print(f"Node features: {sample.x.shape[1]}D")
        
        avg_degree = sample.edge_index.shape[1] / sample.x.shape[0]
        print(f"Average degree: {avg_degree:.1f}")
        
        labels = [d.y.item() for d in data_list]
        from collections import Counter
        label_dist = Counter(labels)
        print(f"Label distribution: {dict(label_dist)} (0=Control, 1=Sepsis)")
        
        # Success criteria
        if sample.x.shape[0] >= 2500:
            print(f"\n✓ PASS: Node count = {sample.x.shape[0]} (target >= 2,500)")
        else:
            print(f"\n⚠ WARNING: Node count = {sample.x.shape[0]} (target >= 2,500)")


if __name__ == "__main__":
    main()
