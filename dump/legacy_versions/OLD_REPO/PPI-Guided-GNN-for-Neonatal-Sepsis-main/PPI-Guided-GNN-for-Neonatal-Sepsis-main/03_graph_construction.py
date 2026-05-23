"""
Module C - Task C.1: Interaction Network Filtering

Filters STRING network for high-confidence interactions
and intersects with expressed genes.
IMPROVED: Uses adaptive thresholding to ensure graph connectivity.
"""

import pandas as pd
import gzip
import os
import networkx as nx

# Paths
DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

def load_string_network(filepath, score_threshold=400):
    """Load and filter STRING network."""
    print(f"Loading STRING network from {filepath}...")
    
    edges = []
    with gzip.open(filepath, 'rt') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                protein1 = parts[0]
                protein2 = parts[1]
                score = int(parts[2])
                
                if score >= score_threshold:
                    edges.append((protein1, protein2, score))
    
    print(f"Loaded {len(edges)} edges with score >= {score_threshold}")
    return edges

def map_proteins_to_genes(edges, gene_list):
    """Map STRING protein IDs to gene symbols using mygene."""
    import mygene
    
    # Get unique protein IDs
    proteins = set()
    for p1, p2, _ in edges:
        proteins.add(p1.replace('9606.', ''))
        proteins.add(p2.replace('9606.', ''))
    
    print(f"Unique proteins: {len(proteins)}")
    
    # Query mygene for mapping
    mg = mygene.MyGeneInfo()
    
    protein_list = list(proteins)
    print(f"Querying mygene for {len(protein_list)} proteins...")
    
    # Query in batches if needed, but mygene handles it
    results = mg.querymany(protein_list, scopes='ensembl.protein', 
                          fields='symbol', species='human', returnall=True)
    
    # Build mapping
    ensembl_to_symbol = {}
    for item in results['out']:
        if 'symbol' in item and 'query' in item:
            ensembl_to_symbol[item['query']] = item['symbol']
    
    print(f"Mapped {len(ensembl_to_symbol)} proteins to symbols")
    
    return ensembl_to_symbol

def main():
    # 1. Load STRING network (Low threshold to capture broader potential interactions)
    string_path = os.path.join(DATA_DIR, "9606.protein.links.v12.0.txt.gz")
    # Load with lowest acceptable threshold (0.4) initially
    all_edges_raw = load_string_network(string_path, score_threshold=400)
    
    # 2. Load gene list from combined expression data
    expression_data = pd.read_csv(os.path.join(PROCESSED_DIR, "combined_expression.csv"), index_col=0)
    gene_list = set(expression_data.index.tolist())
    print(f"\nExpressed genes from training data: {len(gene_list)}")
    
    # 3. Map proteins to gene symbols (DO THIS ONCE)
    print("\nMapping proteins to gene symbols...")
    ensembl_to_symbol = map_proteins_to_genes(all_edges_raw, gene_list)
    
    # 4. Filter edges to expressed genes (Master List)
    print("\nFiltering edges to expressed genes...")
    master_edges = []
    for p1, p2, score in all_edges_raw:
        p1_clean = p1.replace('9606.', '')
        p2_clean = p2.replace('9606.', '')
        symbol1 = ensembl_to_symbol.get(p1_clean)
        symbol2 = ensembl_to_symbol.get(p2_clean)
        
        if symbol1 and symbol2 and symbol1 in gene_list and symbol2 in gene_list:
            master_edges.append((symbol1, symbol2, score))
            
    print(f"Total valid edges (score >= 400): {len(master_edges)}")

    # 5. Adaptive Threshold Selection
    print("\n=== Adaptive Threshold Selection ===")
    selected_threshold = 400
    final_edges = []
    
    # Try thresholds from strict to lenient
    for thresh in [900, 700, 500, 400]:
        current_edges = [e for e in master_edges if e[2] >= thresh]
        
        # Calculate connectivity metrics
        start_nodes = set()
        for u, v, w in current_edges:
            start_nodes.add(u)
            start_nodes.add(v)
            
        if len(start_nodes) > 0:
            avg_deg = 2 * len(current_edges) / len(start_nodes)
        else:
            avg_deg = 0
            
        print(f"Threshold > {thresh/1000:.1f}: {len(current_edges)} edges, {len(start_nodes)} nodes, Avg Deg: {avg_deg:.2f}")
        
        # Criterion: Average Degree >= 5 (or if we reach fallback)
        if avg_deg >= 5.0:
            print(f"✓ Selected optimal threshold: {thresh} (Avg Deg={avg_deg:.2f} >= 5.0)")
            selected_threshold = thresh
            final_edges = current_edges
            break
        
        if thresh == 400:
             print("⚠ Reached minimum threshold (0.4). Using this.")
             selected_threshold = 400
             final_edges = current_edges

    # 6. Build Final Graph
    G = nx.Graph()
    for g1, g2, score in final_edges:
        G.add_edge(g1, g2, weight=score)
    
    print(f"\nFinal Graph Statistics (Threshold={selected_threshold}):")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    # Largest connected component
    if G.number_of_nodes() > 0:
        lcc = max(nx.connected_components(G), key=len)
        coverage = len(lcc) / len(gene_list) * 100
        print(f"  Largest Connected Component: {len(lcc)} nodes ({coverage:.1f}% coverage)")
    
    # 7. Save edge list
    edge_df = pd.DataFrame(final_edges, columns=['source', 'target', 'score'])
    edge_df.to_csv(os.path.join(OUT_DIR, "ppi_network.csv"), index=False)
    
    with open(os.path.join(OUT_DIR, "ppi_network.edgelist"), 'w') as f:
        for g1, g2, score in final_edges:
            f.write(f"{g1}\t{g2}\n")
    
    print(f"\nSaved optimized PPI network to ppi_network.csv and ppi_network.edgelist")

if __name__ == "__main__":
    main()
