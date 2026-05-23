import os
import gzip
import pickle
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
import torch
from torch_geometric.data import Data

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_DIR = os.path.join(ROOT, 'data', 'processed')
RAW_DIR = os.path.join(ROOT, 'data', 'raw')

EXPR_PATH = os.path.join(PROC_DIR, 'expression_combat.csv')
META_PATH = os.path.join(PROC_DIR, 'metadata_combat.csv')
STRING_FILE = os.path.join(RAW_DIR, '9606.protein.links.v12.0.txt.gz')

TOP_K = 1000
STRING_THRESH = 700


def variance_filter(expression, k=TOP_K):
    mad_scores = expression.apply(median_abs_deviation, axis=1)
    mad_scores = mad_scores.sort_values(ascending=False)
    top_genes = mad_scores.head(k).index.tolist()
    return top_genes, mad_scores


def load_string_network(threshold=STRING_THRESH):
    edges = []
    all_proteins = set()
    with gzip.open(STRING_FILE, 'rt') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            p1, p2, score = parts[0], parts[1], int(parts[2])
            if score >= threshold:
                p1 = p1.replace('9606.', '')
                p2 = p2.replace('9606.', '')
                edges.append((p1, p2))
                all_proteins.add(p1)
                all_proteins.add(p2)
    return edges, all_proteins


def map_ensp_to_genes(proteins):
    import mygene
    mg = mygene.MyGeneInfo()
    protein_list = list(proteins)
    ensp_to_gene = {}
    batch_size = 1000
    for i in range(0, len(protein_list), batch_size):
        batch = protein_list[i:i+batch_size]
        results = mg.querymany(batch, scopes='ensembl.protein', fields='symbol', species='human', returnall=False, verbose=False)
        for r in results:
            if 'symbol' in r:
                ensp_to_gene[r['query']] = str(r['symbol']).upper()
    return ensp_to_gene


def build_gene_edges(string_edges, ensp_to_gene, gene_list):
    gene_set = set(gene_list)
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    edges_src, edges_dst = [], []
    unique_edges = set()
    for p1, p2 in string_edges:
        g1 = ensp_to_gene.get(p1)
        g2 = ensp_to_gene.get(p2)
        if g1 and g2 and g1 in gene_set and g2 in gene_set and g1 != g2:
            edge_key = tuple(sorted([g1, g2]))
            if edge_key not in unique_edges:
                unique_edges.add(edge_key)
                i, j = gene_to_idx[g1], gene_to_idx[g2]
                edges_src.extend([i, j])
                edges_dst.extend([j, i])
    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    return edge_index, unique_edges


def compute_graph_features(edge_index, n_nodes):
    degree = torch.zeros(n_nodes, dtype=torch.float)
    for i in range(edge_index.shape[1]):
        degree[edge_index[0, i]] += 1

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
        triangles = 0
        neighbors_list = list(neighbors)
        for i_n in range(len(neighbors_list)):
            for j_n in range(i_n + 1, len(neighbors_list)):
                if neighbors_list[j_n] in adj.get(neighbors_list[i_n], set()):
                    triangles += 1
        clustering[node] = 2.0 * triangles / (k * (k - 1))

    return degree, clustering


def create_graphs(expression, metadata, gene_list, edge_index, mad_scores, degree, clustering):
    n_nodes = len(gene_list)
    gene_mads = np.array([mad_scores.get(g, 0.0) for g in gene_list], dtype=np.float32)
    mad_ranks = np.argsort(np.argsort(-gene_mads)).astype(np.float32) / max(n_nodes - 1, 1)

    deg_np = degree.numpy()
    deg_norm = deg_np / max(deg_np.max(), 1.0)
    clust_np = clustering.numpy()

    graphs = []
    for _, row in metadata.iterrows():
        sample_id = row['SampleID']
        label = int(row['Label'])
        expr_vals = expression.loc[gene_list, sample_id].values.astype(np.float32)
        mean = np.mean(expr_vals)
        std = np.std(expr_vals)
        if std > 1e-8:
            expr_vals = (expr_vals - mean) / std

        features = np.stack([expr_vals, mad_ranks, deg_norm, clust_np], axis=1)
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.sample_id = sample_id
        data.batch_label = row.get('Batch', 'Unknown')
        graphs.append(data)

    return graphs


def main():
    if not os.path.exists(EXPR_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError('Missing inputs. Run 02_combat_correction.py first.')

    expr = pd.read_csv(EXPR_PATH, index_col=0)
    meta = pd.read_csv(META_PATH)

    # Variance filter
    top_genes, mad_scores = variance_filter(expr, k=TOP_K)
    with open(os.path.join(PROC_DIR, 'top_genes.txt'), 'w') as f:
        for g in top_genes:
            f.write(g + '\n')

    # Load STRING
    print('Loading STRING network...')
    string_edges, proteins = load_string_network()
    print(f'STRING edges >= {STRING_THRESH}: {len(string_edges)}')

    # Map ENSP to genes
    ensp_to_gene = map_ensp_to_genes(proteins)

    # Build edges
    edge_index, unique_edges = build_gene_edges(string_edges, ensp_to_gene, top_genes)

    nodes_with_edges = set()
    for g1, g2 in unique_edges:
        nodes_with_edges.add(g1)
        nodes_with_edges.add(g2)

    connected_genes = [g for g in top_genes if g in nodes_with_edges]
    if len(connected_genes) < 100:
        print(f'[warn] Only {len(connected_genes)} connected genes; using Top-K without filtering')
        connected_genes = top_genes

    # Rebuild for final gene list
    gene_to_idx = {g: i for i, g in enumerate(connected_genes)}
    edges_src, edges_dst = [], []
    for g1, g2 in unique_edges:
        if g1 in gene_to_idx and g2 in gene_to_idx:
            i, j = gene_to_idx[g1], gene_to_idx[g2]
            edges_src.extend([i, j])
            edges_dst.extend([j, i])
    final_edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    n_nodes = len(connected_genes)
    n_edges = final_edge_index.shape[1] // 2
    avg_degree = final_edge_index.shape[1] / max(n_nodes, 1)

    print(f'Final graph: nodes={n_nodes} edges={n_edges} avg_degree={avg_degree:.1f}')

    # structural features
    degree, clustering = compute_graph_features(final_edge_index, n_nodes)

    # Save gene list
    with open(os.path.join(PROC_DIR, 'final_genes.txt'), 'w') as f:
        for g in connected_genes:
            f.write(g + '\n')

    # Build graphs
    graphs = create_graphs(expr, meta, connected_genes, final_edge_index, mad_scores.to_dict(), degree, clustering)

    # Save
    torch.save(graphs, os.path.join(PROC_DIR, 'graphs.pt'))
    with open(os.path.join(PROC_DIR, 'graph_metadata.pkl'), 'wb') as f:
        pickle.dump({
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'avg_degree': avg_degree,
            'top_k': TOP_K,
            'string_threshold': STRING_THRESH,
        }, f)

    print(f'Saved graphs: {len(graphs)}')


if __name__ == '__main__':
    main()
