# Graph Construction

Protein interaction network
- STRING v12 protein protein interactions are filtered at a confidence threshold of 700.
- Ensembl protein IDs are mapped to gene symbols using MyGene.

Gene subset
- Top K genes by MAD are selected for graph nodes.
- Edges are kept only when both endpoints map to the selected gene set.
- If edge connectivity is too low, the pipeline falls back to a simple chain graph to ensure graph connectivity for training.

Graph metadata
- Graphs store node features, edge indices, sample labels, and dataset identifiers.
- Metadata for nodes, edges, and thresholds is stored alongside graph files.

Key script
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\03_build_graphs.py`

Artifacts
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\data\processed\graphs.pt`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\data\processed\graph_metadata.pkl`
