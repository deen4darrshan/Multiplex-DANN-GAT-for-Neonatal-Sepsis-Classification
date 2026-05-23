# Feature Engineering

Variance based gene selection
- Per fold feature selection uses median absolute deviation (MAD) to select Top K genes.
- MAD is defined as median(|x - median(x)|).

Node features
Each gene node is represented by a 4D feature vector.
- Expression: standardized per sample, z = (x - mean) / std.
- MAD rank: normalized rank position of each gene based on MAD.
- Degree: normalized degree from the STRING derived graph.
- Clustering coefficient: local clustering per node.

Reasoning
- Expression captures condition specific activity.
- MAD rank emphasizes informative genes.
- Degree and clustering inject prior network structure into the learning signal.

Key scripts
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\03_build_graphs.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\08_run_real_external_eval.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\11_human_grouped5_eval.py`
