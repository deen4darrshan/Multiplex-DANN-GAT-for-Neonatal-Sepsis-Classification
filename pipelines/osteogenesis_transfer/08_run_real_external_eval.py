import os
import json
import gzip
import pickle
import numpy as np
import pandas as pd

from scipy.stats import median_abs_deviation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool

from combat.pycombat import pycombat

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
PROC_DIR = os.path.join(ROOT, 'data', 'processed')
DATASET_DIR = os.path.join(PROC_DIR, 'datasets')
RESULTS_DIR = os.path.join(ROOT, 'results')
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

STRING_FILE = os.path.join(RAW_DIR, '9606.protein.links.v12.0.txt.gz')
ALT_STRING = os.path.abspath(os.path.join(ROOT, '..', 'ALZHEIMERS_STRATEGIC_PATHWAY', 'data', 'adni', 'raw', '9606.protein.links.v12.0.txt.gz'))
CACHE_ENSP = os.path.join(PROC_DIR, 'ensp_to_gene_cache.pkl')

TOP_K = 1000
STRING_THRESH = 700
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def ensure_string_file():
    if os.path.exists(STRING_FILE):
        return
    if os.path.exists(ALT_STRING):
        import shutil
        shutil.copy2(ALT_STRING, STRING_FILE)
        return
    raise FileNotFoundError('STRING file missing')


def load_multicohort_data():
    meta = pd.read_csv(os.path.join(PROC_DIR, 'multicohort_metadata.csv'))
    expr = {}
    for ds in sorted(meta['Dataset'].unique()):
        expr[ds] = pd.read_csv(os.path.join(DATASET_DIR, f'{ds}_expr.csv'), index_col=0)
    return expr, meta


def run_combat_train(train_expr, batch_labels, cond_labels):
    # train_expr: genes x samples
    try:
        corrected = pycombat(train_expr, batch=batch_labels, mod=[list(cond_labels)])
    except Exception:
        corrected = pycombat(train_expr, batch=batch_labels)
    return corrected


def load_string_edges(threshold=STRING_THRESH):
    edges = []
    proteins = set()
    with gzip.open(STRING_FILE, 'rt') as f:
        header = f.readline()
        for line in f:
            p = line.strip().split()
            if len(p) < 3:
                continue
            p1, p2, s = p[0], p[1], int(p[2])
            if s >= threshold:
                p1 = p1.replace('9606.', '')
                p2 = p2.replace('9606.', '')
                edges.append((p1, p2))
                proteins.add(p1)
                proteins.add(p2)
    return edges, proteins


def load_or_build_ensp_mapping(proteins):
    if os.path.exists(CACHE_ENSP):
        with open(CACHE_ENSP, 'rb') as f:
            return pickle.load(f)

    import mygene
    mg = mygene.MyGeneInfo()
    proteins = list(proteins)
    mapping = {}

    batch_size = 1000
    for i in range(0, len(proteins), batch_size):
        batch = proteins[i:i+batch_size]
        res = mg.querymany(batch, scopes='ensembl.protein', fields='symbol', species='human', returnall=False, verbose=False)
        for r in res:
            if 'symbol' in r:
                mapping[r['query']] = str(r['symbol']).upper()

    with open(CACHE_ENSP, 'wb') as f:
        pickle.dump(mapping, f)
    return mapping


def build_topology(top_genes, string_edges, ensp_to_gene):
    gene_set = set(top_genes)
    unique_edges = set()
    for p1, p2 in string_edges:
        g1 = ensp_to_gene.get(p1)
        g2 = ensp_to_gene.get(p2)
        if g1 and g2 and g1 != g2 and g1 in gene_set and g2 in gene_set:
            unique_edges.add(tuple(sorted((g1, g2))))

    connected = set()
    for g1, g2 in unique_edges:
        connected.add(g1)
        connected.add(g2)

    final_genes = [g for g in top_genes if g in connected]
    if len(final_genes) < 100:
        final_genes = top_genes[:min(len(top_genes), 400)]

    g2i = {g: i for i, g in enumerate(final_genes)}
    src = []
    dst = []
    for g1, g2 in unique_edges:
        if g1 in g2i and g2 in g2i:
            i = g2i[g1]
            j = g2i[g2]
            src.extend([i, j])
            dst.extend([j, i])

    if len(src) == 0:
        # fallback chain graph
        n = len(final_genes)
        for i in range(n - 1):
            src.extend([i, i + 1])
            dst.extend([i + 1, i])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return final_genes, edge_index


def compute_structural_features(edge_index, n_nodes):
    degree = torch.zeros(n_nodes, dtype=torch.float)
    for i in range(edge_index.shape[1]):
        degree[edge_index[0, i]] += 1

    adj = {}
    for i in range(edge_index.shape[1]):
        s = edge_index[0, i].item()
        d = edge_index[1, i].item()
        adj.setdefault(s, set()).add(d)

    clustering = torch.zeros(n_nodes, dtype=torch.float)
    for node in range(n_nodes):
        nbr = list(adj.get(node, set()))
        k = len(nbr)
        if k < 2:
            continue
        tri = 0
        for i in range(k):
            for j in range(i + 1, k):
                if nbr[j] in adj.get(nbr[i], set()):
                    tri += 1
        clustering[node] = 2.0 * tri / (k * (k - 1))

    return degree, clustering


def create_graphs(expr, meta, genes, edge_index, mad_scores, degree, clustering):
    n_nodes = len(genes)
    gene_mads = np.array([mad_scores.get(g, 0.0) for g in genes], dtype=np.float32)
    mad_rank = np.argsort(np.argsort(-gene_mads)).astype(np.float32) / max(n_nodes - 1, 1)

    deg = degree.numpy()
    deg_norm = deg / max(float(deg.max()), 1.0)
    clust = clustering.numpy()

    graphs = []
    for _, r in meta.iterrows():
        sid = r['SampleID']
        y = int(r['Label'])

        x_expr = expr.loc[genes, sid].values.astype(np.float32)
        mu = float(np.mean(x_expr))
        sd = float(np.std(x_expr))
        if sd > 1e-8:
            x_expr = (x_expr - mu) / sd

        feats = np.stack([x_expr, mad_rank, deg_norm, clust], axis=1)
        data = Data(
            x=torch.tensor(feats, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor([y], dtype=torch.long),
        )
        data.sample_id = sid
        data.batch_label = r['Dataset']
        graphs.append(data)
    return graphs


class OIGATv2(nn.Module):
    def __init__(self, n_feat=4, hidden=64, heads=4, dropout=0.5):
        super().__init__()
        self.drop = dropout
        self.conv1 = TransformerConv(n_feat, hidden, heads=heads, dropout=0.1)
        self.ln1 = nn.LayerNorm(hidden * heads)
        self.res = nn.Linear(n_feat, hidden * heads)

        self.conv2 = TransformerConv(hidden * heads, hidden, heads=heads, dropout=0.1)
        self.ln2 = nn.LayerNorm(hidden * heads)

        self.conv3 = TransformerConv(hidden * heads, hidden, heads=1, concat=False, dropout=0.1)
        self.ln3 = nn.LayerNorm(hidden)

        self.fc1 = nn.Linear(hidden * 2, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x, edge_index, batch, edge_drop=0.05, noise=0.02):
        if self.training and edge_drop > 0:
            keep = torch.rand(edge_index.size(1), device=edge_index.device) > edge_drop
            edge_index = edge_index[:, keep]
        if self.training and noise > 0:
            x = x + torch.randn_like(x) * noise

        r = self.res(x)
        x = F.leaky_relu(self.ln1(self.conv1(x, edge_index)) + r)
        r = x
        x = F.leaky_relu(self.ln2(self.conv2(x, edge_index)) + r)
        x = F.leaky_relu(self.ln3(self.conv3(x, edge_index)))

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.fc2(x)


@torch.no_grad()
def predict_graphs(model, loader, device):
    model.eval()
    ys = []
    ps = []
    for d in loader:
        d = d.to(device)
        out = model(d.x, d.edge_index, d.batch, edge_drop=0.0, noise=0.0)
        p = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        y = d.y.cpu().numpy()
        ys.extend(y)
        ps.extend(p)
    return np.array(ys), np.array(ps)


def bin_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {
        'auc': float(roc_auc_score(y, p)) if len(set(y)) > 1 else 0.5,
        'accuracy': float(accuracy_score(y, pred)),
        'f1': float(f1_score(y, pred, zero_division=0)),
    }


def train_gat_fold(train_graphs, val_graphs, device, epochs=120, patience=25, batch_size=8):
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)

    labels = np.array([g.y.item() for g in train_graphs])
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    w_pos = len(labels) / (2 * max(n_pos, 1))
    w_neg = len(labels) / (2 * max(n_neg, 1))
    cw = torch.tensor([w_neg, w_pos], dtype=torch.float).to(device)

    model = OIGATv2(n_feat=train_graphs[0].x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=cw)

    best = None
    best_auc = -1
    wait = 0

    for _ in range(epochs):
        model.train()
        for d in train_loader:
            d = d.to(device)
            opt.zero_grad()
            out = model(d.x, d.edge_index, d.batch)
            loss = crit(out, d.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        vy, vp = predict_graphs(model, val_loader, device)
        vauc = roc_auc_score(vy, vp) if len(set(vy)) > 1 else 0.5
        if vauc > best_auc:
            best_auc = vauc
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best)
    return model


def run_gat_internal_external(train_graphs, ext_graphs):
    y = np.array([g.y.item() for g in train_graphs])
    min_class = int(np.bincount(y).min())
    n_splits = max(2, min(4, min_class))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fold_metrics = []
    ext_probs = []
    ext_y_ref = None

    for tr_idx, va_idx in skf.split(np.arange(len(train_graphs)), y):
        tr = [train_graphs[i] for i in tr_idx]
        va = [train_graphs[i] for i in va_idx]

        model = train_gat_fold(tr, va, device)

        va_loader = DataLoader(va, batch_size=8)
        vy, vp = predict_graphs(model, va_loader, device)
        fold_metrics.append(bin_metrics(vy, vp, thr=0.5))

        ext_loader = DataLoader(ext_graphs, batch_size=8)
        ey, ep = predict_graphs(model, ext_loader, device)
        ext_y_ref = ey
        ext_probs.append(ep)

    ext_probs = np.vstack(ext_probs)
    ext_mean_prob = ext_probs.mean(axis=0)
    ext_metrics = bin_metrics(ext_y_ref, ext_mean_prob, thr=0.5)

    internal = {
        'auc_mean': float(np.mean([m['auc'] for m in fold_metrics])),
        'acc_mean': float(np.mean([m['accuracy'] for m in fold_metrics])),
        'f1_mean': float(np.mean([m['f1'] for m in fold_metrics])),
    }
    return internal, ext_metrics, ext_y_ref, ext_mean_prob


def run_lr_internal_external(X_train, y_train, X_ext, y_ext):
    min_class = int(np.bincount(y_train).min())
    n_splits = max(2, min(4, min_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    fold_metrics = []
    ext_probs = []

    for tr_idx, va_idx in skf.split(X_train, y_train):
        Xtr, Xva = X_train[tr_idx], X_train[va_idx]
        ytr, yva = y_train[tr_idx], y_train[va_idx]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xva = scaler.transform(Xva)
        Xex = scaler.transform(X_ext)

        lr = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced', solver='lbfgs', random_state=SEED)
        lr.fit(Xtr, ytr)

        vprob = lr.predict_proba(Xva)[:, 1]
        fold_metrics.append(bin_metrics(yva, vprob, thr=0.5))

        eprob = lr.predict_proba(Xex)[:, 1]
        ext_probs.append(eprob)

    ext_prob = np.vstack(ext_probs).mean(axis=0)
    ext_metrics = bin_metrics(y_ext, ext_prob, thr=0.5)

    internal = {
        'auc_mean': float(np.mean([m['auc'] for m in fold_metrics])),
        'acc_mean': float(np.mean([m['accuracy'] for m in fold_metrics])),
        'f1_mean': float(np.mean([m['f1'] for m in fold_metrics])),
    }
    return internal, ext_metrics, y_ext, ext_prob


def plot_holdout_roc(ds, y, p_gat, p_lr):
    plt.figure(figsize=(5, 5))
    fpr, tpr, _ = roc_curve(y, p_gat)
    auc = roc_auc_score(y, p_gat) if len(set(y)) > 1 else 0.5
    plt.plot(fpr, tpr, label=f'GAT AUC={auc:.3f}')

    fpr2, tpr2, _ = roc_curve(y, p_lr)
    auc2 = roc_auc_score(y, p_lr) if len(set(y)) > 1 else 0.5
    plt.plot(fpr2, tpr2, label=f'LR AUC={auc2:.3f}')

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'External ROC: {ds}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'real_roc_{ds}.png'), dpi=150)
    plt.close()


def main():
    ensure_string_file()
    expr_by_ds, meta = load_multicohort_data()

    string_edges, proteins = load_string_edges()
    ensp_to_gene = load_or_build_ensp_mapping(proteins)

    datasets = sorted(meta['Dataset'].unique())
    results = {'holdouts': {}}

    for holdout in datasets:
        train_ds = [d for d in datasets if d != holdout]

        meta_tr = meta[meta['Dataset'].isin(train_ds)].copy()
        meta_ex = meta[meta['Dataset'] == holdout].copy()

        # genes from training cohorts only, then keep what exists in external
        genes = set(expr_by_ds[train_ds[0]].index)
        for d in train_ds[1:]:
            genes &= set(expr_by_ds[d].index)
        genes &= set(expr_by_ds[holdout].index)
        genes = sorted(genes)

        tr_expr = pd.concat([expr_by_ds[d].loc[genes] for d in train_ds], axis=1)
        tr_expr = tr_expr[meta_tr['SampleID'].tolist()]

        ex_expr = expr_by_ds[holdout].loc[genes]
        ex_expr = ex_expr[meta_ex['SampleID'].tolist()]

        # Clean and impute
        tr_expr = tr_expr.apply(pd.to_numeric, errors='coerce')
        ex_expr = ex_expr.apply(pd.to_numeric, errors='coerce')
        tr_expr = tr_expr.T.fillna(tr_expr.T.mean()).T
        ex_expr = ex_expr.T.fillna(ex_expr.T.mean()).T

        tr_expr = tr_expr.loc[tr_expr.var(axis=1) > 1e-10]
        genes = tr_expr.index.tolist()
        ex_expr = ex_expr.loc[genes]

        # ComBat only on training data
        tr_expr_cb = run_combat_train(
            tr_expr,
            batch_labels=meta_tr['Dataset'].tolist(),
            cond_labels=meta_tr['Condition'].tolist(),
        )

        # Top variable genes from training only
        mad = tr_expr_cb.apply(median_abs_deviation, axis=1)
        top = mad.sort_values(ascending=False).head(min(TOP_K, len(mad))).index.tolist()

        # Build PPI topology
        final_genes, edge_index = build_topology(top, string_edges, ensp_to_gene)
        degree, clustering = compute_structural_features(edge_index, len(final_genes))

        # Build graphs
        tr_graphs = create_graphs(tr_expr_cb, meta_tr, final_genes, edge_index, mad.to_dict(), degree, clustering)
        ex_graphs = create_graphs(ex_expr, meta_ex, final_genes, edge_index, mad.to_dict(), degree, clustering)

        # GAT internal CV + external ensemble
        gat_in, gat_ex, ey, ep_gat = run_gat_internal_external(tr_graphs, ex_graphs)

        # LR baseline with train-corrected matrix and train-stat standardized external
        Xtr = tr_expr_cb.loc[final_genes].T.values
        ytr = meta_tr['Label'].values.astype(int)

        mu = tr_expr_cb.loc[final_genes].mean(axis=1)
        sd = tr_expr_cb.loc[final_genes].std(axis=1).replace(0, 1.0)
        Xex = ((ex_expr.loc[final_genes].subtract(mu, axis=0)).divide(sd, axis=0)).T.values
        yex = meta_ex['Label'].values.astype(int)

        lr_in, lr_ex, _, ep_lr = run_lr_internal_external(Xtr, ytr, Xex, yex)

        plot_holdout_roc(holdout, ey, ep_gat, ep_lr)

        results['holdouts'][holdout] = {
            'n_train': int(len(meta_tr)),
            'n_external': int(len(meta_ex)),
            'train_labels': meta_tr['Condition'].value_counts().to_dict(),
            'external_labels': meta_ex['Condition'].value_counts().to_dict(),
            'n_genes_train_common': int(len(genes)),
            'n_nodes_graph': int(len(final_genes)),
            'n_edges_graph': int(edge_index.shape[1] // 2),
            'gat_internal': gat_in,
            'gat_external': gat_ex,
            'lr_internal': lr_in,
            'lr_external': lr_ex,
        }

        print(f"Holdout {holdout}: GAT ext acc={gat_ex['accuracy']:.3f}, LR ext acc={lr_ex['accuracy']:.3f}")

    # Aggregate plot
    holdouts = list(results['holdouts'].keys())
    gat_acc = [results['holdouts'][h]['gat_external']['accuracy'] for h in holdouts]
    lr_acc = [results['holdouts'][h]['lr_external']['accuracy'] for h in holdouts]

    x = np.arange(len(holdouts))
    w = 0.35
    plt.figure(figsize=(9, 4))
    plt.bar(x - w / 2, gat_acc, width=w, label='GAT external acc')
    plt.bar(x + w / 2, lr_acc, width=w, label='LR external acc')
    plt.axhline(0.76, color='red', linestyle='--', linewidth=1, label='0.76 baseline')
    plt.xticks(x, holdouts, rotation=20)
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Leave-One-Dataset-Out External Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'real_external_accuracy_by_holdout.png'), dpi=150)
    plt.close()

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, 'real_world_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary markdown
    lines = []
    lines.append('# Real-World Multicohort OI Evaluation')
    lines.append('')
    lines.append('Method: leave-one-dataset-out external validation (strict holdout)')
    lines.append('')
    lines.append('| Holdout | GAT Ext Acc | GAT Ext AUC | LR Ext Acc | LR Ext AUC |')
    lines.append('|---|---:|---:|---:|---:|')
    for h in holdouts:
        r = results['holdouts'][h]
        lines.append(
            f"| {h} | {r['gat_external']['accuracy']:.3f} | {r['gat_external']['auc']:.3f} | {r['lr_external']['accuracy']:.3f} | {r['lr_external']['auc']:.3f} |"
        )

    mean_gat = float(np.mean(gat_acc))
    mean_lr = float(np.mean(lr_acc))
    lines.append('')
    lines.append(f'Mean external accuracy (GAT): {mean_gat:.3f}')
    lines.append(f'Mean external accuracy (LR): {mean_lr:.3f}')

    out_md = os.path.join(RESULTS_DIR, 'real_world_summary.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'Saved: {out_json}')
    print(f'Saved: {out_md}')


if __name__ == '__main__':
    main()
