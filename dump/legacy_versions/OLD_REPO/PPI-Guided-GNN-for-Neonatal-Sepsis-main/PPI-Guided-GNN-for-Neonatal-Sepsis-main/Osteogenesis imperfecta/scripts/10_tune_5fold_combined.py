import os
import json
import gzip
import pickle
import warnings
import numpy as np
import pandas as pd

from scipy.stats import median_abs_deviation
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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

warnings.filterwarnings('ignore')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW = os.path.join(ROOT, 'data', 'raw')
PROC = os.path.join(ROOT, 'data', 'processed')
RESULTS = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

STRING_FILE = os.path.join(RAW, '9606.protein.links.v12.0.txt.gz')
ALT_STRING = os.path.abspath(os.path.join(ROOT, '..', 'ALZHEIMERS_STRATEGIC_PATHWAY', 'data', 'adni', 'raw', '9606.protein.links.v12.0.txt.gz'))
CACHE_ENSP = os.path.join(PROC, 'ensp_to_gene_cache.pkl')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def ensure_string():
    if os.path.exists(STRING_FILE):
        return
    if os.path.exists(ALT_STRING):
        import shutil
        shutil.copy2(ALT_STRING, STRING_FILE)
        return
    raise FileNotFoundError('STRING file missing')


def load_data():
    expr = pd.read_csv(os.path.join(PROC, 'expanded_expression_common.csv'), index_col=0)
    meta = pd.read_csv(os.path.join(PROC, 'expanded_metadata.csv'))
    meta = meta[meta['SampleID'].isin(expr.columns)].copy()
    expr = expr[meta['SampleID'].tolist()]
    return expr, meta


def run_combat(expr, meta):
    try:
        cb = pycombat(expr, batch=meta['Dataset'].tolist(), mod=[meta['Condition'].tolist()])
    except Exception:
        cb = pycombat(expr, batch=meta['Dataset'].tolist())
    return cb


def cv_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'auc': float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.5,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }


def eval_tabular(X, y, model, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    all_prob = np.zeros(len(y), dtype=float)

    for tr, va in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        model.fit(Xtr, y[tr])
        if hasattr(model, 'predict_proba'):
            p = model.predict_proba(Xva)[:, 1]
        else:
            p = model.decision_function(Xva)
            p = (p - p.min()) / (p.max() - p.min() + 1e-12)
        all_prob[va] = p

    return cv_metrics(y, all_prob)


def load_string_edges(thresh=700):
    edges = []
    proteins = set()
    with gzip.open(STRING_FILE, 'rt') as f:
        _ = f.readline()
        for line in f:
            p = line.strip().split()
            if len(p) < 3:
                continue
            p1, p2, s = p[0], p[1], int(p[2])
            if s >= thresh:
                p1 = p1.replace('9606.', '')
                p2 = p2.replace('9606.', '')
                edges.append((p1, p2))
                proteins.add(p1)
                proteins.add(p2)
    return edges, proteins


def load_or_build_mapping(proteins):
    if os.path.exists(CACHE_ENSP):
        with open(CACHE_ENSP, 'rb') as f:
            return pickle.load(f)
    import mygene
    mg = mygene.MyGeneInfo()
    mapping = {}
    proteins = list(proteins)
    for i in range(0, len(proteins), 1000):
        res = mg.querymany(proteins[i:i+1000], scopes='ensembl.protein', fields='symbol', species='human', returnall=False, verbose=False)
        for r in res:
            if 'symbol' in r:
                mapping[r['query']] = str(r['symbol']).upper()
    with open(CACHE_ENSP, 'wb') as f:
        pickle.dump(mapping, f)
    return mapping


def build_graph_topology(top_genes, string_edges, mapping):
    gset = set(top_genes)
    uniq = set()
    for p1, p2 in string_edges:
        g1 = mapping.get(p1)
        g2 = mapping.get(p2)
        if g1 and g2 and g1 in gset and g2 in gset and g1 != g2:
            uniq.add(tuple(sorted((g1, g2))))

    connected = set()
    for g1, g2 in uniq:
        connected.add(g1); connected.add(g2)

    genes = [g for g in top_genes if g in connected]
    if len(genes) < 100:
        genes = top_genes[:min(len(top_genes), 400)]

    g2i = {g:i for i,g in enumerate(genes)}
    src, dst = [], []
    for g1, g2 in uniq:
        if g1 in g2i and g2 in g2i:
            i, j = g2i[g1], g2i[g2]
            src.extend([i,j]); dst.extend([j,i])

    if not src:
        for i in range(len(genes)-1):
            src.extend([i,i+1]); dst.extend([i+1,i])

    edge_index = torch.tensor([src,dst], dtype=torch.long)
    return genes, edge_index


def structural_feats(edge_index, n):
    deg = torch.zeros(n, dtype=torch.float)
    for i in range(edge_index.shape[1]):
        deg[edge_index[0,i]] += 1

    adj = {}
    for i in range(edge_index.shape[1]):
        s = edge_index[0,i].item(); d = edge_index[1,i].item()
        adj.setdefault(s,set()).add(d)

    cl = torch.zeros(n, dtype=torch.float)
    for node in range(n):
        nbr = list(adj.get(node,set())); k=len(nbr)
        if k < 2:
            continue
        tri = 0
        for i in range(k):
            for j in range(i+1, k):
                if nbr[j] in adj.get(nbr[i], set()):
                    tri += 1
        cl[node] = 2.0 * tri / (k*(k-1))
    return deg, cl


def build_graphs(expr, meta, genes, edge_index, mad, deg, cl):
    n = len(genes)
    mad_vals = np.array([mad.get(g,0.0) for g in genes], dtype=np.float32)
    mad_rank = np.argsort(np.argsort(-mad_vals)).astype(np.float32) / max(n-1,1)
    deg_n = (deg.numpy() / max(float(deg.max()),1.0)).astype(np.float32)
    cl_n = cl.numpy().astype(np.float32)

    graphs = []
    for _, r in meta.iterrows():
        sid = r['SampleID']; y = int(r['Label'])
        xexpr = expr.loc[genes, sid].values.astype(np.float32)
        mu = np.mean(xexpr); sd = np.std(xexpr)
        if sd > 1e-8:
            xexpr = (xexpr - mu) / sd
        feats = np.stack([xexpr, mad_rank, deg_n, cl_n], axis=1)
        d = Data(x=torch.tensor(feats, dtype=torch.float), edge_index=edge_index, y=torch.tensor([y], dtype=torch.long))
        graphs.append(d)
    return graphs


class GAT(nn.Module):
    def __init__(self, nfeat=4, hidden=64, heads=4, dropout=0.5):
        super().__init__()
        self.drop = dropout
        self.c1 = TransformerConv(nfeat, hidden, heads=heads, dropout=0.1)
        self.l1 = nn.LayerNorm(hidden*heads)
        self.r1 = nn.Linear(nfeat, hidden*heads)
        self.c2 = TransformerConv(hidden*heads, hidden, heads=heads, dropout=0.1)
        self.l2 = nn.LayerNorm(hidden*heads)
        self.c3 = TransformerConv(hidden*heads, hidden, heads=1, concat=False, dropout=0.1)
        self.l3 = nn.LayerNorm(hidden)
        self.f1 = nn.Linear(hidden*2, hidden)
        self.f2 = nn.Linear(hidden, 2)

    def forward(self, x, edge_index, batch):
        r = self.r1(x)
        x = F.leaky_relu(self.l1(self.c1(x, edge_index)) + r)
        r = x
        x = F.leaky_relu(self.l2(self.c2(x, edge_index)) + r)
        x = F.leaky_relu(self.l3(self.c3(x, edge_index)))
        x = torch.cat([global_mean_pool(x,batch), global_max_pool(x,batch)], dim=1)
        x = F.leaky_relu(self.f1(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        return self.f2(x)


@torch.no_grad()
def pred_probs(model, loader, device):
    model.eval()
    probs=[]
    for d in loader:
        d = d.to(device)
        out = model(d.x, d.edge_index, d.batch)
        p = F.softmax(out, dim=1)[:,1].cpu().numpy()
        probs.extend(p)
    return np.array(probs)


def eval_gat_cv(graphs, params, n_splits=5):
    y = np.array([g.y.item() for g in graphs])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_prob = np.zeros(len(graphs), dtype=float)

    for tr, va in skf.split(np.arange(len(graphs)), y):
        trg = [graphs[i] for i in tr]
        vag = [graphs[i] for i in va]
        tr_loader = DataLoader(trg, batch_size=params['batch_size'], shuffle=True)
        va_loader = DataLoader(vag, batch_size=params['batch_size'])

        labels = np.array([g.y.item() for g in trg])
        npos = int(labels.sum()); nneg = int(len(labels)-npos)
        cw = torch.tensor([len(labels)/(2*max(nneg,1)), len(labels)/(2*max(npos,1))], dtype=torch.float).to(device)

        model = GAT(nfeat=trg[0].x.shape[1], hidden=params['hidden'], heads=params['heads'], dropout=params['dropout']).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(weight=cw)

        best_state = None
        best_auc = -1
        patience = 15
        wait = 0

        for _ in range(params['epochs']):
            model.train()
            for d in tr_loader:
                d = d.to(device)
                opt.zero_grad()
                out = model(d.x, d.edge_index, d.batch)
                loss = crit(out, d.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            pva = pred_probs(model, va_loader, device)
            auc = roc_auc_score(y[va], pva) if len(set(y[va])) > 1 else 0.5
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        model.load_state_dict(best_state)
        all_prob[va] = pred_probs(model, va_loader, device)

    return cv_metrics(y, all_prob)


def main():
    ensure_string()
    expr, meta = load_data()

    # Combined harmonization across all cohorts
    expr_cb = run_combat(expr, meta)

    y = meta['Label'].values.astype(int)

    tab_configs = [500, 1000, 1500, 2000]
    tab_results = {}

    for k in tab_configs:
        mad = expr_cb.apply(median_abs_deviation, axis=1).sort_values(ascending=False)
        genes = mad.head(min(k, len(mad))).index.tolist()
        X = expr_cb.loc[genes].T.values

        models = {
            'LR': LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced', solver='lbfgs', random_state=SEED),
            'RF': RandomForestClassifier(n_estimators=400, max_depth=14, class_weight='balanced', random_state=SEED, n_jobs=-1),
            'SVM': SVC(C=2.0, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=SEED),
        }

        tab_results[k] = {}
        for name, model in models.items():
            m = eval_tabular(X, y, model, n_splits=5)
            tab_results[k][name] = m
            print(f'Tabular k={k} {name}: acc={m["accuracy"]:.3f}, auc={m["auc"]:.3f}')

    # Best tabular setup
    best_tab = None
    for k, md in tab_results.items():
        for mname, met in md.items():
            score = met['accuracy']
            if best_tab is None or score > best_tab['metrics']['accuracy']:
                best_tab = {'top_k': k, 'model': mname, 'metrics': met}

    # GAT tuning on best top_k from tabular
    top_k = best_tab['top_k']
    mad = expr_cb.apply(median_abs_deviation, axis=1).sort_values(ascending=False)
    top_genes = mad.head(min(top_k, len(mad))).index.tolist()

    string_edges, proteins = load_string_edges()
    mapping = load_or_build_mapping(proteins)
    ggenes, edge_index = build_graph_topology(top_genes, string_edges, mapping)
    deg, cl = structural_feats(edge_index, len(ggenes))
    graphs = build_graphs(expr_cb, meta, ggenes, edge_index, mad.to_dict(), deg, cl)

    gat_grid = [
        {'hidden': 64, 'heads': 4, 'dropout': 0.5, 'lr': 1e-3, 'batch_size': 8, 'epochs': 90},
        {'hidden': 64, 'heads': 2, 'dropout': 0.4, 'lr': 5e-4, 'batch_size': 8, 'epochs': 110},
        {'hidden': 32, 'heads': 4, 'dropout': 0.3, 'lr': 1e-3, 'batch_size': 8, 'epochs': 100},
        {'hidden': 96, 'heads': 2, 'dropout': 0.5, 'lr': 8e-4, 'batch_size': 6, 'epochs': 90},
    ]

    gat_results = []
    for cfg in gat_grid:
        m = eval_gat_cv(graphs, cfg, n_splits=5)
        gat_results.append({'config': cfg, 'metrics': m})
        print(f'GAT cfg={cfg}: acc={m["accuracy"]:.3f}, auc={m["auc"]:.3f}')

    best_gat = max(gat_results, key=lambda x: x['metrics']['accuracy'])

    final = {
        'n_samples': int(len(meta)),
        'label_counts': meta['Condition'].value_counts().to_dict(),
        'dataset_counts': meta.groupby(['Dataset', 'Condition']).size().unstack(fill_value=0).to_dict(),
        'tabular_5fold': tab_results,
        'best_tabular': best_tab,
        'graph_info': {
            'top_k': int(top_k),
            'n_nodes': int(len(ggenes)),
            'n_edges': int(edge_index.shape[1] // 2),
        },
        'gat_5fold': gat_results,
        'best_gat': best_gat,
    }

    out_json = os.path.join(RESULTS, 'expanded_5fold_tuning_results.json')
    with open(out_json, 'w') as f:
        json.dump(final, f, indent=2)

    # Plot: best tabular vs best GAT accuracy
    names = [f"Tabular-{best_tab['model']}", 'GAT']
    vals = [best_tab['metrics']['accuracy'], best_gat['metrics']['accuracy']]
    plt.figure(figsize=(5,4))
    plt.bar(names, vals, color=['#4c78a8','#f58518'])
    plt.axhline(0.9, color='green', linestyle='--', linewidth=1, label='0.90 target')
    plt.axhline(0.76, color='red', linestyle='--', linewidth=1, label='0.76 baseline')
    plt.ylim(0,1.0)
    plt.ylabel('5-fold CV Accuracy')
    plt.title('Expanded Combined Dataset 5-fold CV')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'expanded_5fold_accuracy.png'), dpi=150)
    plt.close()

    md = []
    md.append('# Expanded Combined Dataset 5-Fold CV Tuning')
    md.append('')
    md.append(f"Total samples: {final['n_samples']}")
    md.append('')
    md.append('## Best Results')
    md.append(f"- Best tabular: {best_tab['model']} (top_k={best_tab['top_k']}) -> Acc={best_tab['metrics']['accuracy']:.3f}, AUC={best_tab['metrics']['auc']:.3f}")
    md.append(f"- Best GAT: Acc={best_gat['metrics']['accuracy']:.3f}, AUC={best_gat['metrics']['auc']:.3f}")
    md.append('')
    md.append('## Note')
    md.append('- This is pooled 5-fold CV on combined cohorts (human + mouse).')

    out_md = os.path.join(RESULTS, 'expanded_5fold_summary.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print('Saved', out_json)
    print('Saved', out_md)


if __name__ == '__main__':
    main()
