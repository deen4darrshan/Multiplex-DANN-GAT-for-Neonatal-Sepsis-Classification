import os
import json
import gzip
import pickle
import warnings
import numpy as np
import pandas as pd

from scipy.stats import median_abs_deviation
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool

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


def load_human_data():
    expr = pd.read_csv(os.path.join(PROC, 'multicohort_expression_common.csv'), index_col=0)
    meta = pd.read_csv(os.path.join(PROC, 'multicohort_metadata.csv'))
    meta = meta[meta['SampleID'].isin(expr.columns)].copy()
    expr = expr[meta['SampleID'].tolist()]
    return expr, meta


def metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'auc': float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.5,
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }


def load_string_edges(thresh=700):
    edges = []
    prots = set()
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
                prots.add(p1)
                prots.add(p2)
    return edges, prots


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


def build_topology(top_genes, string_edges, mapping):
    gset = set(top_genes)
    uniq = set()
    for p1, p2 in string_edges:
        g1 = mapping.get(p1)
        g2 = mapping.get(p2)
        if g1 and g2 and g1 != g2 and g1 in gset and g2 in gset:
            uniq.add(tuple(sorted((g1, g2))))

    connected = set()
    for g1, g2 in uniq:
        connected.add(g1)
        connected.add(g2)

    genes = [g for g in top_genes if g in connected]
    if len(genes) < 100:
        genes = top_genes[:min(len(top_genes), 400)]

    g2i = {g: i for i, g in enumerate(genes)}
    src, dst = [], []
    for g1, g2 in uniq:
        if g1 in g2i and g2 in g2i:
            i, j = g2i[g1], g2i[g2]
            src.extend([i, j])
            dst.extend([j, i])

    if not src:
        for i in range(len(genes) - 1):
            src.extend([i, i + 1])
            dst.extend([i + 1, i])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return genes, edge_index


def graph_feats(edge_index, n_nodes):
    deg = torch.zeros(n_nodes, dtype=torch.float)
    for i in range(edge_index.shape[1]):
        deg[edge_index[0, i]] += 1

    adj = {}
    for i in range(edge_index.shape[1]):
        s = edge_index[0, i].item()
        d = edge_index[1, i].item()
        adj.setdefault(s, set()).add(d)

    cl = torch.zeros(n_nodes, dtype=torch.float)
    for n in range(n_nodes):
        nbr = list(adj.get(n, set()))
        k = len(nbr)
        if k < 2:
            continue
        tri = 0
        for i in range(k):
            for j in range(i + 1, k):
                if nbr[j] in adj.get(nbr[i], set()):
                    tri += 1
        cl[n] = 2.0 * tri / (k * (k - 1))
    return deg, cl


def build_graphs(expr, sample_ids, labels, genes, edge_index, mad_dict, deg, cl):
    n = len(genes)
    mad_vals = np.array([mad_dict.get(g, 0.0) for g in genes], dtype=np.float32)
    mad_rank = np.argsort(np.argsort(-mad_vals)).astype(np.float32) / max(n - 1, 1)
    deg_n = (deg.numpy() / max(float(deg.max()), 1.0)).astype(np.float32)
    cl_n = cl.numpy().astype(np.float32)

    graphs = []
    for sid, y in zip(sample_ids, labels):
        xexpr = expr.loc[genes, sid].values.astype(np.float32)
        mu = np.mean(xexpr)
        sd = np.std(xexpr)
        if sd > 1e-8:
            xexpr = (xexpr - mu) / sd
        feats = np.stack([xexpr, mad_rank, deg_n, cl_n], axis=1)
        d = Data(x=torch.tensor(feats, dtype=torch.float), edge_index=edge_index, y=torch.tensor([int(y)], dtype=torch.long))
        graphs.append(d)
    return graphs


class GAT(nn.Module):
    def __init__(self, n_feat=4, hidden=64, heads=4, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.c1 = TransformerConv(n_feat, hidden, heads=heads, dropout=0.1)
        self.l1 = nn.LayerNorm(hidden * heads)
        self.r = nn.Linear(n_feat, hidden * heads)
        self.c2 = TransformerConv(hidden * heads, hidden, heads=heads, dropout=0.1)
        self.l2 = nn.LayerNorm(hidden * heads)
        self.c3 = TransformerConv(hidden * heads, hidden, heads=1, concat=False, dropout=0.1)
        self.l3 = nn.LayerNorm(hidden)
        self.f1 = nn.Linear(hidden * 2, hidden)
        self.f2 = nn.Linear(hidden, 2)

    def forward(self, x, edge_index, batch):
        r = self.r(x)
        x = F.leaky_relu(self.l1(self.c1(x, edge_index)) + r)
        r = x
        x = F.leaky_relu(self.l2(self.c2(x, edge_index)) + r)
        x = F.leaky_relu(self.l3(self.c3(x, edge_index)))
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x = F.leaky_relu(self.f1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.f2(x)


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs = []
    for d in loader:
        d = d.to(device)
        out = model(d.x, d.edge_index, d.batch)
        p = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        probs.extend(p)
    return np.array(probs)


def run():
    ensure_string()
    expr, meta = load_human_data()

    y = meta['Label'].values.astype(int)
    groups = meta['GroupID'].values
    sample_ids = meta['SampleID'].values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(np.arange(len(y)), y, groups))

    # Tabular grouped 5-fold tuning (feature selection done inside each train fold)
    top_ks = [200, 500, 1000, 1500, 2000]
    tab_models = {
        'LR': lambda: LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced', solver='lbfgs', random_state=SEED),
        'RF': lambda: RandomForestClassifier(n_estimators=500, max_depth=16, class_weight='balanced', random_state=SEED, n_jobs=-1),
        'SVM': lambda: SVC(C=2.0, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=SEED),
    }

    tab_results = {}
    for k in top_ks:
        tab_results[k] = {}
        for mname, ctor in tab_models.items():
            oof = np.zeros(len(y), dtype=float)
            for tr_idx, va_idx in splits:
                tr_ids = sample_ids[tr_idx]
                va_ids = sample_ids[va_idx]

                tr_expr = expr[tr_ids]
                mad = tr_expr.apply(median_abs_deviation, axis=1).sort_values(ascending=False)
                genes = mad.head(min(k, len(mad))).index.tolist()

                Xtr = tr_expr.loc[genes].T.values
                Xva = expr.loc[genes, va_ids].T.values

                scaler = StandardScaler()
                Xtr = scaler.fit_transform(Xtr)
                Xva = scaler.transform(Xva)

                model = ctor()
                model.fit(Xtr, y[tr_idx])
                p = model.predict_proba(Xva)[:, 1]
                oof[va_idx] = p

            m = metrics(y, oof)
            tab_results[k][mname] = m
            print(f'Grouped5 Human k={k} {mname}: acc={m["accuracy"]:.3f}, auc={m["auc"]:.3f}')

    # Choose best tabular config by accuracy then auc
    best_tab = None
    for k, md in tab_results.items():
        for mname, m in md.items():
            if best_tab is None:
                best_tab = {'top_k': k, 'model': mname, 'metrics': m}
            else:
                b = best_tab['metrics']
                if (m['accuracy'] > b['accuracy']) or (m['accuracy'] == b['accuracy'] and m['auc'] > b['auc']):
                    best_tab = {'top_k': k, 'model': mname, 'metrics': m}

    # Grouped 5-fold GAT using best top_k from tabular
    string_edges, proteins = load_string_edges(700)
    mapping = load_or_build_mapping(proteins)

    gat_cfg = {'hidden': 64, 'heads': 4, 'dropout': 0.5, 'lr': 1e-3, 'batch_size': 8, 'epochs': 100, 'patience': 15}
    oof_gat = np.zeros(len(y), dtype=float)
    graph_meta = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for tr_idx, va_idx in splits:
        tr_ids = sample_ids[tr_idx]
        va_ids = sample_ids[va_idx]
        ytr = y[tr_idx]

        tr_expr = expr[tr_ids]
        mad = tr_expr.apply(median_abs_deviation, axis=1).sort_values(ascending=False)
        top_genes = mad.head(min(best_tab['top_k'], len(mad))).index.tolist()

        genes, edge_index = build_topology(top_genes, string_edges, mapping)
        deg, cl = graph_feats(edge_index, len(genes))
        graph_meta = {'n_nodes': len(genes), 'n_edges': int(edge_index.shape[1] // 2)}

        trg = build_graphs(expr, tr_ids, ytr, genes, edge_index, mad.to_dict(), deg, cl)
        vag = build_graphs(expr, va_ids, y[va_idx], genes, edge_index, mad.to_dict(), deg, cl)

        tr_loader = DataLoader(trg, batch_size=gat_cfg['batch_size'], shuffle=True)
        va_loader = DataLoader(vag, batch_size=gat_cfg['batch_size'])

        n_pos = int(ytr.sum())
        n_neg = int(len(ytr) - n_pos)
        cw = torch.tensor([len(ytr) / (2 * max(n_neg, 1)), len(ytr) / (2 * max(n_pos, 1))], dtype=torch.float).to(device)

        model = GAT(n_feat=4, hidden=gat_cfg['hidden'], heads=gat_cfg['heads'], dropout=gat_cfg['dropout']).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=gat_cfg['lr'], weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(weight=cw)

        best_state = None
        best_auc = -1
        wait = 0

        for _ in range(gat_cfg['epochs']):
            model.train()
            for d in tr_loader:
                d = d.to(device)
                opt.zero_grad()
                out = model(d.x, d.edge_index, d.batch)
                loss = crit(out, d.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            pva = predict_probs(model, va_loader, device)
            auc = roc_auc_score(y[va_idx], pva) if len(set(y[va_idx])) > 1 else 0.5
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= gat_cfg['patience']:
                    break

        model.load_state_dict(best_state)
        oof_gat[va_idx] = predict_probs(model, va_loader, device)

    gat_metrics = metrics(y, oof_gat)
    print(f'Grouped5 Human GAT: acc={gat_metrics["accuracy"]:.3f}, auc={gat_metrics["auc"]:.3f}')

    out = {
        'setup': {
            'validation': 'human_only',
            'cv': 'StratifiedGroupKFold(n_splits=5)',
            'group_column': 'GroupID',
            'n_samples': int(len(meta)),
            'label_counts': meta['Condition'].value_counts().to_dict(),
            'dataset_counts': meta.groupby(['Dataset', 'Condition']).size().unstack(fill_value=0).to_dict(),
        },
        'tabular_grouped_5fold': tab_results,
        'best_tabular': best_tab,
        'gat_grouped_5fold': {
            'config': gat_cfg,
            'metrics': gat_metrics,
            'graph_info_last_fold': graph_meta,
        },
    }

    out_json = os.path.join(RESULTS, 'human_grouped5_results.json')
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)

    # Figure
    names = [f"Tabular-{best_tab['model']}", 'GAT']
    vals = [best_tab['metrics']['accuracy'], gat_metrics['accuracy']]
    plt.figure(figsize=(5, 4))
    plt.bar(names, vals, color=['#4c78a8', '#f58518'])
    plt.axhline(0.9, color='green', linestyle='--', linewidth=1, label='0.90 target')
    plt.axhline(0.76, color='red', linestyle='--', linewidth=1, label='0.76 baseline')
    plt.ylim(0, 1.0)
    plt.ylabel('Grouped 5-fold Accuracy')
    plt.title('Human-only Grouped 5-fold CV')
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(FIG, 'human_grouped5_accuracy.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # Markdown summary
    lines = []
    lines.append('# Human-only Grouped 5-Fold Results')
    lines.append('')
    lines.append('- Validation data: human only')
    lines.append('- CV: StratifiedGroupKFold, 5 folds, grouping by GroupID')
    lines.append(f"- Total samples: {len(meta)}")
    lines.append('')
    lines.append('## Best')
    lines.append(f"- Best tabular: {best_tab['model']} (top_k={best_tab['top_k']}) -> Acc={best_tab['metrics']['accuracy']:.3f}, AUC={best_tab['metrics']['auc']:.3f}, F1={best_tab['metrics']['f1']:.3f}")
    lines.append(f"- GAT: Acc={gat_metrics['accuracy']:.3f}, AUC={gat_metrics['auc']:.3f}, F1={gat_metrics['f1']:.3f}")

    out_md = os.path.join(RESULTS, 'human_grouped5_summary.md')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('Saved', out_json)
    print('Saved', out_md)


if __name__ == '__main__':
    run()
