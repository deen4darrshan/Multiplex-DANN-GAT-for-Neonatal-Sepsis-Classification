import os
import json
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(ROOT, 'results')
FIG_DIR = os.path.join(ROOT, 'figures')

BASELINE = os.path.join(RESULTS_DIR, 'baseline_results.json')
GNN = os.path.join(RESULTS_DIR, 'gnn_results.json')
OUT = os.path.join(RESULTS_DIR, 'summary.md')


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def main():
    base = load_json(BASELINE) or {}
    gnn = load_json(GNN) or {}

    def best_acc(d):
        best = None
        for k, v in d.items():
            if best is None or v.get('mean_acc', 0) > best[1]:
                best = (k, v.get('mean_acc', 0))
        return best

    base_best = best_acc(base)
    gnn_best = best_acc(gnn)

    lines = []
    lines.append('# Osteogenesis Imperfecta Results Summary')
    lines.append('')
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append('')

    if base:
        lines.append('## Baselines')
        for k, v in base.items():
            lines.append(f"- {k}: AUC={v['mean_auc']:.3f} +/- {v['std_auc']:.3f}, F1={v['mean_f1']:.3f}, Acc={v['mean_acc']:.3f}")
        lines.append('')
    if gnn:
        lines.append('## GNN Models')
        for k, v in gnn.items():
            lines.append(f"- {k}: AUC={v['mean_auc']:.3f} +/- {v['std_auc']:.3f}, F1={v['mean_f1']:.3f}, Acc={v['mean_acc']:.3f}")
        lines.append('')

    lines.append('## Best Accuracy')
    if base_best:
        lines.append(f"- Best baseline: {base_best[0]} (Acc={base_best[1]:.3f})")
    if gnn_best:
        lines.append(f"- Best GNN: {gnn_best[0]} (Acc={gnn_best[1]:.3f})")

    lines.append('')
    lines.append('## Figures')
    for fname in ['pca_before_combat.png','pca_after_combat.png','roc_logisticregression.png','roc_randomforest.png','roc_gat_v2.png','roc_gcn.png']:
        fpath = os.path.join(FIG_DIR, fname)
        if os.path.exists(fpath):
            lines.append(f"- {fname}")

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Summary written to {OUT}")


if __name__ == '__main__':
    main()
