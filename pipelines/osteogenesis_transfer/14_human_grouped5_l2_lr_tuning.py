import os
import json
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EXPR_PATH = os.path.join(ROOT, 'data', 'processed', 'multicohort_expression_common.csv')
META_PATH = os.path.join(ROOT, 'data', 'processed', 'multicohort_metadata.csv')
OUT_JSON = os.path.join(ROOT, 'results', 'human_grouped5_l2_lr_tuning.json')
OUT_MD = os.path.join(ROOT, 'results', 'human_grouped5_l2_lr_tuning_summary.md')


def metric_pack(y, prob):
    pred05 = (prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y, prob))
    fixed = {
        'acc': float(accuracy_score(y, pred05)),
        'auc': auc,
        'f1': float(f1_score(y, pred05, zero_division=0)),
    }

    best_t = 0.5
    best_acc = -1
    for t in np.arange(0.05, 0.96, 0.01):
        acc = accuracy_score(y, (prob >= t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)

    predt = (prob >= best_t).astype(int)
    tuned = {
        'acc': float(accuracy_score(y, predt)),
        'auc': auc,
        'f1': float(f1_score(y, predt, zero_division=0)),
        'thr': round(best_t, 2),
    }
    return fixed, tuned


def main():
    expr = pd.read_csv(EXPR_PATH, index_col=0)
    meta = pd.read_csv(META_PATH)
    expr = expr[meta['SampleID'].tolist()]

    ids = meta['SampleID'].values
    y = meta['Label'].values.astype(int)
    groups = meta['GroupID'].values

    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(len(y)), y, groups))

    rankings = {}
    for fi, (tr_idx, _) in enumerate(splits):
        tr_ids = ids[tr_idx]
        Xtr = expr[tr_ids]
        rankings[(fi, 'mad')] = Xtr.apply(median_abs_deviation, axis=1).sort_values(ascending=False).index.tolist()

        ytr = meta.set_index('SampleID').loc[tr_ids, 'Label'].values
        pos = Xtr.loc[:, ytr == 1].mean(axis=1)
        neg = Xtr.loc[:, ytr == 0].mean(axis=1)
        rankings[(fi, 'diff')] = (pos - neg).abs().sort_values(ascending=False).index.tolist()

    configs = []
    for feat in ['mad', 'diff']:
        for k in [2000, 5000, 8000, 12000, 15000]:
            for C in [0.1, 0.3, 1, 3, 5, 8, 10, 15, 20, 30, 50, 100]:
                configs.append({'feat': feat, 'top_k': k, 'C': C})

    best_fixed = None
    best_tuned = None
    records = []

    for cfg in configs:
        oof = np.zeros(len(y), dtype=float)

        for fi, (tr_idx, va_idx) in enumerate(splits):
            tr_ids = ids[tr_idx]
            va_ids = ids[va_idx]
            genes = rankings[(fi, cfg['feat'])][: cfg['top_k']]

            Xtr = expr.loc[genes, tr_ids].T.values
            Xva = expr.loc[genes, va_ids].T.values

            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xva = scaler.transform(Xva)

            m = LogisticRegression(
                C=cfg['C'],
                penalty='l2',
                solver='lbfgs',
                class_weight='balanced',
                max_iter=2000,
                random_state=42,
            )
            m.fit(Xtr, y[tr_idx])
            oof[va_idx] = m.predict_proba(Xva)[:, 1]

        fixed, tuned = metric_pack(y, oof)
        rec = {'cfg': cfg, 'fixed': fixed, 'tuned': tuned}
        records.append(rec)

        if (best_fixed is None) or (fixed['acc'] > best_fixed['fixed']['acc']) or (fixed['acc'] == best_fixed['fixed']['acc'] and fixed['auc'] > best_fixed['fixed']['auc']):
            best_fixed = rec

        if (best_tuned is None) or (tuned['acc'] > best_tuned['tuned']['acc']) or (tuned['acc'] == best_tuned['tuned']['acc'] and tuned['auc'] > best_tuned['tuned']['auc']):
            best_tuned = rec

    out = {
        'configs': len(configs),
        'best_fixed': best_fixed,
        'best_tuned': best_tuned,
        'top10': sorted(records, key=lambda r: (r['tuned']['acc'], r['tuned']['auc']), reverse=True)[:10],
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('# Human Grouped 5-fold L2 LR Tuning\n\n')
        f.write(f"Configs: {out['configs']}\n\n")
        f.write(f"Best fixed: {out['best_fixed']}\n\n")
        f.write(f"Best tuned: {out['best_tuned']}\n")

    print('Saved', OUT_JSON)
    print('Saved', OUT_MD)


if __name__ == '__main__':
    main()
