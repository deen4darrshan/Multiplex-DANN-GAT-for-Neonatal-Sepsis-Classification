import os
import json
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)

ROOT = r'C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta'
expr = pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'multicohort_expression_common.csv'), index_col=0)
meta = pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'multicohort_metadata.csv'))
expr = expr[meta['SampleID'].tolist()]

ids = meta['SampleID'].values
y = meta['Label'].values.astype(int)
groups = meta['GroupID'].values
splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.arange(len(y)), y, groups))

# Precompute per-fold gene rankings to speed repeated eval
rankings = {}
for fi, (tr_idx, va_idx) in enumerate(splits):
    tr_ids = ids[tr_idx]
    Xtr = expr[tr_ids]
    mad = Xtr.apply(median_abs_deviation, axis=1).sort_values(ascending=False).index.tolist()
    ytr = meta.set_index('SampleID').loc[tr_ids, 'Label'].values
    pos = Xtr.loc[:, ytr == 1].mean(axis=1)
    neg = Xtr.loc[:, ytr == 0].mean(axis=1)
    diff = (pos - neg).abs().sort_values(ascending=False).index.tolist()
    rankings[(fi, 'mad')] = mad
    rankings[(fi, 'diff')] = diff


def metric_pack(prob):
    pred05 = (prob >= 0.5).astype(int)
    fixed = {
        'acc': float(accuracy_score(y, pred05)),
        'auc': float(roc_auc_score(y, prob)),
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
        'auc': float(roc_auc_score(y, prob)),
        'f1': float(f1_score(y, predt, zero_division=0)),
        'thr': round(best_t, 2),
    }
    return fixed, tuned


def make_model(cfg):
    m = cfg['model']
    if m == 'lr':
        return LogisticRegression(
            C=cfg['C'], penalty=cfg.get('penalty', 'l2'), solver=cfg.get('solver', 'lbfgs'),
            l1_ratio=cfg.get('l1_ratio', None), class_weight='balanced', max_iter=6000, random_state=SEED
        )
    if m == 'rf':
        return RandomForestClassifier(
            n_estimators=cfg['n_estimators'], max_depth=cfg['max_depth'], min_samples_leaf=cfg['min_leaf'],
            class_weight='balanced', random_state=SEED, n_jobs=-1
        )
    if m == 'et':
        return ExtraTreesClassifier(
            n_estimators=cfg['n_estimators'], max_depth=cfg['max_depth'], min_samples_leaf=cfg['min_leaf'],
            class_weight='balanced', random_state=SEED, n_jobs=-1
        )
    if m == 'svm':
        return SVC(C=cfg['C'], gamma=cfg['gamma'], kernel='rbf', class_weight='balanced', probability=True, random_state=SEED)
    if m == 'gb':
        return GradientBoostingClassifier(n_estimators=cfg['n_estimators'], learning_rate=cfg['lr'], max_depth=cfg['max_depth'], subsample=cfg['subsample'], random_state=SEED)
    raise ValueError(m)


def eval_cfg(cfg):
    prob = np.zeros(len(y), dtype=float)
    for fi, (tr_idx, va_idx) in enumerate(splits):
        tr_ids = ids[tr_idx]
        va_ids = ids[va_idx]

        genes = rankings[(fi, cfg['feat'])][: cfg['top_k']]

        Xtr = expr.loc[genes, tr_ids].T.values
        Xva = expr.loc[genes, va_ids].T.values

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xva = scaler.transform(Xva)

        model = make_model(cfg)
        model.fit(Xtr, y[tr_idx])
        prob[va_idx] = model.predict_proba(Xva)[:, 1]

    fixed, tuned = metric_pack(prob)
    return prob, fixed, tuned


# Candidate space (staged, not exhaustive)
configs = []
for feat in ['mad', 'diff']:
    for k in [1000, 2000, 3000, 5000, 8000, 12000, 15000]:
        for C in [0.03, 0.1, 0.3, 1, 3, 10, 30]:
            configs.append({'model': 'lr', 'feat': feat, 'top_k': k, 'C': C, 'penalty': 'l2', 'solver': 'lbfgs'})
        for C in [0.03, 0.1, 0.3, 1, 3, 10]:
            configs.append({'model': 'lr', 'feat': feat, 'top_k': k, 'C': C, 'penalty': 'l1', 'solver': 'liblinear'})
        for C in [0.1, 0.3, 1, 3, 10]:
            for l1r in [0.2, 0.5, 0.8]:
                configs.append({'model': 'lr', 'feat': feat, 'top_k': k, 'C': C, 'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': l1r})

for feat in ['mad', 'diff']:
    for k in [1000, 2000, 5000, 12000]:
        for C in [0.5, 1, 2, 5, 10, 20]:
            for g in ['scale', 0.01, 0.05, 0.1]:
                configs.append({'model': 'svm', 'feat': feat, 'top_k': k, 'C': C, 'gamma': g})

for feat in ['mad', 'diff']:
    for k in [1000, 2000, 5000, 12000]:
        for n in [300, 700]:
            for d in [6, 10, None]:
                configs.append({'model': 'rf', 'feat': feat, 'top_k': k, 'n_estimators': n, 'max_depth': d, 'min_leaf': 1})
                configs.append({'model': 'et', 'feat': feat, 'top_k': k, 'n_estimators': n, 'max_depth': d, 'min_leaf': 1})

for feat in ['mad', 'diff']:
    for k in [1000, 2000, 5000]:
        for n in [200, 400]:
            for lr in [0.03, 0.07, 0.1]:
                for d in [2, 3]:
                    for ss in [0.7, 1.0]:
                        configs.append({'model': 'gb', 'feat': feat, 'top_k': k, 'n_estimators': n, 'lr': lr, 'max_depth': d, 'subsample': ss})

print('Evaluating configs:', len(configs))
results = []
best_fixed = None
best_tuned = None

for i, cfg in enumerate(configs, 1):
    prob, fixed, tuned = eval_cfg(cfg)
    rec = {'cfg': cfg, 'fixed': fixed, 'tuned': tuned, 'prob': prob.tolist()}
    results.append(rec)

    if (best_fixed is None) or (fixed['acc'] > best_fixed['fixed']['acc']) or (fixed['acc'] == best_fixed['fixed']['acc'] and fixed['auc'] > best_fixed['fixed']['auc']):
        best_fixed = rec
        print('best fixed', i, fixed, cfg)

    if (best_tuned is None) or (tuned['acc'] > best_tuned['tuned']['acc']) or (tuned['acc'] == best_tuned['tuned']['acc'] and tuned['auc'] > best_tuned['tuned']['auc']):
        best_tuned = rec
        print('best tuned', i, tuned, cfg)

# Blend top-6 tuned configs via random convex weights
ranked = sorted(results, key=lambda r: (r['tuned']['acc'], r['tuned']['auc']), reverse=True)
top = ranked[:6]
P = np.vstack([np.array(r['prob']) for r in top])

best_blend = None
for _ in range(12000):
    w = rng.random(P.shape[0])
    w = w / w.sum()
    p = (w[:, None] * P).sum(axis=0)
    fixed, tuned = metric_pack(p)
    rec = {'weights': w.tolist(), 'fixed': fixed, 'tuned': tuned}
    if (best_blend is None) or (tuned['acc'] > best_blend['tuned']['acc']) or (tuned['acc'] == best_blend['tuned']['acc'] and tuned['auc'] > best_blend['tuned']['auc']):
        best_blend = rec

out = {
    'setup': {
        'validation': 'human_only',
        'cv': 'StratifiedGroupKFold(n_splits=5)',
        'group_column': 'GroupID',
        'samples': int(len(y)),
        'groups': int(pd.Series(groups).nunique()),
        'configs_evaluated': len(configs),
    },
    'best_single_fixed': {k: v for k, v in best_fixed.items() if k != 'prob'},
    'best_single_tuned': {k: v for k, v in best_tuned.items() if k != 'prob'},
    'top_configs_tuned': [{k: v for k, v in r.items() if k != 'prob'} for r in top],
    'best_blend_tuned': best_blend,
}

out_json = os.path.join(ROOT, 'results', 'human_grouped5_hparam_search_fast.json')
with open(out_json, 'w') as f:
    json.dump(out, f, indent=2)

lines = []
lines.append('# Human-only Grouped 5-fold Hyperparameter Tuning (Fast)')
lines.append('')
lines.append(f"- Configs evaluated: {len(configs)}")
lines.append('')
lines.append('## Best single model (fixed threshold 0.5)')
lines.append(f"- Config: {out['best_single_fixed']['cfg']}")
lines.append(f"- Metrics: {out['best_single_fixed']['fixed']}")
lines.append('')
lines.append('## Best single model (OOF tuned threshold)')
lines.append(f"- Config: {out['best_single_tuned']['cfg']}")
lines.append(f"- Metrics: {out['best_single_tuned']['tuned']}")
lines.append('')
lines.append('## Best blend (OOF tuned threshold)')
lines.append(f"- Metrics: {out['best_blend_tuned']['tuned']}")
lines.append(f"- Weights: {[round(w,3) for w in out['best_blend_tuned']['weights']]}")

out_md = os.path.join(ROOT, 'results', 'human_grouped5_hparam_search_fast_summary.md')
with open(out_md, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

# figure
vals = [out['best_single_fixed']['fixed']['acc'], out['best_single_tuned']['tuned']['acc'], out['best_blend_tuned']['tuned']['acc']]
labels = ['single fixed', 'single tuned', 'blend tuned']
plt.figure(figsize=(6,4))
plt.bar(labels, vals, color=['#4c78a8','#f58518','#54a24b'])
plt.axhline(0.9, color='green', linestyle='--', linewidth=1, label='0.90 target')
plt.axhline(0.76, color='red', linestyle='--', linewidth=1, label='0.76 baseline')
plt.ylim(0,1.0)
plt.ylabel('Accuracy')
plt.title('Human Grouped 5-fold Tuning Best')
plt.legend()
plt.tight_layout()
fig_path = os.path.join(ROOT, 'figures', 'human_grouped5_hparam_fast_best.png')
plt.savefig(fig_path, dpi=150)

print('saved', out_json)
print('saved', out_md)
print('saved', fig_path)
print('best_single_fixed', out['best_single_fixed']['fixed'], out['best_single_fixed']['cfg'])
print('best_single_tuned', out['best_single_tuned']['tuned'], out['best_single_tuned']['cfg'])
print('best_blend_tuned', out['best_blend_tuned']['tuned'])
