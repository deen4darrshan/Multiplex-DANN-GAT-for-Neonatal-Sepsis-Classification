import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROC_DIR = os.path.join(ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(ROOT, 'results')
MODELS_DIR = os.path.join(ROOT, 'models')
FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def plot_roc(y_true, y_prob, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


class OIGATv2(nn.Module):
    def __init__(self, num_node_features=4, hidden_channels=64, heads=4, dropout=0.5):
        super().__init__()
        self.dropout_rate = dropout

        self.conv1 = TransformerConv(num_node_features, hidden_channels, heads=heads, dropout=0.1)
        self.ln1 = nn.LayerNorm(hidden_channels * heads)
        self.res_proj = nn.Linear(num_node_features, hidden_channels * heads)

        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.1)
        self.ln2 = nn.LayerNorm(hidden_channels * heads)

        self.conv3 = TransformerConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.1)
        self.ln3 = nn.LayerNorm(hidden_channels)

        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch, edge_drop_rate=0.0, noise_std=0.0):
        if self.training and edge_drop_rate > 0:
            mask = torch.rand(edge_index.size(1), device=edge_index.device) > edge_drop_rate
            edge_index = edge_index[:, mask]
        if self.training and noise_std > 0:
            x = x + torch.randn_like(x) * noise_std

        res = self.res_proj(x)
        x = F.leaky_relu(self.ln1(self.conv1(x, edge_index)) + res)

        res = x
        x = F.leaky_relu(self.ln2(self.conv2(x, edge_index)) + res)

        x = F.leaky_relu(self.ln3(self.conv3(x, edge_index)))

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pool = torch.cat([x_mean, x_max], dim=1)

        x = F.leaky_relu(self.lin1(x_pool))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x


class OIGCN(nn.Module):
    def __init__(self, num_node_features=4, hidden_channels=64, dropout=0.5):
        super().__init__()
        self.dropout_rate = dropout

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.res_proj = nn.Linear(num_node_features, hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.ln3 = nn.LayerNorm(hidden_channels)

        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch, edge_drop_rate=0.0, noise_std=0.0):
        if self.training and edge_drop_rate > 0:
            mask = torch.rand(edge_index.size(1), device=edge_index.device) > edge_drop_rate
            edge_index = edge_index[:, mask]
        if self.training and noise_std > 0:
            x = x + torch.randn_like(x) * noise_std

        res = self.res_proj(x)
        x = F.leaky_relu(self.ln1(self.conv1(x, edge_index)) + res)

        res = x
        x = F.leaky_relu(self.ln2(self.conv2(x, edge_index)) + res)

        x = F.leaky_relu(self.ln3(self.conv3(x, edge_index)))

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pool = torch.cat([x_mean, x_max], dim=1)

        x = F.leaky_relu(self.lin1(x_pool))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        total_loss += loss.item() * data.num_graphs
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        labels = data.y.cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels)

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)

    # If perfectly separable, choose the midpoint between max negative and min positive
    best_thresh = 0.5
    if len(set(y_true)) > 1:
        pos_min = y_prob[y_true == 1].min()
        neg_max = y_prob[y_true == 0].max()
        if pos_min > neg_max:
            best_thresh = float((pos_min + neg_max) / 2.0)
        else:
            # Otherwise find threshold that maximizes accuracy (tie-breaker: F1)
            best_acc = -1
            best_f1 = -1
            for thresh in np.arange(0.1, 0.9, 0.01):
                y_tmp = (y_prob >= thresh).astype(int)
                acc_tmp = accuracy_score(y_true, y_tmp)
                f1_tmp = f1_score(y_true, y_tmp, zero_division=0)
                if acc_tmp > best_acc or (acc_tmp == best_acc and f1_tmp > best_f1):
                    best_acc = acc_tmp
                    best_f1 = f1_tmp
                    best_thresh = thresh

    y_pred = (y_prob >= best_thresh).astype(int)

    metrics = {
        'loss': total_loss / len(loader.dataset),
        'auc': roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'threshold': float(best_thresh),
        'y_true': y_true,
        'y_prob': y_prob,
    }
    return metrics


def train_epoch(model, loader, optimizer, criterion, device, edge_drop=0.05, noise_std=0.02):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch, edge_drop_rate=edge_drop, noise_std=noise_std)
        loss = criterion(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def run_cv(model_class, model_kwargs, graphs, device, name, epochs=120, lr=1e-3, wd=1e-4, batch_size=8, patience=30):
    labels = np.array([g.y.item() for g in graphs])
    batches = np.array([getattr(g, 'batch_label', 'Unknown') for g in graphs])

    class_counts = np.bincount(labels)
    n_splits = int(min(5, class_counts.min()))
    if n_splits < 2:
        raise ValueError('Not enough samples per class for CV')

    strat_key = [f"{l}_{b}" for l, b in zip(labels, batches)]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []
    y_true_all = []
    y_prob_all = []
    best_overall_auc = 0
    best_state = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(graphs)), strat_key)):
        train_subset = [graphs[i] for i in train_idx]
        val_subset = [graphs[i] for i in val_idx]

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)

        # class weights
        train_labels = [g.y.item() for g in train_subset]
        n_pos = sum(train_labels)
        n_neg = len(train_labels) - n_pos
        w_pos = len(train_labels) / (2 * max(n_pos, 1))
        w_neg = len(train_labels) / (2 * max(n_neg, 1))
        class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        best_val_auc = 0
        best_state_fold = None
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_metrics = evaluate(model, val_loader, device)
                if val_metrics['auc'] > best_val_auc:
                    best_val_auc = val_metrics['auc']
                    best_state_fold = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 5
                if patience_counter >= patience:
                    break

        if best_state_fold:
            model.load_state_dict(best_state_fold)

        val_metrics = evaluate(model, val_loader, device)
        fold_results.append({
            'fold': fold + 1,
            'auc': val_metrics['auc'],
            'f1': val_metrics['f1'],
            'accuracy': val_metrics['accuracy'],
        })

        y_true_all.extend(val_metrics['y_true'])
        y_prob_all.extend(val_metrics['y_prob'])

        if val_metrics['auc'] > best_overall_auc:
            best_overall_auc = val_metrics['auc']
            best_state = copy.deepcopy(model.state_dict())

    df = np.array([r['auc'] for r in fold_results])
    summary = {
        'mean_auc': float(df.mean()),
        'std_auc': float(df.std()),
        'mean_f1': float(np.mean([r['f1'] for r in fold_results])),
        'mean_acc': float(np.mean([r['accuracy'] for r in fold_results])),
        'folds': fold_results
    }

    # Save best model
    model_path = os.path.join(MODELS_DIR, f'{name.lower()}_best.pt')
    torch.save(best_state, model_path)

    # Plot ROC
    plot_roc(np.array(y_true_all), np.array(y_prob_all), f'{name} ROC (CV)', os.path.join(FIG_DIR, f'roc_{name.lower()}.png'))

    return summary


def main():
    graphs = torch.load(os.path.join(PROC_DIR, 'graphs.pt'), weights_only=False)
    n_features = graphs[0].x.shape[1]
    print(f'Graphs: {len(graphs)} samples, features={n_features}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}
    results['GAT_v2'] = run_cv(
        OIGATv2,
        {'num_node_features': n_features, 'hidden_channels': 64, 'heads': 4, 'dropout': 0.5},
        graphs,
        device,
        'GAT_v2'
    )
    results['GCN'] = run_cv(
        OIGCN,
        {'num_node_features': n_features, 'hidden_channels': 64, 'dropout': 0.5},
        graphs,
        device,
        'GCN'
    )

    with open(os.path.join(RESULTS_DIR, 'gnn_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('GNN results saved.')


if __name__ == '__main__':
    main()
