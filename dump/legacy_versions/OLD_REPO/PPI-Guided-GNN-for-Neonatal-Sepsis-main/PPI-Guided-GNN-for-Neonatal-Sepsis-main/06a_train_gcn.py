"""
Module D - GCN Training with Model Weight Saving

Saves the best model weights from each fold and the overall best model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import dropout_edge

# ... (Imports)

# Hyperparameters
HIDDEN_CHANNELS = 32
NUM_LAYERS = 2
DROPOUT = 0.7
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0005
N_SPLITS = 5
WEIGHT_DECAY = 1e-3

# Paths
GRAPH_DIR = "data/graphs"
OUT_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"\n=== GCN CONFIGURATION (OPTIMIZED) ===")
print(f"Hidden Channels: {HIDDEN_CHANNELS}")
print(f"Layers: {NUM_LAYERS}")
print(f"Dropout: {DROPOUT}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Weight Decay: {WEIGHT_DECAY}")
print(f"Epochs: {EPOCHS}")
print(f"Model weights will be saved to: {MODEL_DIR}/")
print(f"==========================\n")

# ... (GCN Class - no change needed, uses HIDDEN_CHANNELS)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Data Augmentation: Dropout Edge
        # Remove 5% of edges randomly during training
        edge_index, _ = dropout_edge(data.edge_index, p=0.05, force_undirected=True, training=True)
        
        out = model(data.x, edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ... (rest of the file)

# Update Helper: Since we are creating a partial replace, we need to target specific blocks carefully.
# The user wants "on-the-fly augmentation" inside train_epoch.


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        probs = F.softmax(out, dim=1)[:, 1]
        preds = out.argmax(dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
    auroc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    return auroc, acc

def train_cv(data_list):
    labels = np.array([d.y.item() for d in data_list])
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    fold_aurocs, fold_accs = [], []
    best_overall_auroc = 0
    best_overall_model_state = None
    
    print(f"\n{'='*60}")
    print("Training GCN with 5-Fold Cross-Validation")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(data_list)), labels)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        
        train_data = [data_list[i] for i in train_idx]
        val_data = [data_list[i] for i in val_idx]
        
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
        
        in_channels = data_list[0].x.shape[1]
        model = GCN(in_channels, HIDDEN_CHANNELS, num_classes=2, dropout=DROPOUT).to(device)
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if fold == 0:
            print(f"Model parameters: {num_params:,}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        
        best_auroc = 0
        best_model_state = None
        
        pbar = tqdm(range(1, EPOCHS + 1), desc=f"Fold {fold+1}", ncols=80)
        for epoch in pbar:
            loss = train_epoch(model, train_loader, optimizer, criterion)
            
            if epoch % 10 == 0 or epoch == EPOCHS:
                auroc, acc = evaluate(model, val_loader)
                pbar.set_postfix({'loss': f'{loss:.4f}', 'auroc': f'{auroc:.3f}'})
                
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_model_state = model.state_dict().copy()
        
        # Save best model for this fold
        fold_model_path = os.path.join(MODEL_DIR, f"gcn_fold{fold+1}.pt")
        torch.save(best_model_state, fold_model_path)
        print(f"Saved fold {fold+1} best model (AUROC={best_auroc:.4f}) to {fold_model_path}")
        
        # Track overall best
        if best_auroc > best_overall_auroc:
            best_overall_auroc = best_auroc
            best_overall_model_state = best_model_state.copy()
        
        fold_aurocs.append(best_auroc)
        _, final_acc = evaluate(model, val_loader)
        fold_accs.append(final_acc)
        print(f"Fold {fold+1} Best AUROC: {best_auroc:.4f}")
    
    # Save overall best model
    best_model_path = os.path.join(MODEL_DIR, "gcn_best.pt")
    torch.save({
        'model_state_dict': best_overall_model_state,
        'auroc': best_overall_auroc,
        'config': {
            'in_channels': data_list[0].x.shape[1],
            'hidden_channels': HIDDEN_CHANNELS,
            'num_classes': 2,
            'dropout': DROPOUT
        }
    }, best_model_path)
    print(f"\n✓ Saved overall best model (AUROC={best_overall_auroc:.4f}) to {best_model_path}")
    
    mean_auroc = np.mean(fold_aurocs)
    std_auroc = np.std(fold_aurocs)
    mean_acc = np.mean(fold_accs)
    
    return mean_auroc, std_auroc, mean_acc, fold_aurocs, best_overall_auroc

def main():
    data_list = load_graphs()
    gcn_auroc, gcn_std, gcn_acc, fold_aurocs, best_auroc = train_cv(data_list)
    
    # Save results
    results = pd.DataFrame({
        'Model': ['GCN'],
        'AUROC': [gcn_auroc],
        'Std': [gcn_std],
        'Accuracy': [gcn_acc],
        'Best_Fold_AUROC': [best_auroc]
    })
    results.to_csv(os.path.join(OUT_DIR, "gcn_results_optimized.csv"), index=False)
    
    print(f"\n{'='*60}")
    print("=== GCN TRAINING COMPLETE ===")
    print(f"{'='*60}")
    print(f"Mean AUROC: {gcn_auroc:.4f} (+/- {gcn_std:.4f})")
    print(f"Mean Accuracy: {gcn_acc*100:.1f}%")
    print(f"Best Fold AUROC: {best_auroc:.4f}")
    print(f"\nModel weights saved to: {MODEL_DIR}/")
    print(f"  - gcn_fold1.pt through gcn_fold5.pt")
    print(f"  - gcn_best.pt (overall best)")

if __name__ == "__main__":
    main()
