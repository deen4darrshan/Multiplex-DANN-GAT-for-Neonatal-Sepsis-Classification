# Model Architecture

Primary model
OIGATv2, a GAT style model using TransformerConv layers with residual connections and pooled graph level classification.

Layer structure
- TransformerConv layer 1 with multi head attention, followed by LayerNorm and residual projection.
- TransformerConv layer 2 with multi head attention, followed by LayerNorm and residual addition.
- TransformerConv layer 3 with single head attention, followed by LayerNorm.
- Global mean pooling and global max pooling are concatenated.
- Two fully connected layers map pooled features to two class logits.

Activation and regularization
- LeakyReLU is used after each convolution and hidden layer.
- Dropout is applied before the final linear layer.
- Gradient clipping is applied during training.

Baseline model
OIGCN uses three GCNConv layers with a similar pooling and MLP head for comparison.

Math sketch
- Attention message passing: h_i' = sum_j alpha_ij W h_j
- Global pooling: h_graph = concat(mean(h_i), max(h_i))
- Class logits: y = W2 * relu(W1 * h_graph)

Key scripts
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\05_train_gnn.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\08_run_real_external_eval.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\11_human_grouped5_eval.py`
