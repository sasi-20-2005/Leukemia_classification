"""
train_torch.py
---------------
Reference training script for the PyTorch + PyTorch Geometric HybridCapsGNN
(Proposal Section 20/21). Reuses the already-tested preprocessing, SMOTE, and gene-graph
construction from ../src (pure NumPy / scikit-learn / networkx - no torch needed there),
and only swaps in the PyTorch model for training on a GPU/Colab.

Usage (after `pip install -r requirements_torch.txt`):
    python train_torch.py --synthetic --epochs 60
    python train_torch.py --csv /path/to/cumida_leukemia.csv --label-col class --epochs 60
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_utils import generate_synthetic_cumida_like, load_csv_dataset
from src.preprocessing import full_preprocessing_pipeline
from src.graph_construction import build_gene_graph
from src.metrics import compute_all_metrics
from model_torch import HybridCapsGNN, margin_loss


class GeneDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def adjacency_to_edge_index(A_norm: np.ndarray):
    """Convert the dense normalized adjacency from src.graph_construction into
    (edge_index, edge_weight) tensors for a single graph template, to be repeated
    per-sample via torch_geometric's batching convention."""
    rows, cols = np.nonzero(A_norm)
    weights = A_norm[rows, cols]
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def make_batched_graph(batch_size: int, n_genes: int, edge_index, edge_weight, device):
    """Repeats the fixed gene graph once per sample in the batch and offsets node indices,
    matching what torch_geometric.data.Batch would produce for `batch_size` copies of the
    same graph. Returns (batched_edge_index, batched_edge_weight, batch_vector)."""
    E = edge_index.size(1)
    offsets = (torch.arange(batch_size, device=device) * n_genes).repeat_interleave(E)
    rep_edge_index = edge_index.to(device).repeat(1, batch_size) + offsets.unsqueeze(0)
    rep_edge_weight = edge_weight.to(device).repeat(batch_size)
    batch_vec = torch.arange(batch_size, device=device).repeat_interleave(n_genes)
    return rep_edge_index, rep_edge_weight, batch_vec


def run_epoch(model, loader, edge_index, edge_weight, n_genes, device, optimizer=None,
              lambda_margin: float = 0.2):
    training = optimizer is not None
    model.train(training)
    total_loss, n_seen, correct = 0.0, 0, 0
    all_probs, all_true = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        b = xb.size(0)
        b_edge_index, b_edge_weight, b_vec = make_batched_graph(b, n_genes, edge_index, edge_weight, device)

        if training:
            optimizer.zero_grad()
        logits, cap_lengths = model(xb, b_edge_index, b_edge_weight, b_vec)
        ce = F.cross_entropy(logits, yb)
        loss = ce + lambda_margin * margin_loss(cap_lengths, yb)
        if training:
            loss.backward()
            optimizer.step()

        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        all_probs.append(probs)
        all_true.append(yb.cpu().numpy())
        total_loss += loss.item() * b
        n_seen += b
        correct += (probs.argmax(axis=1) == yb.cpu().numpy()).sum()

    return {
        "loss": total_loss / n_seen,
        "accuracy": correct / n_seen,
        "y_true": np.concatenate(all_true),
        "y_proba": np.concatenate(all_probs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--label-col", type=str, default="class")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--top-k-genes", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--correlation-threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    if args.csv:
        X_raw, y_raw = load_csv_dataset(args.csv, label_col=args.label_col)
    else:
        X_raw, y_raw = generate_synthetic_cumida_like(random_state=args.seed)

    prep = full_preprocessing_pipeline(X_raw, y_raw, top_k_genes=args.top_k_genes,
                                        random_state=args.seed)
    class_names = list(prep["label_encoder"].classes_)
    n_genes = prep["X_train"].shape[1]

    A_norm, _ = build_gene_graph(prep["X_train"], correlation_threshold=args.correlation_threshold)
    edge_index, edge_weight = adjacency_to_edge_index(A_norm)

    train_loader = DataLoader(GeneDataset(prep["X_train"], prep["y_train"]),
                               batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(GeneDataset(prep["X_test"], prep["y_test"]),
                              batch_size=args.batch_size, shuffle=False)

    model = HybridCapsGNN(n_genes=n_genes, n_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        tr = run_epoch(model, train_loader, edge_index, edge_weight, n_genes, device, optimizer)
        te = run_epoch(model, test_loader, edge_index, edge_weight, n_genes, device, optimizer=None)
        print(f"epoch {epoch+1:>3}/{args.epochs}  train_loss={tr['loss']:.4f} "
              f"train_acc={tr['accuracy']:.3f}  test_acc={te['accuracy']:.3f}")

    te = run_epoch(model, test_loader, edge_index, edge_weight, n_genes, device, optimizer=None)
    preds = te["y_proba"].argmax(axis=1)
    metrics = compute_all_metrics(te["y_true"], preds, te["y_proba"], class_names)
    print("\nFinal test metrics:")
    for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"]:
        print(f"  {k}: {metrics[k]}")


if __name__ == "__main__":
    main()
