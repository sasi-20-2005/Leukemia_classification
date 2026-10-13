"""
train.py
--------
Command-line entry point implementing the full Proposed Workflow (Section 21):

    Load Data -> Clean -> Normalize -> Split -> SMOTE -> Feature Reduction
        -> [Capsule Network | Gene Graph -> GNN] -> Feature Fusion
        -> Fully Connected -> Softmax -> Prediction -> Performance Evaluation

Usage
-----
    # with your own real dataset (CSV: samples x genes, one label column)
    python train.py --csv path/to/cumida_leukemia.csv --label-col class --epochs 40

    # with the built-in synthetic CuMiDa-like demo dataset (no file needed)
    python train.py --synthetic --epochs 40

Outputs (written to ./outputs/):
    model.pkl, metrics.json, confusion_matrix.png, training_curve.png, gene_graph.png
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.data_utils import generate_synthetic_cumida_like, load_csv_dataset
from src.preprocessing import full_preprocessing_pipeline
from src.graph_construction import build_gene_graph
from src.hybrid_model import HybridCapsGNN
from src.metrics import compute_all_metrics, cross_validate_summary


def plot_confusion_matrix(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                     color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curve(losses, accs, path):
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(losses, color="tab:red", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(accs, color="tab:blue", label="test accuracy")
    ax2.set_ylabel("test accuracy", color="tab:blue")
    fig.suptitle("Training Curve")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gene_graph(G, path, max_nodes=80):
    fig, ax = plt.subplots(figsize=(6, 6))
    if G.number_of_nodes() > max_nodes:
        sub_nodes = list(G.nodes)[:max_nodes]
        G = G.subgraph(sub_nodes)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, node_size=60, node_color="#4C72B0", edge_color="#BBBBBB", with_labels=False)
    ax.set_title(f"Gene Relationship Graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Path to real gene-expression CSV")
    ap.add_argument("--label-col", type=str, default="class")
    ap.add_argument("--synthetic", action="store_true", help="Use built-in synthetic demo dataset")
    ap.add_argument("--top-k-genes", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--correlation-threshold", type=float, default=0.5)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.csv:
        print(f"Loading real dataset from {args.csv} ...")
        X_raw, y_raw = load_csv_dataset(args.csv, label_col=args.label_col)
    else:
        print("No --csv provided: generating synthetic CuMiDa-like demo dataset ...")
        X_raw, y_raw = generate_synthetic_cumida_like(random_state=args.seed)

    print(f"Raw data: {X_raw.shape[0]} samples x {X_raw.shape[1]} genes, "
          f"{y_raw.nunique()} classes: {sorted(y_raw.unique())}")

    print("Running preprocessing pipeline (clean -> normalize -> split -> "
          "feature reduction -> SMOTE)...")
    prep = full_preprocessing_pipeline(X_raw, y_raw, top_k_genes=args.top_k_genes,
                                        random_state=args.seed)
    class_names = list(prep["label_encoder"].classes_)
    print(f"After SMOTE, training class distribution: {prep['class_counts_after_smote']}")

    print("Constructing gene relationship graph ...")
    A_norm, G = build_gene_graph(prep["X_train"], correlation_threshold=args.correlation_threshold)
    plot_gene_graph(G, os.path.join(args.outdir, "gene_graph.png"))

    model = HybridCapsGNN(
        n_genes=prep["X_train"].shape[1], n_classes=len(class_names), A_norm=A_norm, seed=args.seed
    )

    print(f"Training Hybrid CapsNet+GNN for {args.epochs} epochs ...")
    losses, test_accs = [], []
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        loss = model.train_epoch(prep["X_train"], prep["y_train"], batch_size=args.batch_size, rng=rng)
        preds = model.predict(prep["X_test"])
        acc = float((preds == prep["y_test"]).mean())
        losses.append(loss); test_accs.append(acc)
        print(f"  epoch {epoch + 1:>3}/{args.epochs}  loss={loss:.4f}  test_acc={acc:.3f}")
    train_time = time.time() - t0
    plot_training_curve(losses, test_accs, os.path.join(args.outdir, "training_curve.png"))

    print("Evaluating on held-out test set ...")
    proba = model.predict_proba(prep["X_test"])
    preds = np.argmax(proba, axis=1)
    metrics = compute_all_metrics(prep["y_test"], preds, proba, class_names)
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), class_names,
                           os.path.join(args.outdir, "confusion_matrix.png"))

    print(f"Test Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision(macro): {metrics['precision_macro']:.4f}")
    print(f"Recall(macro)   : {metrics['recall_macro']:.4f}")
    print(f"F1(macro)       : {metrics['f1_macro']:.4f}")
    print(f"ROC-AUC(macro)  : {metrics['roc_auc_macro']}")

    if args.cv_folds and args.cv_folds > 1:
        print(f"Running {args.cv_folds}-fold stratified cross-validation "
              "(Section 19: recommended due to small sample size) ...")

        def build_and_train_fn(Xtr, ytr, Xval, yval):
            A_norm_cv, _ = build_gene_graph(Xtr, correlation_threshold=args.correlation_threshold)
            m = HybridCapsGNN(n_genes=Xtr.shape[1], n_classes=len(class_names),
                               A_norm=A_norm_cv, seed=args.seed)
            for _ in range(max(10, args.epochs // 2)):
                m.train_epoch(Xtr, ytr, batch_size=args.batch_size)
            proba_val = m.predict_proba(Xval)
            return yval, np.argmax(proba_val, axis=1), proba_val

        X_all = np.vstack([prep["X_train"], prep["X_test"]])
        y_all = np.concatenate([prep["y_train"], prep["y_test"]])
        cv_summary = cross_validate_summary(build_and_train_fn, X_all, y_all, n_splits=args.cv_folds)
        print(f"CV accuracy: {cv_summary['mean_accuracy']:.4f} +/- {cv_summary['std_accuracy']:.4f}")
        metrics["cross_validation"] = cv_summary

    metrics["train_time_seconds"] = train_time
    metrics["selected_genes"] = prep["selected_genes"]
    metrics["class_names"] = class_names
    metrics["training_curve"] = {"loss": losses, "test_accuracy": test_accs}

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    with open(os.path.join(args.outdir, "model.pkl"), "wb") as f:
        pickle.dump({
            "model": model,
            "label_encoder": prep["label_encoder"],
            "scaler": prep["scaler"],
            "selected_genes": prep["selected_genes"],
        }, f)

    print(f"\nDone. Outputs written to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
