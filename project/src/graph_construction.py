"""
graph_construction.py
----------------------
Proposal Section 12: Graph Construction.

Builds a gene-relationship graph G = (V, E):
  - V: one node per selected gene
  - E: an edge between two genes if the absolute Pearson correlation between their
       expression profiles (across training samples) exceeds a threshold

Returns a normalized adjacency matrix (with self-loops), suitable for a GCN-style
propagation rule: A_norm = D^(-1/2) (A + I) D^(-1/2)   (Kipf & Welling, 2017)
"""
from __future__ import annotations
import numpy as np
import networkx as nx


def build_gene_graph(X_train: np.ndarray, correlation_threshold: float = 0.6,
                      max_edges_per_node: int = 15) -> tuple[np.ndarray, nx.Graph]:
    """
    Parameters
    ----------
    X_train : (n_samples, n_genes) array (already cleaned/normalized/feature-reduced)
    correlation_threshold : minimum |correlation| to draw an edge
    max_edges_per_node : cap to keep the graph sparse for small sample counts

    Returns
    -------
    A_norm : (n_genes, n_genes) symmetric-normalized adjacency (with self loops)
    G : networkx.Graph, for visualization / inspection
    """
    n_genes = X_train.shape[1]
    # correlation across genes -> need genes as variables (columns), samples as observations
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(X_train, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)

    A = np.zeros((n_genes, n_genes))
    G = nx.Graph()
    G.add_nodes_from(range(n_genes))

    abs_corr = np.abs(corr)
    for i in range(n_genes):
        # connect to top correlated genes above threshold (keeps graph sparse & connected)
        candidates = np.argsort(abs_corr[i])[::-1]
        added = 0
        for j in candidates:
            if j == i or added >= max_edges_per_node:
                break
            if abs_corr[i, j] >= correlation_threshold:
                A[i, j] = A[j, i] = 1.0
                G.add_edge(i, j, weight=float(abs_corr[i, j]))
                added += 1

    # Fallback: if the threshold left isolated nodes / an empty graph (common with few
    # samples), connect every node to its single most-correlated neighbour.
    isolated = np.where(A.sum(axis=1) == 0)[0]
    for i in isolated:
        j = np.argsort(abs_corr[i])[::-1][0]
        if j != i:
            A[i, j] = A[j, i] = 1.0
            G.add_edge(i, j, weight=float(abs_corr[i, j]))

    A_self = A + np.eye(n_genes)
    deg = A_self.sum(axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A_self @ D_inv_sqrt
    return A_norm, G
