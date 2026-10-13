"""
data_utils.py
-------------
Module 1: Gene Expression Data Acquisition (Proposal Section 10, Module 1)

Provides:
  - load_csv_dataset(path): load a real gene-expression CSV (samples x genes + label column)
  - generate_synthetic_cumida_like(...): generates a synthetic dataset that mimics the
    shape/characteristics of the CuMiDa Leukemia Gene Expression dataset described in the
    base paper (Section 4): 64 samples, thousands of features, 5 classes. This is used only
    when the user has not uploaded the real CuMiDa dataset (e.g. running the demo offline),
    so the rest of the pipeline (preprocessing -> CapsNet -> GNN -> fusion -> classification)
    can be exercised end-to-end.

CuMiDa itself must be downloaded by the user from https://sbcb.inf.ufrgs.br/cumida
(license/terms require manual download), so it is not bundled here.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CLASS_NAMES = ["BM_CD34", "BM", "AML", "PB", "PBSC_CD34"]


def generate_synthetic_cumida_like(
    n_samples: int = 64,
    n_genes: int = 2000,
    n_classes: int = 5,
    class_names=None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Creates a synthetic microarray-like gene expression dataset:
      - n_genes columns of continuous "expression" values
      - each class has a distinct small set of "marker genes" with a mean shift,
        so the classification problem is learnable but noisy (mirrors real microarray data)
      - class sizes are imbalanced on purpose (to exercise the SMOTE stage)

    Returns
    -------
    X : DataFrame (n_samples, n_genes), columns = gene_0001, gene_0002, ...
    y : Series of class labels (strings)
    """
    rng = np.random.default_rng(random_state)
    class_names = class_names or CLASS_NAMES[:n_classes]

    # Deliberately imbalanced class sizes (like real clinical cohorts)
    base = n_samples // n_classes
    sizes = [base] * n_classes
    sizes[0] += n_samples - sum(sizes)
    # skew: make one class much bigger, one much smaller
    sizes = sorted(sizes)
    sizes[-1] += 6
    sizes[0] = max(3, sizes[0] - 6)

    X_rows = []
    y_rows = []
    genes_per_class_signal = 30

    # background expression: log-normal-ish, plus a few NaN / noisy columns to exercise cleaning
    for cls_idx, (cname, n) in enumerate(zip(class_names, sizes)):
        base_expr = rng.normal(loc=8.0, scale=1.5, size=(n, n_genes))
        # inject a class-specific marker-gene signature
        marker_idx = rng.choice(n_genes, size=genes_per_class_signal, replace=False)
        shift = rng.normal(loc=3.0, scale=0.5, size=genes_per_class_signal)
        base_expr[:, marker_idx] += shift
        X_rows.append(base_expr)
        y_rows += [cname] * n

    X = np.vstack(X_rows)

    # shuffle samples
    perm = rng.permutation(X.shape[0])
    X = X[perm]
    y = np.array(y_rows)[perm]

    # inject some missing values and a few constant/irrelevant columns (Stage 1: Data Cleaning target)
    n_missing = int(0.001 * X.size)
    flat_idx = rng.choice(X.size, size=n_missing, replace=False)
    X.ravel()[flat_idx] = np.nan
    constant_cols = rng.choice(n_genes, size=max(1, n_genes // 200), replace=False)
    X[:, constant_cols] = 5.0  # zero-variance / uninformative genes

    gene_cols = [f"gene_{i:05d}" for i in range(n_genes)]
    X_df = pd.DataFrame(X, columns=gene_cols)
    y_s = pd.Series(y, name="class")
    return X_df, y_s


def load_csv_dataset(path: str, label_col: str = "class") -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads a real gene-expression dataset from CSV.
    Expected layout: rows = samples, columns = genes, one column = class label.
    (This matches the CuMiDa distribution format after transposition.)
    """
    df = pd.read_csv(path)
    if label_col not in df.columns:
        # try common alternatives used by CuMiDa exports
        for alt in ["type", "Class", "Type", "label", "Label"]:
            if alt in df.columns:
                label_col = alt
                break
        else:
            raise ValueError(
                f"Could not find a label column in {path}. "
                f"Expected one of: class/type/label. Columns found: {list(df.columns)[:10]}..."
            )
    y = df[label_col]
    X = df.drop(columns=[label_col])
    # drop non-numeric identifier columns (e.g. "sample_id")
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)
    return X, y
