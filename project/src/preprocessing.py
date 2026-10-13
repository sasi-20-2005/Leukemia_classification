"""
preprocessing.py
-----------------
Implements Proposal Section 10 (Module 2 - Data Preprocessing, Module 3 - Class Balancing)
and Existing-System Stages 1-2 (Data Cleaning, Normalization), reused identically in the
proposed pipeline per Section 16 ("Existing vs Proposed System": Preprocessing / Imbalance
Handling rows are unchanged).

Stages implemented, in order:
  1. Data Cleaning        -> handle missing values, drop zero-variance / redundant genes
  2. Normalization         -> Min-Max scaling to [0, 1] (matches base paper)
  3. Initial Feature Reduction -> variance/ANOVA F-score ranking to a manageable gene count
                                   before the CapsNet/GNN branches (keeps the pipeline tractable,
                                   analogous in spirit to BO-CLS's dimensionality reduction goal,
                                   Section 6.1 / 6.3)
  4. Train/test split
  5. SMOTE                 -> synthetic minority oversampling on the TRAINING split only
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def clean_data(X: pd.DataFrame) -> pd.DataFrame:
    """Stage 1: Data Cleaning.
    - impute missing values with the column (gene) median
    - drop zero-variance (constant / uninformative) genes
    - drop exact duplicate gene columns (redundant features)
    """
    X = X.copy()
    X = X.fillna(X.median(numeric_only=True))
    # any residual NaN columns (all-NaN) -> drop
    X = X.dropna(axis=1, how="all")
    X = X.fillna(0.0)

    variances = X.var(axis=0)
    nonconstant_cols = variances[variances > 1e-12].index
    X = X[nonconstant_cols]

    X = X.loc[:, ~X.T.duplicated()]
    return X


def normalize_data(X_train: pd.DataFrame, X_test: pd.DataFrame | None = None):
    """Stage 2: Normalization (Min-Max scaling, fit on train only to avoid leakage)."""
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler


def reduce_features(X_train: pd.DataFrame, y_train_enc: np.ndarray, X_test: pd.DataFrame,
                     top_k: int = 256):
    """Initial Feature Reduction: rank genes by ANOVA F-score (computed on TRAIN only)
    and keep the top_k most discriminative genes. Reduces the 20k+ gene dimensionality
    down to a size the CapsNet/GNN branches can operate on (Section 6.1 motivation).
    """
    top_k = min(top_k, X_train.shape[1])
    f_scores, _ = f_classif(X_train.values, y_train_enc)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    top_idx = np.argsort(f_scores)[::-1][:top_k]
    selected_genes = X_train.columns[top_idx]
    return X_train[selected_genes], X_test[selected_genes], list(selected_genes)


def smote_oversample(X: np.ndarray, y: np.ndarray, k_neighbors: int = 5,
                      random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Stage 3: SMOTE (Synthetic Minority Over-sampling Technique), implemented from
    scratch with scikit-learn's NearestNeighbors (no external imblearn dependency needed).

    For every minority class, synthetic samples are generated along the line segment
    between a real minority sample and one of its k nearest minority-class neighbours,
    until every class matches the size of the majority class.
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_res = [X]
    y_res = [y]

    for cls, cnt in zip(classes, counts):
        n_to_generate = max_count - cnt
        if n_to_generate <= 0:
            continue
        X_cls = X[y == cls]
        k = min(k_neighbors, max(1, len(X_cls) - 1))
        if k < 1:
            # only one sample in this class -> duplicate it
            synthetic = np.repeat(X_cls, n_to_generate, axis=0)[:n_to_generate]
            X_res.append(synthetic)
            y_res.append(np.full(n_to_generate, cls))
            continue

        nn = NearestNeighbors(n_neighbors=k + 1).fit(X_cls)
        _, neighbors = nn.kneighbors(X_cls)

        synthetic = np.zeros((n_to_generate, X.shape[1]))
        for i in range(n_to_generate):
            sample_idx = rng.integers(0, len(X_cls))
            neighbor_choices = neighbors[sample_idx][1:]  # exclude self
            neighbor_idx = rng.choice(neighbor_choices)
            gap = rng.random()
            synthetic[i] = X_cls[sample_idx] + gap * (X_cls[neighbor_idx] - X_cls[sample_idx])

        X_res.append(synthetic)
        y_res.append(np.full(n_to_generate, cls))

    X_out = np.vstack(X_res)
    y_out = np.concatenate(y_res)
    perm = rng.permutation(len(y_out))
    return X_out[perm], y_out[perm]


def full_preprocessing_pipeline(X_raw: pd.DataFrame, y_raw: pd.Series, top_k_genes: int = 256,
                                 test_size: float = 0.25, random_state: int = 42):
    """Runs Stages 1-5 end to end and returns everything downstream modules need."""
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y_raw.values)

    # Stage 1: Data Cleaning
    X_clean = clean_data(X_raw)

    # Split BEFORE fitting normalization / feature selection / SMOTE (avoid data leakage)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_clean, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    # Stage 2: Normalization
    X_train_norm, X_test_norm, scaler = normalize_data(X_train_raw, X_test_raw)

    # Initial Feature Reduction
    X_train_sel, X_test_sel, selected_genes = reduce_features(
        X_train_norm, y_train, X_test_norm, top_k=top_k_genes
    )

    # Stage 3: SMOTE (training data only)
    X_train_bal, y_train_bal = smote_oversample(X_train_sel.values, y_train)

    return {
        "X_train": X_train_bal,
        "y_train": y_train_bal,
        "X_test": X_test_sel.values,
        "y_test": y_test,
        "selected_genes": selected_genes,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "class_counts_before_smote": dict(zip(*np.unique(y_train, return_counts=True))),
        "class_counts_after_smote": dict(zip(*np.unique(y_train_bal, return_counts=True))),
    }
