# Hybrid Capsule Network + GNN — Multi-Class Leukemia Classification

Full implementation of the proposal *"Hybrid Capsule Network and Graph Neural Network
Framework for Multi-Class Leukemia Classification Using Gene Expression Data"*, benchmarked
conceptually against the base paper's Optimized Dueling Double Deep Q-Network (DDDQ-N).

```
Gene Expression Input
        |
  Data Cleaning -> Normalization -> SMOTE -> Feature Reduction
        |
   +----+----------------------------+
   |                                 |
CAPSULE NETWORK                  GENE GRAPH -> GRAPH NEURAL NETWORK
   |                                 |
Hierarchical Gene Features    Relational Gene Features
   |                                 |
   +----------------+----------------+
                     |
              FEATURE FUSION
                     |
        FULLY CONNECTED -> SOFTMAX
                     |
        Multi-Class Leukemia Prediction
```

## What's in this project

| Path | Proposal section(s) it implements |
|---|---|
| `src/data_utils.py` | §10 Module 1 — data acquisition (+ synthetic CuMiDa-like generator) |
| `src/preprocessing.py` | §10 Modules 2-3 — cleaning, normalization, feature reduction, SMOTE |
| `src/graph_construction.py` | §12 — gene relationship graph G=(V,E) |
| `src/nn_core.py`, `src/hybrid_model.py` | §9, §11-15 — Capsule Network, GNN, fusion, softmax |
| `src/metrics.py` | §19 — accuracy, precision, recall, F1, ROC-AUC, confusion matrix, k-fold CV |
| `train.py` | §21 — full CLI training/evaluation workflow, saves plots + `model.pkl` |
| `app/` | Web UI (Flask) — Home / Register / Login (SQLite) + the pipeline dashboard |
| `notebook/Hybrid_CapsNet_GNN_Leukemia.ipynb` | §20 — Jupyter/Colab notebook version, same pipeline |
| `torch_reference/` | §20 — PyTorch + PyTorch Geometric version of the model (for GPU/Colab training on the real dataset) |
| `sample_data/synthetic_leukemia_demo.csv` | A ready-to-upload demo CSV for the UI |

## Quick start — the Web UI (recommended)

```bash
cd project
pip install -r requirements.txt
python app/app.py
```
Open **http://127.0.0.1:5000** — you'll land on the **Home** page. Click **Create account**
(username, email, password — stored in a local SQLite file `app/users.db`, password
hashed with Werkzeug's salted hash, never stored in plain text), then **Log in** with your
email + password. The dashboard (`/dashboard`) and all pipeline/prediction endpoints
require being logged in; logging out returns you to the Home page.

Once logged in, the dashboard UI lets you:
1. Pick the built-in **synthetic demo dataset** or **upload your own CSV** (rows = samples,
   columns = genes, one label column — matches the CuMiDa export format).
2. Configure preprocessing/model settings: genes kept after feature reduction, train/test
   split, gene-graph correlation threshold, epochs, batch size.
3. Click **Run Pipeline** and watch the architecture diagram highlight each stage while
   it trains.
4. Review Accuracy / Precision / Recall / F1 / ROC-AUC, the confusion matrix, the learned
   gene graph, the SMOTE before/after class balance, and a live training curve.
5. **Predict** individual held-out samples — output matches the exact format specified in
   the proposal (Section 22): Sample ID, Predicted Class, per-class probabilities.

## Quick start — command line

```bash
pip install -r requirements.txt
python train.py --synthetic --epochs 30          # built-in demo dataset
python train.py --csv path/to/cumida.csv --label-col class --epochs 30   # real dataset
```
Writes `outputs/metrics.json`, `outputs/model.pkl`, `outputs/confusion_matrix.png`,
`outputs/training_curve.png`, `outputs/gene_graph.png`.

## Quick start — Jupyter / Colab

Open `notebook/Hybrid_CapsNet_GNN_Leukemia.ipynb` (keep it inside `notebook/`, next to
`../src`). It runs with only `numpy, pandas, scikit-learn, networkx, matplotlib` — all
preinstalled on Colab, no GPU or extra installs needed. Each section is annotated with the
proposal section it implements, ending with the same evaluation + single-sample prediction
format as the UI.

## Using the real CuMiDa dataset

The base paper's dataset, *Leukemia Gene Expression — CuMiDa*, must be downloaded manually
(license/terms require it) from **https://sbcb.inf.ufrgs.br/cumida**. Export/format it as a
CSV with samples as rows, genes as columns, and one label column (`class`), then either:
- upload it in the Web UI, or
- pass `--csv your_file.csv --label-col class` to `train.py` / `train_torch.py`, or
- set `CSV_PATH` in the notebook.

Without it, every entry point falls back to a **synthetic** dataset shaped like CuMiDa
(64 samples, thousands of genes, 5 classes: `AML, BM, BM_CD34, PB, PBSC_CD34`, with
injected class imbalance and noise) so the full pipeline can still be exercised and
demonstrated end to end. **Do not report the synthetic-data results as leukemia
classification performance** — they only demonstrate that the pipeline runs correctly.

## Two model implementations, one architecture

Because this delivery environment doesn't have PyTorch/TensorFlow/PyTorch Geometric
installed (and can't install them without network access), the project ships two
implementations of the *same* architecture:

1. **`src/hybrid_model.py`** — a from-scratch NumPy engine (Capsule Network with dynamic
   routing + a custom graph-convolution GNN, manually back-propagated). This is what
   powers the Web UI and `train.py`, runs anywhere with just NumPy/scikit-learn, and has
   been gradient-checked and tested end-to-end (see "Implementation notes" below).
2. **`torch_reference/model_torch.py` + `train_torch.py`** — a PyTorch + PyTorch Geometric
   version matching the exact tool stack requested in Proposal Section 20, intended for
   real training runs on a GPU/Colab against the actual CuMiDa dataset. Install
   `torch_reference/requirements_torch.txt` to use it.

Both follow the identical architecture: PrimaryCaps → dynamic routing → class capsules
(hierarchical features) in parallel with a 2-layer GCN over the gene graph → mean-pool
readout (relational features) → concatenation → FC → softmax.

## Implementation notes (read before citing results)

- **Capsule routing backprop**: gradients through the dynamic-routing coupling
  coefficients are treated as constants during the backward pass in the NumPy engine —
  a standard simplification in from-scratch CapsNet implementations. Gradient-checking
  confirms all other parameters match analytic finite-difference gradients closely; the
  routing-transform weights (`Wr`) show a larger (but bounded, ~15-25%) discrepancy from
  this simplification. This does not prevent the model from training (see the notebook's
  training curve), but it means the capsule branch is a weaker learner than a full
  autodiff implementation (e.g. `torch_reference/`) would give you.
- **Feature reduction**: an ANOVA F-score filter (not BO-CLS) is used to cut gene count
  down to a tractable size before the CapsNet/GNN branches, matching the proposal's
  motivation (Section 6.1) without depending on a metaheuristic optimizer.
- **Per-sample training loop**: the NumPy engine loops over samples within each batch in
  plain Python rather than batching linear algebra, which is fine at CuMiDa's scale
  (dozens to a few hundred samples) but would need vectorizing (or just use
  `torch_reference/`) for larger cohorts.
- Per Section 18 of the proposal: **improved accuracy over the DDDQ-N base paper is not
  claimed**. Compare both systems on the identical real dataset split before drawing any
  conclusion.

## Evaluation metrics implemented (Section 19)

Accuracy, macro Precision, macro Recall, macro F1, macro ROC-AUC (one-vs-rest), full
confusion matrix, per-class precision/recall/F1 report, and stratified k-fold
cross-validation (recommended given the CuMiDa dataset's small sample count).

## Tools & technologies

| Proposal (Section 20) | This delivery |
|---|---|
| Python 3.x | ✅ |
| Jupyter Notebook / Google Colab | ✅ `notebook/` |
| TensorFlow/Keras or PyTorch | ✅ `torch_reference/` (PyTorch); NumPy engine used where torch isn't available |
| Scikit-learn | ✅ (feature selection, metrics, splitting, kNN for SMOTE) |
| PyTorch Geometric | ✅ `torch_reference/model_torch.py` (`GCNConv`, `global_mean_pool`) |
| NumPy & Pandas | ✅ |
| Matplotlib | ✅ (`train.py`, notebook, and the UI's server-rendered charts) |

## Project layout

```
project/
├── src/
│   ├── data_utils.py          # data loading + synthetic CuMiDa-like generator
│   ├── preprocessing.py       # cleaning, normalization, feature reduction, SMOTE
│   ├── graph_construction.py  # gene relationship graph
│   ├── nn_core.py             # squash, softmax, relu, Adam optimizer (NumPy)
│   ├── hybrid_model.py        # CapsNet + GNN + fusion + softmax (NumPy, trainable)
│   └── metrics.py             # evaluation metrics + cross-validation
├── app/
│   ├── app.py                 # Flask backend / API + auth routes
│   ├── db.py                  # SQLite user store (register/login)
│   ├── users.db                # created automatically on first run
│   ├── static/theme.css       # shared dark navy/teal theme
│   └── templates/
│       ├── home.html          # landing page
│       ├── login.html         # email + password
│       ├── register.html      # username + email + password
│       └── index.html         # the pipeline dashboard (Chart.js training curve)
├── torch_reference/
│   ├── model_torch.py         # PyTorch + PyTorch Geometric model
│   ├── train_torch.py         # training script for the real dataset / GPU
│   └── requirements_torch.txt
├── notebook/
│   └── Hybrid_CapsNet_GNN_Leukemia.ipynb
├── sample_data/
│   └── synthetic_leukemia_demo.csv
├── train.py                   # CLI training entry point
└── requirements.txt
```



https://sbcb.inf.ufrgs.br/cumida  ----> use this site for test data