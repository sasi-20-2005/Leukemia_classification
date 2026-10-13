"""
app.py
------
Flask web UI for the Hybrid Capsule Network + GNN Leukemia Classification project.

Run:
    cd project
    pip install -r requirements.txt
    python app/app.py
Then open http://127.0.0.1:5000 in a browser.

The UI lets you:
  1. Pick the synthetic CuMiDa-like demo dataset OR upload your own gene-expression CSV
  2. Configure preprocessing / model hyperparameters
  3. Run the full pipeline (clean -> normalize -> SMOTE -> feature reduction ->
     CapsNet + GNN -> fusion -> softmax) and watch training progress
  4. View accuracy/precision/recall/F1/ROC-AUC, the confusion matrix, the gene graph,
     and the SMOTE class-balance effect
  5. Run predictions on individual test samples in the exact output format specified
     in the proposal (Section 22)
"""
from __future__ import annotations
import base64
import functools
import io
import json
import os
import re
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from flask import (Flask, jsonify, render_template, request, redirect,
                    url_for, session, flash)
from werkzeug.security import generate_password_hash, check_password_hash

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_utils import generate_synthetic_cumida_like, CLASS_NAMES
from src.preprocessing import full_preprocessing_pipeline
from src.graph_construction import build_gene_graph
from src.hybrid_model import HybridCapsGNN
from src.metrics import compute_all_metrics
from db import init_db, get_user_by_email, get_user_by_id, create_user

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB upload cap
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-this-secret-key")

init_db()

STATE = {}  # in-memory session state for this single-user demo (model, data, class names)
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def login_required(view_fn):
    @functools.wraps(view_fn)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "error")
            return redirect(url_for("login", next=request.path))
        return view_fn(*args, **kwargs)
    return wrapped


@app.context_processor
def inject_user():
    return {"current_username": session.get("username")}


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def make_confusion_matrix_fig(cm, class_names):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=40, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                     color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def make_graph_fig(G, max_nodes=70):
    if G.number_of_nodes() > max_nodes:
        G = G.subgraph(list(G.nodes)[:max_nodes])
    fig, ax = plt.subplots(figsize=(5, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, node_size=50, node_color="#4C72B0", edge_color="#C9CFDB", with_labels=False)
    ax.set_title(f"{G.number_of_nodes()} genes, {G.number_of_edges()} edges shown")
    fig.tight_layout()
    return fig


def make_class_balance_fig(before: dict, after: dict, class_names):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    idx = list(range(len(class_names)))
    before_vals = [before.get(i, before.get(str(i), 0)) for i in idx]
    after_vals = [after.get(i, after.get(str(i), 0)) for i in idx]
    width = 0.35
    ax.bar([i - width / 2 for i in idx], before_vals, width, label="Before SMOTE", color="#E07A5F")
    ax.bar([i + width / 2 for i in idx], after_vals, width, label="After SMOTE", color="#3D5A80")
    ax.set_xticks(idx); ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Training samples")
    ax.legend()
    fig.tight_layout()
    return fig


@app.route("/")
def root():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("home"))


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""

        error = None
        if not username or len(username) < 2:
            error = "Please enter a username (at least 2 characters)."
        elif not EMAIL_RE.match(email):
            error = "Please enter a valid email address."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        elif password != confirm:
            error = "Passwords do not match."
        elif get_user_by_email(email) is not None:
            error = "An account with that email already exists."

        if error:
            flash(error, "error")
            return render_template("register.html", username=username, email=email)

        create_user(username, email, generate_password_hash(password))
        flash("Account created — please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        next_url = request.form.get("next") or url_for("dashboard")

        user = get_user_by_email(email)
        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Incorrect email or password.", "error")
            return render_template("login.html", email=email, next=next_url)

        session.clear()
        session["user_id"] = user["id"]
        session["username"] = user["username"]
        flash(f"Welcome back, {user['username']}!", "success")
        return redirect(next_url)

    next_url = request.args.get("next") or url_for("dashboard")
    return render_template("login.html", next=next_url)


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
@login_required
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file provided"}), 400
    path = os.path.join(UPLOAD_DIR, "dataset.csv")
    f.save(path)
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {e}"}), 400
    STATE["uploaded_csv_path"] = path
    return jsonify({"columns": list(df.columns), "preview": df.to_dict(orient="records")})


@app.route("/api/run", methods=["POST"])
@login_required
def run_pipeline():
    cfg = request.get_json(force=True)
    try:
        use_synthetic = cfg.get("use_synthetic", True)
        top_k = int(cfg.get("top_k_genes", 128))
        epochs = int(cfg.get("epochs", 25))
        batch_size = int(cfg.get("batch_size", 8))
        corr_threshold = float(cfg.get("correlation_threshold", 0.5))
        test_size = float(cfg.get("test_size", 0.25))
        seed = int(cfg.get("seed", 42))

        if use_synthetic or "uploaded_csv_path" not in STATE:
            X_raw, y_raw = generate_synthetic_cumida_like(random_state=seed)
            dataset_source = "synthetic_cumida_like_demo"
        else:
            df = pd.read_csv(STATE["uploaded_csv_path"])
            label_col = cfg.get("label_col") or "class"
            if label_col not in df.columns:
                return jsonify({"error": f"Label column '{label_col}' not found in uploaded CSV."}), 400
            y_raw = df[label_col]
            X_raw = df.drop(columns=[label_col])
            X_raw = X_raw.select_dtypes(include=[np.number])
            dataset_source = "uploaded_csv"

        prep = full_preprocessing_pipeline(X_raw, y_raw, top_k_genes=top_k,
                                            test_size=test_size, random_state=seed)
        class_names = list(prep["label_encoder"].classes_)

        A_norm, G = build_gene_graph(prep["X_train"], correlation_threshold=corr_threshold)

        model = HybridCapsGNN(n_genes=prep["X_train"].shape[1], n_classes=len(class_names),
                               A_norm=A_norm, seed=seed)

        rng = np.random.default_rng(seed)
        losses, test_accs = [], []
        for _ in range(epochs):
            loss = model.train_epoch(prep["X_train"], prep["y_train"], batch_size=batch_size, rng=rng)
            preds = model.predict(prep["X_test"])
            acc = float((preds == prep["y_test"]).mean())
            losses.append(loss); test_accs.append(acc)

        proba = model.predict_proba(prep["X_test"])
        preds = np.argmax(proba, axis=1)
        metrics = compute_all_metrics(prep["y_test"], preds, proba, class_names)

        cm_fig = make_confusion_matrix_fig(metrics["confusion_matrix"], class_names)
        graph_fig = make_graph_fig(G)
        balance_fig = make_class_balance_fig(prep["class_counts_before_smote"],
                                              prep["class_counts_after_smote"], class_names)

        STATE.update({
            "model": model, "prep": prep, "class_names": class_names,
            "A_norm": A_norm,
        })

        return jsonify({
            "dataset_source": dataset_source,
            "n_train": int(len(prep["y_train"])), "n_test": int(len(prep["y_test"])),
            "n_genes_selected": int(prep["X_train"].shape[1]),
            "class_names": class_names,
            "graph_edges": int(G.number_of_edges()), "graph_nodes": int(G.number_of_nodes()),
            "metrics": {k: v for k, v in metrics.items()
                        if k not in ("confusion_matrix", "per_class_report")},
            "per_class_report": metrics["per_class_report"],
            "confusion_matrix": metrics["confusion_matrix"],
            "training_curve": {"loss": losses, "test_accuracy": test_accs},
            "confusion_matrix_img": fig_to_base64(cm_fig),
            "gene_graph_img": fig_to_base64(graph_fig),
            "class_balance_img": fig_to_base64(balance_fig),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
@login_required
def predict():
    if "model" not in STATE:
        return jsonify({"error": "Train a model first (click 'Run Pipeline')."}), 400
    body = request.get_json(force=True) or {}
    prep = STATE["prep"]
    model = STATE["model"]
    class_names = STATE["class_names"]

    idx = body.get("test_index")
    if idx is None:
        idx = int(np.random.default_rng().integers(0, len(prep["y_test"])))
    idx = int(idx) % len(prep["y_test"])

    x = prep["X_test"][idx]
    true_label = class_names[int(prep["y_test"][idx])]
    probs, _ = model.forward_single(x)
    pred_idx = int(np.argmax(probs))

    return jsonify({
        "sample_id": f"S-{idx:03d}",
        "true_class": true_label,
        "predicted_class": class_names[pred_idx],
        "class_probabilities": {cn: float(p) for cn, p in zip(class_names, probs)},
        "correct": bool(class_names[pred_idx] == true_label),
        "n_test_samples": int(len(prep["y_test"])),
    })


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode, port=int(os.environ.get("PORT", 5000)), use_reloader=False)
