"""
hybrid_model.py
----------------
Proposal Sections 9, 11-15: the Hybrid Capsule Network + Graph Neural Network model.

Architecture (matches the ASCII diagram in Section 9):

    selected gene features (per sample)
              |
      +-------+--------+
      |                |
   CAPSULE NETWORK   GENE GRAPH (fixed, built once from training data - graph_construction.py)
      |                |
      |          GRAPH NEURAL NETWORK
      |                |
  Hierarchical      Relational
  Gene Features     Gene Features
      |                |
      +-------+--------+
              |
        FEATURE FUSION (concatenation)
              |
      FULLY CONNECTED LAYER
              |
           SOFTMAX
              |
   MULTI-CLASS LEUKEMIA CLASSIFICATION

Implementation notes / simplifications (documented for transparency, per Section 18's
guidance not to overclaim before evaluation):
  - Built with NumPy only (no autodiff engine). Backprop through capsule dynamic
    routing treats the routing coefficients as constants for the backward pass
    (a common simplification in from-scratch CapsNet implementations) - gradients
    still flow correctly through the transformation weights and squash non-linearities.
  - The class capsules (post dynamic-routing) are used directly as the "Hierarchical
    Gene Features" fed into fusion; a standard CapsNet margin loss is added as an
    auxiliary term to give the capsule branch direct class-supervision.
  - Per-sample forward/backward is looped in Python (dataset sizes here, following the
    base paper's CuMiDa dataset, are small: tens to a few hundred samples), so this
    is fast enough for CPU training without needing GPU batching.
"""
from __future__ import annotations
import numpy as np
from .nn_core import relu, relu_grad, softmax, squash, squash_backward, Adam


class HybridCapsGNN:
    def __init__(self, n_genes: int, n_classes: int, A_norm: np.ndarray,
                 n_primary_caps: int = 8, primary_dim: int = 8, class_dim: int = 12,
                 gnn_hidden1: int = 8, gnn_hidden2: int = 8, fc_hidden: int = 32,
                 routing_iters: int = 3, lambda_margin: float = 0.2,
                 seed: int = 42):
        self.K = n_genes
        self.C = n_classes
        self.A_norm = A_norm
        self.P = n_primary_caps
        self.D1 = primary_dim
        self.D2 = class_dim
        self.Gh1 = gnn_hidden1
        self.Gh2 = gnn_hidden2
        self.FCh = fc_hidden
        self.routing_iters = routing_iters
        self.lambda_margin = lambda_margin
        self.fusion_dim = self.C * self.D2 + self.Gh2

        rng = np.random.default_rng(seed)

        def xavier(shape):
            fan_in = shape[0] if len(shape) > 1 else 1
            fan_out = shape[-1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=shape)

        self.params = {
            "W1": xavier((self.K, self.P * self.D1)),
            "b1": np.zeros(self.P * self.D1),
            "Wr": xavier((self.P, self.C, self.D2, self.D1)) * 0.5,
            "Wg1": xavier((1, self.Gh1)),
            "bg1": np.zeros(self.Gh1),
            "Wg2": xavier((self.Gh1, self.Gh2)),
            "bg2": np.zeros(self.Gh2),
            "Wf1": xavier((self.fusion_dim, self.FCh)),
            "bf1": np.zeros(self.FCh),
            "Wf2": xavier((self.FCh, self.C)),
            "bf2": np.zeros(self.C),
        }
        self.optimizer = Adam(self.params, lr=1e-3)

    # ---------------------------------------------------------------- forward
    def forward_single(self, x: np.ndarray):
        P, D1, C, D2 = self.P, self.D1, self.C, self.D2
        W1, b1, Wr = self.params["W1"], self.params["b1"], self.params["Wr"]

        # ---- Capsule Network branch ----
        s_p_flat = x @ W1 + b1
        s_p = s_p_flat.reshape(P, D1)
        u = squash(s_p, axis=-1)                       # primary capsules (P, D1)

        u_hat = np.einsum("pcdk,pk->pcd", Wr, u)        # (P, C, D2) prediction vectors

        b_log = np.zeros((P, C))
        c_coef = None
        s_c = None
        v = None
        for r in range(self.routing_iters):
            c_coef = np.apply_along_axis(softmax, 1, b_log)   # (P, C), softmax over classes
            s_c = np.einsum("pc,pcd->cd", c_coef, u_hat)      # (C, D2)
            v = squash(s_c, axis=-1)                          # class capsules (C, D2)
            if r < self.routing_iters - 1:
                b_log = b_log + np.einsum("pcd,cd->pc", u_hat, v)

        hierarchical_features = v.reshape(-1)            # (C*D2,)
        capsule_lengths = np.linalg.norm(v, axis=-1)      # (C,)

        # ---- Graph Neural Network branch ----
        Wg1, bg1, Wg2, bg2 = (self.params[k] for k in ["Wg1", "bg1", "Wg2", "bg2"])
        x0 = x.reshape(-1, 1)                             # (K, 1) node features
        M0 = self.A_norm @ x0                             # (K, 1)
        z1 = M0 @ Wg1 + bg1                                # (K, Gh1)
        h1 = relu(z1)
        M1 = self.A_norm @ h1                              # (K, Gh1)
        z2 = M1 @ Wg2 + bg2                                 # (K, Gh2)
        h2 = relu(z2)
        relational_features = h2.mean(axis=0)               # (Gh2,)  readout

        # ---- Feature Fusion ----
        fused = np.concatenate([hierarchical_features, relational_features])

        Wf1, bf1, Wf2, bf2 = (self.params[k] for k in ["Wf1", "bf1", "Wf2", "bf2"])
        z_fc1 = fused @ Wf1 + bf1
        a_fc1 = relu(z_fc1)
        logits = a_fc1 @ Wf2 + bf2
        probs = softmax(logits)

        cache = dict(x=x, s_p=s_p, u=u, u_hat=u_hat, c_coef=c_coef, s_c=s_c, v=v,
                     capsule_lengths=capsule_lengths, x0=x0, M0=M0, z1=z1, h1=h1,
                     M1=M1, z2=z2, h2=h2, relational_features=relational_features,
                     fused=fused, z_fc1=z_fc1, a_fc1=a_fc1, logits=logits, probs=probs)
        return probs, cache

    # -------------------------------------------------------------- loss/back
    def _margin_loss_and_grad(self, v: np.ndarray, capsule_lengths: np.ndarray, y: int,
                               m_plus=0.9, m_minus=0.1, lambda_absent=0.5):
        C, D2 = v.shape
        loss = 0.0
        d_v = np.zeros_like(v)
        for c in range(C):
            Lc = capsule_lengths[c]
            if c == y:
                margin = max(0.0, m_plus - Lc)
                loss += margin ** 2
                dL_dLc = -2.0 * margin
            else:
                margin = max(0.0, Lc - m_minus)
                loss += lambda_absent * margin ** 2
                dL_dLc = 2.0 * lambda_absent * margin
            if Lc > 1e-9:
                d_v[c] = dL_dLc * (v[c] / Lc)
        return loss, d_v

    def backward_single(self, cache: dict, y: int):
        P, D1, C, D2 = self.P, self.D1, self.C, self.D2
        Wr, Wf1, Wf2, Wg1, Wg2 = (self.params[k] for k in ["Wr", "Wf1", "Wf2", "Wg1", "Wg2"])

        probs, logits = cache["probs"], cache["logits"]
        y_onehot = np.zeros(C); y_onehot[y] = 1.0
        d_logits = probs - y_onehot                       # softmax + cross-entropy grad

        a_fc1, fused, z_fc1 = cache["a_fc1"], cache["fused"], cache["z_fc1"]
        grads = {}
        grads["Wf2"] = np.outer(a_fc1, d_logits)
        grads["bf2"] = d_logits
        d_a_fc1 = Wf2 @ d_logits
        d_z_fc1 = d_a_fc1 * relu_grad(z_fc1)
        grads["Wf1"] = np.outer(fused, d_z_fc1)
        grads["bf1"] = d_z_fc1
        d_fused = Wf1 @ d_z_fc1

        d_hier = d_fused[: C * D2].reshape(C, D2).copy()
        d_relational = d_fused[C * D2:]

        # auxiliary capsule margin loss adds directly onto the class-capsule gradient
        margin_loss, d_v_margin = self._margin_loss_and_grad(cache["v"], cache["capsule_lengths"], y)
        d_v = d_hier + self.lambda_margin * d_v_margin

        # ---- backprop through capsule branch ----
        d_s_c = squash_backward(cache["s_c"], d_v)          # (C, D2)
        c_coef = cache["c_coef"]                             # (P, C), treated as constant
        d_u_hat = np.einsum("pc,cd->pcd", c_coef, d_s_c)     # (P, C, D2)
        grads["Wr"] = np.einsum("pcd,pk->pcdk", d_u_hat, cache["u"])
        d_u = np.einsum("pcd,pcdk->pk", d_u_hat, Wr)         # (P, D1)
        d_s_p = squash_backward(cache["s_p"], d_u)           # (P, D1)
        d_s_p_flat = d_s_p.reshape(-1)
        grads["W1"] = np.outer(cache["x"], d_s_p_flat)
        grads["b1"] = d_s_p_flat

        # ---- backprop through GNN branch ----
        K = self.K
        d_h2 = np.tile((d_relational / K).reshape(1, -1), (K, 1))
        d_z2 = d_h2 * relu_grad(cache["z2"])
        grads["Wg2"] = cache["M1"].T @ d_z2
        grads["bg2"] = d_z2.sum(axis=0)
        d_M1 = d_z2 @ Wg2.T
        d_h1 = self.A_norm.T @ d_M1
        d_z1 = d_h1 * relu_grad(cache["z1"])
        grads["Wg1"] = cache["M0"].T @ d_z1
        grads["bg1"] = d_z1.sum(axis=0)

        ce_loss = -np.log(probs[y] + 1e-12)
        total_loss = ce_loss + self.lambda_margin * margin_loss
        return total_loss, grads

    # ---------------------------------------------------------------- public
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.forward_single(x)[0] for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 8,
                     rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()
        idx = rng.permutation(len(X))
        total_loss = 0.0
        for start in range(0, len(X), batch_size):
            batch_idx = idx[start: start + batch_size]
            accum_grads = {k: np.zeros_like(v) for k, v in self.params.items()}
            batch_loss = 0.0
            for i in batch_idx:
                probs, cache = self.forward_single(X[i])
                loss, grads = self.backward_single(cache, int(y[i]))
                batch_loss += loss
                for k in accum_grads:
                    accum_grads[k] += grads[k]
            n = len(batch_idx)
            for k in accum_grads:
                accum_grads[k] /= n
            self.optimizer.step(self.params, accum_grads)
            total_loss += batch_loss
        return total_loss / len(X)
