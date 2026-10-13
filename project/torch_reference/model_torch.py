"""
model_torch.py
---------------
Reference PyTorch + PyTorch Geometric implementation of the Hybrid Capsule Network +
GNN architecture (Proposal Sections 9, 11-15), matching the tool stack requested in
Section 20 (Python, PyTorch, PyTorch Geometric, scikit-learn).

This is the version intended for real training runs (e.g. on Google Colab with a GPU,
per Section 20) on the actual CuMiDa Leukemia Gene Expression dataset. It is not
executed inside this delivery environment (torch / torch-geometric are not installed
there), so treat it as a reviewed, ready-to-run reference — sanity-check on your own
machine before reporting results. The NumPy engine in `src/` (used by the Flask UI) is
functionally equivalent and has been tested end-to-end.

Install:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install torch_geometric
    pip install scikit-learn pandas numpy networkx matplotlib

Usage:
    from model_torch import HybridCapsGNN
    model = HybridCapsGNN(n_genes=128, n_classes=5)
    logits, class_capsule_lengths = model(x_batch, edge_index, edge_weight)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for this reference implementation. "
        "Install with: pip install torch_geometric"
    ) from e


def squash(s: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    """CapsNet squashing non-linearity (Sabour et al., 2017)."""
    n2 = (s ** 2).sum(dim=dim, keepdim=True)
    n = torch.sqrt(n2 + eps)
    return (n2 / (1.0 + n2)) * (s / n)


class PrimaryCaps(nn.Module):
    """Projects gene-expression input into a set of primary capsules (Section 11)."""

    def __init__(self, n_genes: int, n_primary_caps: int = 8, primary_dim: int = 8):
        super().__init__()
        self.n_primary_caps = n_primary_caps
        self.primary_dim = primary_dim
        self.proj = nn.Linear(n_genes, n_primary_caps * primary_dim)

    def forward(self, x):  # x: (B, n_genes)
        s = self.proj(x).view(-1, self.n_primary_caps, self.primary_dim)
        return squash(s, dim=-1)  # (B, P, D1)


class DigitCaps(nn.Module):
    """Class capsules produced via dynamic routing-by-agreement (Section 11)."""

    def __init__(self, n_primary_caps: int, primary_dim: int, n_classes: int,
                 class_dim: int = 16, routing_iters: int = 3):
        super().__init__()
        self.P, self.D1, self.C, self.D2 = n_primary_caps, primary_dim, n_classes, class_dim
        self.routing_iters = routing_iters
        self.W = nn.Parameter(0.01 * torch.randn(1, self.P, self.C, self.D2, self.D1))

    def forward(self, u):  # u: (B, P, D1)
        B = u.size(0)
        u = u.unsqueeze(2).unsqueeze(-1)                      # (B, P, 1, D1, 1)
        W = self.W.expand(B, -1, -1, -1, -1)                  # (B, P, C, D2, D1)
        u_hat = torch.matmul(W, u).squeeze(-1)                # (B, P, C, D2)

        b_logits = torch.zeros(B, self.P, self.C, device=u.device)
        v = None
        for r in range(self.routing_iters):
            c_coef = F.softmax(b_logits, dim=2)                # (B, P, C)
            s = (c_coef.unsqueeze(-1) * u_hat).sum(dim=1)       # (B, C, D2)
            v = squash(s, dim=-1)                               # (B, C, D2)
            if r < self.routing_iters - 1:
                agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1)  # (B, P, C)
                b_logits = b_logits + agreement
        return v  # class capsules (B, C, D2)


class GeneGNN(nn.Module):
    """Graph Neural Network branch (Section 13): 2-layer GCN over the gene graph,
    followed by mean-pool readout to a per-sample relational feature vector."""

    def __init__(self, hidden1: int = 16, hidden2: int = 16):
        super().__init__()
        self.conv1 = GCNConv(1, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)

    def forward(self, x, edge_index, edge_weight, batch):
        # x: (total_nodes_in_batch, 1) node features (gene expression values)
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = F.relu(self.conv2(h, edge_index, edge_weight))
        from torch_geometric.nn import global_mean_pool
        return global_mean_pool(h, batch)  # (B, hidden2)


class HybridCapsGNN(nn.Module):
    """Full pipeline: PrimaryCaps -> DigitCaps (hierarchical features) in parallel with
    GCN (relational features) -> concat fusion -> FC -> softmax (Sections 9, 14-15)."""

    def __init__(self, n_genes: int, n_classes: int, n_primary_caps=8, primary_dim=8,
                 class_dim=16, gnn_hidden1=16, gnn_hidden2=16, fc_hidden=64, routing_iters=3):
        super().__init__()
        self.n_classes = n_classes
        self.primary_caps = PrimaryCaps(n_genes, n_primary_caps, primary_dim)
        self.digit_caps = DigitCaps(n_primary_caps, primary_dim, n_classes, class_dim, routing_iters)
        self.gnn = GeneGNN(gnn_hidden1, gnn_hidden2)
        fusion_dim = n_classes * class_dim + gnn_hidden2
        self.fc1 = nn.Linear(fusion_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_classes)

    def forward(self, x, edge_index, edge_weight, batch):
        """
        x           : (B, n_genes) dense gene-expression batch, for the capsule branch
        edge_index  : (2, E) gene-graph edges (shared graph, batched via torch_geometric)
        edge_weight : (E,) correlation-based edge weights
        batch       : (total_nodes_in_batch,) graph-batch index, for the GNN branch's
                      node-feature view of the SAME x (reshaped to one node per gene)
        """
        v = self.digit_caps(self.primary_caps(x))              # (B, C, D2)
        hierarchical_features = v.reshape(v.size(0), -1)         # (B, C*D2)
        capsule_lengths = v.norm(dim=-1)                          # (B, C)

        node_x = x.reshape(-1, 1)                                  # (B*n_genes, 1)
        relational_features = self.gnn(node_x, edge_index, edge_weight, batch)  # (B, Gh2)

        fused = torch.cat([hierarchical_features, relational_features], dim=-1)
        h = F.relu(self.fc1(fused))
        logits = self.fc2(h)
        return logits, capsule_lengths


def margin_loss(capsule_lengths: torch.Tensor, targets: torch.Tensor,
                 m_plus=0.9, m_minus=0.1, lambda_absent=0.5) -> torch.Tensor:
    """Standard CapsNet margin loss (auxiliary supervision for the capsule branch)."""
    C = capsule_lengths.size(1)
    T = F.one_hot(targets, num_classes=C).float()
    present = T * torch.clamp(m_plus - capsule_lengths, min=0) ** 2
    absent = (1 - T) * lambda_absent * torch.clamp(capsule_lengths - m_minus, min=0) ** 2
    return (present + absent).sum(dim=1).mean()
