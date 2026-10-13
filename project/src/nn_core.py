"""
nn_core.py
----------
Minimal, dependency-free (NumPy only) neural-network primitives used to build the
Hybrid Capsule Network + GNN model from scratch, since torch / tensorflow / PyTorch
Geometric are not guaranteed to be installed in every environment (see README for the
PyTorch + PyTorch Geometric reference implementation that mirrors the tool-stack
requested in Proposal Section 20, for use on a machine/Colab where those are available).

Provides:
  - relu, relu_grad
  - softmax (numerically stable)
  - squash / squash_backward : the CapsNet non-linearity (Sabour et al., 2017) and its
    closed-form vector-Jacobian product, used to backprop through capsule layers
    without needing full autodiff.
  - Adam: a small parameter-dict optimizer.
"""
from __future__ import annotations
import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def squash(s: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    """CapsNet squashing non-linearity: v = (||s||^2 / (1+||s||^2)) * s/||s||
    Works on a single capsule vector or a batch of capsule vectors along `axis`.
    """
    n2 = np.sum(s * s, axis=axis, keepdims=True)
    n = np.sqrt(n2 + eps)
    return (n2 / (1.0 + n2)) * (s / n)


def squash_backward(s: np.ndarray, g: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Closed-form vector-Jacobian product for `squash`.
    Given s (the pre-squash capsule vector) and g = dL/d(squash(s)), returns dL/ds.

    Derivation: let n = ||s||, f(n) = n/(1+n^2). squash(s) = f(n)*s.
        d(squash_i)/d(s_j) = f(n)*delta_ij + s_i * f'(n) * (s_j/n)
        f'(n) = (1 - n^2) / (1 + n^2)^2
    =>  dL/ds = f(n)*g + (f'(n)/n) * (g . s) * s
    Operates along the last axis; s and g may have leading batch/capsule dims.
    """
    n2 = np.sum(s * s, axis=-1, keepdims=True)
    n = np.sqrt(n2 + eps)
    f = n / (1.0 + n2)
    f_prime = (1.0 - n2) / np.square(1.0 + n2)
    gs = np.sum(g * s, axis=-1, keepdims=True)
    return f * g + (f_prime / n) * gs * s


class Adam:
    """Adam optimizer operating on a flat dict of {name: np.ndarray} parameters."""

    def __init__(self, params: dict, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g * g)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
