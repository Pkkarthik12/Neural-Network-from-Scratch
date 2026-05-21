"""
losses.py — Loss Functions & Their Gradients
All implemented from scratch using NumPy only.
"""

import numpy as np


_EPS = 1e-9  # numerical stability


# ── Mean Squared Error ────────────────────────────────
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]


# ── Mean Absolute Error ───────────────────────────────
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mae_prime(y_true, y_pred):
    return np.sign(y_pred - y_true) / y_true.shape[0]


# ── Binary Cross-Entropy ──────────────────────────────
def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, _EPS, 1 - _EPS)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    y_pred = np.clip(y_pred, _EPS, 1 - _EPS)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])


# ── Categorical Cross-Entropy ─────────────────────────
def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, _EPS, 1.0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def cross_entropy_prime(y_true, y_pred):
    # Combined gradient for softmax + cross-entropy
    return y_pred - y_true


# ── Huber Loss ────────────────────────────────────────
def huber(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return np.mean(0.5 * quadratic ** 2 + delta * linear)

def huber_prime(y_true, y_pred, delta=1.0):
    error = y_pred - y_true
    return np.where(np.abs(error) <= delta, error, delta * np.sign(error)) / y_true.shape[0]


# ── Registry ──────────────────────────────────────────
_LOSSES = {
    "mse":                   (mse,                  mse_prime),
    "mae":                   (mae,                  mae_prime),
    "binary_cross_entropy":  (binary_cross_entropy, binary_cross_entropy_prime),
    "cross_entropy":         (cross_entropy,        cross_entropy_prime),
    "huber":                 (huber,                huber_prime),
}


def get_loss(name: str):
    name = name.lower()
    if name not in _LOSSES:
        raise ValueError(
            f"Unknown loss '{name}'. "
            f"Available: {list(_LOSSES.keys())}"
        )
    return _LOSSES[name]


def list_losses():
    return list(_LOSSES.keys())
