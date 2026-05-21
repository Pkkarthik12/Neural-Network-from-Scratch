"""
activations.py — Activation Functions & Their Derivatives
All implemented from scratch using NumPy only.
"""

import numpy as np


# ── ReLU ──────────────────────────────────────────────
def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)


# ── Leaky ReLU ────────────────────────────────────────
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_prime(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)


# ── Sigmoid ───────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)


# ── Tanh ──────────────────────────────────────────────
def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    return 1.0 - np.tanh(z) ** 2


# ── Softmax ───────────────────────────────────────────
def softmax(z):
    # Numerically stable softmax
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def softmax_prime(z):
    # When paired with cross-entropy loss, gradient simplifies to (pred - true)
    # This placeholder returns ones; actual gradient handled in loss backprop
    return np.ones_like(z)


# ── Linear (no activation) ────────────────────────────
def linear(z):
    return z

def linear_prime(z):
    return np.ones_like(z)


# ── ELU ───────────────────────────────────────────────
def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))

def elu_prime(z, alpha=1.0):
    return np.where(z > 0, 1.0, elu(z, alpha) + alpha)


# ── Swish ─────────────────────────────────────────────
def swish(z):
    return z * sigmoid(z)

def swish_prime(z):
    s = sigmoid(z)
    return s + z * s * (1 - s)


# ── Registry ──────────────────────────────────────────
_ACTIVATIONS = {
    "relu":       (relu,       relu_prime),
    "leaky_relu": (leaky_relu, leaky_relu_prime),
    "sigmoid":    (sigmoid,    sigmoid_prime),
    "tanh":       (tanh,       tanh_prime),
    "softmax":    (softmax,    softmax_prime),
    "linear":     (linear,     linear_prime),
    "elu":        (elu,        elu_prime),
    "swish":      (swish,      swish_prime),
}


def get_activation(name: str):
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. "
            f"Available: {list(_ACTIVATIONS.keys())}"
        )
    return _ACTIVATIONS[name]


def list_activations():
    return list(_ACTIVATIONS.keys())
