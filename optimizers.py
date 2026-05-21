"""
optimizers.py — Gradient Descent Optimizers
SGD, Momentum, RMSProp, and Adam — all from scratch.
"""

import numpy as np


class SGD:
    """Vanilla Stochastic Gradient Descent."""

    def step(self, layer, lr: float, t: int):
        layer.W -= lr * layer.dW
        layer.b -= lr * layer.db


class MomentumSGD:
    """SGD with Momentum (Polyak momentum)."""

    def __init__(self, beta: float = 0.9):
        self.beta = beta

    def step(self, layer, lr: float, t: int):
        layer.mW = self.beta * layer.mW + (1 - self.beta) * layer.dW
        layer.mb = self.beta * layer.mb + (1 - self.beta) * layer.db
        layer.W -= lr * layer.mW
        layer.b -= lr * layer.mb


class RMSProp:
    """RMSProp optimizer."""

    def __init__(self, beta: float = 0.9, eps: float = 1e-8):
        self.beta = beta
        self.eps = eps

    def step(self, layer, lr: float, t: int):
        layer.vW = self.beta * layer.vW + (1 - self.beta) * layer.dW ** 2
        layer.vb = self.beta * layer.vb + (1 - self.beta) * layer.db ** 2
        layer.W -= lr * layer.dW / (np.sqrt(layer.vW) + self.eps)
        layer.b -= lr * layer.db / (np.sqrt(layer.vb) + self.eps)


class Adam:
    """
    Adam optimizer — Adaptive Moment Estimation.
    Kingma & Ba (2015): https://arxiv.org/abs/1412.6980
    """

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def step(self, layer, lr: float, t: int):
        b1, b2 = self.beta1, self.beta2

        # First moment (mean)
        layer.mW = b1 * layer.mW + (1 - b1) * layer.dW
        layer.mb = b1 * layer.mb + (1 - b1) * layer.db

        # Second moment (uncentered variance)
        layer.vW = b2 * layer.vW + (1 - b2) * layer.dW ** 2
        layer.vb = b2 * layer.vb + (1 - b2) * layer.db ** 2

        # Bias correction
        mW_hat = layer.mW / (1 - b1 ** t)
        mb_hat = layer.mb / (1 - b1 ** t)
        vW_hat = layer.vW / (1 - b2 ** t)
        vb_hat = layer.vb / (1 - b2 ** t)

        # Weight update
        layer.W -= lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
        layer.b -= lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


# ── Registry ──────────────────────────────────────────
_OPTIMIZERS = {
    "sgd":      SGD,
    "momentum": MomentumSGD,
    "rmsprop":  RMSProp,
    "adam":     Adam,
}


def get_optimizer(name: str):
    name = name.lower()
    if name not in _OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Available: {list(_OPTIMIZERS.keys())}"
        )
    return _OPTIMIZERS[name]()


def list_optimizers():
    return list(_OPTIMIZERS.keys())
