"""
example_regression.py — Nonlinear Regression

Fits a neural network to a noisy sine wave — a classic
demonstration that NNs are universal function approximators.

Run: python examples/example_regression.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.network import NeuralNetwork
from src.utils import min_max_scale, train_test_split, r_squared

# ── Data: noisy sine wave ─────────────────────────────
np.random.seed(0)
X = np.linspace(-np.pi, np.pi, 500).reshape(-1, 1)
y = np.sin(X) + np.random.randn(*X.shape) * 0.15

X_scaled, xmin, xmax = min_max_scale(X)
y_scaled, ymin, ymax = min_max_scale(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2)

# ── Build Network ─────────────────────────────────────
nn = (
    NeuralNetwork(loss="mse")
    .add_layer(1, 32, activation="tanh")
    .add_layer(32, 32, activation="tanh")
    .add_layer(32, 1, activation="linear")
)

nn.summary()

# ── Train ─────────────────────────────────────────────
history = nn.train(
    X_train, y_train,
    epochs=500,
    lr=0.005,
    batch_size=32,
    optimizer="adam",
    validation_data=(X_test, y_test),
    verbose=True,
)

# ── Evaluate ──────────────────────────────────────────
y_pred = nn.predict(X_test)
r2 = r_squared(y_test, y_pred)
print(f"\n✓ R² Score on test set: {r2:.4f}")

# ── Sample Predictions ────────────────────────────────
print("\nSample Predictions (scaled):")
for xi, yi, yp in zip(X_test[:5], y_test[:5], y_pred[:5]):
    print(f"  x={xi[0]:.3f}  true={yi[0]:.3f}  pred={yp[0]:.3f}")
