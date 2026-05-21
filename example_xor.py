"""
example_xor.py — Solving XOR with a Neural Network

XOR is the classic "non-linearly separable" problem that
a single-layer perceptron CANNOT solve. A hidden layer fixes it.
Run: python examples/example_xor.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.network import NeuralNetwork
from src.utils import make_xor, train_test_split, one_hot

# ── Data ──────────────────────────────────────────────
X, y = make_xor(n=800, noise=0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train_oh = one_hot(y_train, num_classes=2)
y_test_oh  = one_hot(y_test,  num_classes=2)

# ── Build Network ─────────────────────────────────────
nn = (
    NeuralNetwork(loss="cross_entropy")
    .add_layer(2, 16, activation="relu")
    .add_layer(16, 8, activation="relu")
    .add_layer(8, 2, activation="softmax")
)

nn.summary()

# ── Train ─────────────────────────────────────────────
history = nn.train(
    X_train, y_train_oh,
    epochs=200,
    lr=0.01,
    batch_size=32,
    optimizer="adam",
    validation_data=(X_test, y_test_oh),
    verbose=True,
)

# ── Evaluate ──────────────────────────────────────────
test_acc = nn.accuracy(X_test, y_test_oh)
print(f"\n✓ Final Test Accuracy: {test_acc:.4f}")

# ── Manual Verification ───────────────────────────────
print("\nXOR Truth Table Verification:")
print(f"  [0,0] → {nn.predict_classes(np.array([[0,0]]))[0]}  (expected 0)")
print(f"  [0,1] → {nn.predict_classes(np.array([[0,1]]))[0]}  (expected 1)")
print(f"  [1,0] → {nn.predict_classes(np.array([[1,0]]))[0]}  (expected 1)")
print(f"  [1,1] → {nn.predict_classes(np.array([[1,1]]))[0]}  (expected 0)")
