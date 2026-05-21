"""
example_mnist.py — Handwritten Digit Classification (MNIST)

Downloads MNIST via sklearn or keras datasets and trains a
fully-connected network to classify digits 0–9.

Run: python examples/example_mnist.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.network import NeuralNetwork
from src.utils import normalize, one_hot, train_test_split, classification_report

# ── Load MNIST ────────────────────────────────────────
try:
    from sklearn.datasets import fetch_openml
    print("Loading MNIST from sklearn...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_raw = mnist.data.astype(np.float32)
    y_raw = mnist.target.astype(int)
except Exception:
    print("sklearn not available, generating synthetic digit data...")
    np.random.seed(42)
    X_raw = np.random.rand(5000, 784).astype(np.float32)
    y_raw = np.random.randint(0, 10, 5000)

# ── Preprocess ────────────────────────────────────────
X_norm, mean, std = normalize(X_raw)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_raw, test_size=0.15)

y_train_oh = one_hot(y_train, num_classes=10)
y_test_oh  = one_hot(y_test,  num_classes=10)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ── Build Network ─────────────────────────────────────
nn = (
    NeuralNetwork(loss="cross_entropy")
    .add_layer(784, 256, activation="relu")
    .add_layer(256, 128, activation="relu")
    .add_layer(128, 64,  activation="relu")
    .add_layer(64,  10,  activation="softmax")
)

nn.summary()

# ── Train ─────────────────────────────────────────────
history = nn.train(
    X_train, y_train_oh,
    epochs=30,
    lr=0.001,
    batch_size=64,
    optimizer="adam",
    validation_data=(X_test, y_test_oh),
    verbose=True,
)

# ── Evaluate ──────────────────────────────────────────
y_pred = nn.predict_classes(X_test)
print("\nClassification Report:")
classification_report(y_test, y_pred, num_classes=10)

# ── Save Model ────────────────────────────────────────
nn.save("data/mnist_model")
