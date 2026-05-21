"""
network.py — Core Neural Network Engine
Pure NumPy implementation. No deep learning frameworks.
"""

import numpy as np
from src.activations import get_activation
from src.losses import get_loss


class Layer:
    """A single fully-connected (dense) layer."""

    def __init__(self, input_size: int, output_size: int, activation: str = "relu"):
        # He initialization for ReLU, Xavier for others
        if activation == "relu":
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(1.0 / input_size)

        self.W = np.random.randn(input_size, output_size) * scale
        self.b = np.zeros((1, output_size))
        self.activation_name = activation
        self.activate, self.activate_prime = get_activation(activation)

        # Cache for backprop
        self.input = None
        self.z = None
        self.output = None

        # Gradients
        self.dW = None
        self.db = None

        # Momentum / Adam state
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.z = x @ self.W + self.b
        self.output = self.activate(self.z)
        return self.output

    def backward(self, delta: np.ndarray) -> np.ndarray:
        m = self.input.shape[0]
        dz = delta * self.activate_prime(self.z)
        self.dW = (self.input.T @ dz) / m
        self.db = np.mean(dz, axis=0, keepdims=True)
        return dz @ self.W.T


class NeuralNetwork:
    """
    Feedforward Neural Network built from scratch.

    Example:
        nn = NeuralNetwork(loss="cross_entropy")
        nn.add_layer(784, 128, activation="relu")
        nn.add_layer(128, 64, activation="relu")
        nn.add_layer(64, 10, activation="softmax")
        nn.train(X_train, y_train, epochs=50, lr=0.001)
    """

    def __init__(self, loss: str = "mse"):
        self.layers: list[Layer] = []
        self.loss_fn, self.loss_prime = get_loss(loss)
        self.loss_name = loss
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def add_layer(self, input_size: int, output_size: int, activation: str = "relu"):
        self.layers.append(Layer(input_size, output_size, activation))
        return self  # chainable

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        delta = self.loss_prime(y_true, y_pred)
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def update(self, optimizer, lr: float, t: int):
        for layer in self.layers:
            optimizer.step(layer, lr, t)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 32,
        optimizer: str = "adam",
        validation_data=None,
        verbose: bool = True,
    ):
        from src.optimizers import get_optimizer
        opt = get_optimizer(optimizer)
        n = X.shape[0]

        for epoch in range(1, epochs + 1):
            # Shuffle
            idx = np.random.permutation(n)
            X_s, y_s = X[idx], y[idx]

            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                xb = X_s[start : start + batch_size]
                yb = y_s[start : start + batch_size]
                pred = self.forward(xb)
                epoch_loss += self.loss_fn(yb, pred)
                self.backward(yb, pred)
                self.update(opt, lr, epoch)

            train_loss = epoch_loss / (n // batch_size)
            train_acc = self.accuracy(X, y)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            val_info = ""
            if validation_data:
                Xv, yv = validation_data
                val_pred = self.forward(Xv)
                val_loss = self.loss_fn(yv, val_pred)
                val_acc = self.accuracy(Xv, yv)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                val_info = f"  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(
                    f"Epoch {epoch:4d}/{epochs} — "
                    f"loss: {train_loss:.4f}  acc: {train_acc:.4f}{val_info}"
                )

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict(X)
        if probs.shape[1] == 1:
            return (probs >= 0.5).astype(int).flatten()
        return np.argmax(probs, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict_classes(X)
        if y.ndim > 1 and y.shape[1] > 1:
            labels = np.argmax(y, axis=1)
        else:
            labels = y.flatten()
        return np.mean(preds == labels)

    def summary(self):
        print("\n" + "=" * 50)
        print("  Neural Network Architecture")
        print("=" * 50)
        total_params = 0
        for i, layer in enumerate(self.layers):
            params = layer.W.size + layer.b.size
            total_params += params
            print(
                f"  Layer {i+1}: {layer.W.shape[0]:>4} → {layer.W.shape[1]:<4} "
                f"| {layer.activation_name:<8} | params: {params:,}"
            )
        print("-" * 50)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Loss function:    {self.loss_name}")
        print("=" * 50 + "\n")

    def save(self, path: str):
        """Save weights to a .npz file."""
        data = {}
        for i, layer in enumerate(self.layers):
            data[f"W{i}"] = layer.W
            data[f"b{i}"] = layer.b
        np.savez(path, **data)
        print(f"Model saved → {path}.npz")

    def load(self, path: str):
        """Load weights from a .npz file."""
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            layer.W = data[f"W{i}"]
            layer.b = data[f"b{i}"]
        print(f"Model loaded ← {path}")
