"""
test_network.py — Unit Tests for Neural Network Components
Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.activations import get_activation, list_activations
from src.losses import get_loss, list_losses
from src.optimizers import get_optimizer
from src.network import NeuralNetwork, Layer
from src.utils import one_hot, normalize, make_circles, make_xor


# ── Activations ───────────────────────────────────────

class TestActivations:
    def test_relu_positive(self):
        relu, _ = get_activation("relu")
        assert relu(np.array([1.0, 2.0])).tolist() == [1.0, 2.0]

    def test_relu_negative(self):
        relu, _ = get_activation("relu")
        assert relu(np.array([-1.0, -2.0])).tolist() == [0.0, 0.0]

    def test_sigmoid_range(self):
        sigmoid, _ = get_activation("sigmoid")
        out = sigmoid(np.array([-100.0, 0.0, 100.0]))
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_sigmoid_midpoint(self):
        sigmoid, _ = get_activation("sigmoid")
        assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-6

    def test_softmax_sum_to_one(self):
        softmax, _ = get_activation("softmax")
        z = np.array([[1.0, 2.0, 3.0]])
        out = softmax(z)
        assert abs(out.sum() - 1.0) < 1e-6

    def test_tanh_range(self):
        tanh, _ = get_activation("tanh")
        out = tanh(np.linspace(-10, 10, 100))
        assert np.all(out >= -1) and np.all(out <= 1)

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError):
            get_activation("unknown_fn")

    def test_all_activations_have_primes(self):
        for name in list_activations():
            fn, fn_prime = get_activation(name)
            z = np.array([[0.5, -0.5]])
            fn(z)   # should not raise
            fn_prime(z)  # should not raise


# ── Losses ────────────────────────────────────────────

class TestLosses:
    def test_mse_zero_when_equal(self):
        mse, _ = get_loss("mse")
        y = np.array([[1.0, 2.0]])
        assert mse(y, y) == 0.0

    def test_mse_positive(self):
        mse, _ = get_loss("mse")
        y_true = np.array([[1.0]])
        y_pred = np.array([[2.0]])
        assert mse(y_true, y_pred) > 0

    def test_cross_entropy_perfect(self):
        ce, _ = get_loss("cross_entropy")
        y = np.array([[1.0, 0.0]])
        pred = np.array([[0.9999, 0.0001]])
        assert ce(y, pred) < 0.1

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError):
            get_loss("unknown_loss")


# ── Network ───────────────────────────────────────────

class TestNetwork:
    def test_forward_shape(self):
        nn = NeuralNetwork()
        nn.add_layer(4, 8, "relu")
        nn.add_layer(8, 3, "softmax")
        X = np.random.randn(10, 4)
        out = nn.forward(X)
        assert out.shape == (10, 3)

    def test_layer_chain(self):
        nn = (
            NeuralNetwork()
            .add_layer(2, 4, "relu")
            .add_layer(4, 1, "sigmoid")
        )
        assert len(nn.layers) == 2

    def test_predict_classes_binary(self):
        nn = NeuralNetwork(loss="binary_cross_entropy")
        nn.add_layer(2, 4, "relu")
        nn.add_layer(4, 1, "sigmoid")
        X = np.random.randn(5, 2)
        classes = nn.predict_classes(X)
        assert classes.shape == (5,)
        assert set(classes).issubset({0, 1})

    def test_train_xor_converges(self):
        """Network should solve XOR above 90% accuracy."""
        X, y = make_xor(n=400, noise=0.0, seed=1)
        y_oh = one_hot(y, 2)
        nn = (
            NeuralNetwork(loss="cross_entropy")
            .add_layer(2, 16, "relu")
            .add_layer(16, 2, "softmax")
        )
        nn.train(X, y_oh, epochs=300, lr=0.01, verbose=False)
        acc = nn.accuracy(X, y_oh)
        assert acc > 0.90, f"XOR accuracy too low: {acc:.4f}"

    def test_history_keys(self):
        nn = NeuralNetwork()
        nn.add_layer(2, 2, "relu")
        nn.add_layer(2, 1, "sigmoid")
        X = np.random.randn(20, 2)
        y = np.random.randint(0, 2, (20, 1)).astype(float)
        nn.train(X, y, epochs=5, verbose=False)
        assert "train_loss" in nn.history
        assert len(nn.history["train_loss"]) == 5


# ── Utilities ─────────────────────────────────────────

class TestUtils:
    def test_one_hot_shape(self):
        y = np.array([0, 1, 2, 1])
        oh = one_hot(y, num_classes=3)
        assert oh.shape == (4, 3)

    def test_one_hot_values(self):
        y = np.array([0, 2])
        oh = one_hot(y, num_classes=3)
        assert oh[0, 0] == 1 and oh[0, 1] == 0
        assert oh[1, 2] == 1 and oh[1, 0] == 0

    def test_normalize_mean_std(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_norm, mean, std = normalize(X)
        assert abs(X_norm.mean()) < 1e-6

    def test_make_circles_shapes(self):
        X, y = make_circles(n=100)
        assert X.shape == (100, 2)
        assert set(y) == {0, 1}
