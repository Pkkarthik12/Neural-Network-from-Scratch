# 🧠 Neural Network from Scratch

A fully-connected feedforward neural network built **entirely from scratch** using NumPy — no TensorFlow, no PyTorch, no shortcuts. Every forward pass, backpropagation step, and optimizer update is implemented mathematically from first principles.

---

## ✨ Features

| Component | Options |
|-----------|---------|
| **Activations** | ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax, ELU, Swish, Linear |
| **Loss Functions** | MSE, MAE, Binary Cross-Entropy, Categorical Cross-Entropy, Huber |
| **Optimizers** | SGD, Momentum SGD, RMSProp, Adam |
| **Tasks** | Binary classification, Multi-class classification, Regression |

---

## 📁 Project Structure

```
neural-network-from-scratch/
├── src/
│   ├── __init__.py         # Public API
│   ├── network.py          # Core: Layer + NeuralNetwork classes
│   ├── activations.py      # All activation functions + derivatives
│   ├── losses.py           # All loss functions + gradients
│   ├── optimizers.py       # SGD, Momentum, RMSProp, Adam
│   └── utils.py            # Data preprocessing, metrics, dataset generators
├── examples/
│   ├── example_xor.py      # XOR problem (non-linear classification)
│   ├── example_mnist.py    # Handwritten digit recognition
│   └── example_regression.py  # Sine wave regression
├── tests/
│   └── test_network.py     # Unit tests (pytest)
├── data/                   # Saved model weights (.npz)
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### Install

```bash
git clone https://github.com/yourusername/neural-network-from-scratch.git
cd neural-network-from-scratch
pip install -r requirements.txt
```

### Build and Train a Network

```python
from src.network import NeuralNetwork
from src.utils import make_circles, one_hot, train_test_split

# Generate data
X, y = make_circles(n=500, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y)
y_train_oh = one_hot(y_train, num_classes=2)
y_test_oh  = one_hot(y_test,  num_classes=2)

# Build network (chainable API)
nn = (
    NeuralNetwork(loss="cross_entropy")
    .add_layer(2, 32, activation="relu")
    .add_layer(32, 16, activation="relu")
    .add_layer(16, 2,  activation="softmax")
)

nn.summary()

# Train
history = nn.train(
    X_train, y_train_oh,
    epochs=100,
    lr=0.001,
    batch_size=32,
    optimizer="adam",
    validation_data=(X_test, y_test_oh),
)

# Evaluate
print(f"Test Accuracy: {nn.accuracy(X_test, y_test_oh):.4f}")

# Save
nn.save("data/my_model")
```

---

## 🔬 Examples

```bash
# XOR (non-linear binary classification)
python examples/example_xor.py

# MNIST digit recognition
python examples/example_mnist.py

# Sine wave regression
python examples/example_regression.py
```

---

## 🧪 Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## 🧮 How It Works

### Forward Pass
Each layer computes:
```
z = X @ W + b
output = activation(z)
```

### Backpropagation
Gradients flow backwards via the chain rule:
```
dz = delta * activation'(z)
dW = (input.T @ dz) / m
db = mean(dz)
delta_prev = dz @ W.T
```

### Adam Optimizer
```
m = β₁·m + (1-β₁)·dW          # 1st moment
v = β₂·v + (1-β₂)·dW²         # 2nd moment
m̂ = m / (1 - β₁ᵗ)             # bias correction
v̂ = v / (1 - β₂ᵗ)
W -= lr · m̂ / (√v̂ + ε)
```

---

## 📊 Supported Tasks

| Task | Loss | Output Activation |
|------|------|------------------|
| Binary Classification | `binary_cross_entropy` | `sigmoid` |
| Multi-class Classification | `cross_entropy` | `softmax` |
| Regression | `mse` or `huber` | `linear` |

---

## 📄 License

MIT — see [LICENSE](LICENSE)
