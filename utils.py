"""
utils.py — Data Preprocessing & Utility Functions
"""

import numpy as np


# ── Normalization ─────────────────────────────────────

def normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalization. Returns (X_norm, mean, std)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


def min_max_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max scaling to [0, 1]. Returns (X_scaled, min, max)."""
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    return (X - xmin) / (xmax - xmin + 1e-8), xmin, xmax


# ── Encoding ──────────────────────────────────────────

def one_hot(y: np.ndarray, num_classes: int = None) -> np.ndarray:
    """Convert integer labels to one-hot encoded matrix."""
    y = y.flatten().astype(int)
    if num_classes is None:
        num_classes = y.max() + 1
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded


# ── Train/Test Split ──────────────────────────────────

def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Split data into train and test sets."""
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ── Dataset Generators ────────────────────────────────

def make_circles(n: int = 500, noise: float = 0.1, seed: int = 42) -> tuple:
    """Generate two concentric circles dataset (binary classification)."""
    np.random.seed(seed)
    n_half = n // 2
    theta = np.linspace(0, 2 * np.pi, n_half)

    X_inner = np.column_stack([np.cos(theta), np.sin(theta)]) * 0.5
    X_outer = np.column_stack([np.cos(theta), np.sin(theta)])

    X = np.vstack([X_inner, X_outer])
    X += np.random.randn(*X.shape) * noise
    y = np.array([0] * n_half + [1] * n_half)
    return X, y


def make_xor(n: int = 400, noise: float = 0.1, seed: int = 42) -> tuple:
    """Generate XOR dataset (binary classification)."""
    np.random.seed(seed)
    X = np.random.randn(n, 2)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X += np.random.randn(*X.shape) * noise
    return X, y


def make_blobs(n: int = 300, centers: int = 3, seed: int = 42) -> tuple:
    """Generate Gaussian blob clusters (multi-class classification)."""
    np.random.seed(seed)
    X_list, y_list = [], []
    for i in range(centers):
        center = np.random.randn(2) * 3
        pts = np.random.randn(n // centers, 2) + center
        X_list.append(pts)
        y_list.extend([i] * (n // centers))
    return np.vstack(X_list), np.array(y_list)


def make_regression(n: int = 200, features: int = 1, noise: float = 0.3, seed: int = 42) -> tuple:
    """Generate a simple regression dataset."""
    np.random.seed(seed)
    X = np.random.randn(n, features)
    true_weights = np.random.randn(features)
    y = X @ true_weights + np.random.randn(n) * noise
    return X, y.reshape(-1, 1)


# ── Metrics ───────────────────────────────────────────

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = None):
    """Compute confusion matrix."""
    y_true = y_true.flatten().astype(int)
    y_pred = y_pred.flatten().astype(int)
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = None):
    """Print precision, recall, F1 per class."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    n = cm.shape[0]
    print(f"\n{'Class':>6} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 50)
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        support = cm[i, :].sum()
        print(f"{i:>6} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")
    acc = np.trace(cm) / cm.sum()
    print(f"\nOverall Accuracy: {acc:.4f}\n")


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² score for regression."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-9)
