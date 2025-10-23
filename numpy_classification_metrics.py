
"""
numpy_classification_metrics.py
--------------------------------
Pure NumPy implementations of common classification metrics.
Includes Accuracy, Precision, Recall, F1, Confusion Matrix,
Specificity, TPR, FPR, and Top-k Accuracy.
"""

import numpy as np


# ------------------------------------------------------------
#  Basic Metrics
# ------------------------------------------------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes=None) -> np.ndarray:
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


# ------------------------------------------------------------
#  Binary Helpers
# ------------------------------------------------------------
def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray):
    cm = confusion_matrix(y_true, y_pred, num_classes=2)
    tn, fp, fn, tp = cm.ravel()
    return tp, fp, tn, fn


def tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, fp, tn, fn = _binary_counts(y_true, y_pred)
    return tp / np.clip(tp + fn, 1e-12, None)


def fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, fp, tn, fn = _binary_counts(y_true, y_pred)
    return fp / np.clip(fp + tn, 1e-12, None)


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, fp, tn, fn = _binary_counts(y_true, y_pred)
    return tn / np.clip(tn + fp, 1e-12, None)


# ------------------------------------------------------------
#  Multi-class Metrics
# ------------------------------------------------------------
def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary") -> float:
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)
    prec = tp / np.clip(tp + fp, 1e-12, None)

    if average == "binary":
        return prec[1] if len(prec) > 1 else prec[0]
    elif average == "macro":
        return np.nanmean(prec)
    elif average == "micro":
        return tp.sum() / np.clip(tp.sum() + fp.sum(), 1e-12, None)
    elif average == "weighted":
        return np.sum(prec * support) / np.clip(support.sum(), 1e-12, None)
    else:
        raise ValueError(f"Unknown average: {average}")


def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary") -> float:
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1)
    rec = tp / np.clip(tp + fn, 1e-12, None)

    if average == "binary":
        return rec[1] if len(rec) > 1 else rec[0]
    elif average == "macro":
        return np.nanmean(rec)
    elif average == "micro":
        return tp.sum() / np.clip(tp.sum() + fn.sum(), 1e-12, None)
    elif average == "weighted":
        return np.sum(rec * support) / np.clip(support.sum(), 1e-12, None)
    else:
        raise ValueError(f"Unknown average: {average}")


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary") -> float:
    p = precision(y_true, y_pred, average)
    r = recall(y_true, y_pred, average)
    return 2 * p * r / np.clip(p + r, 1e-12, None)


def top_k_accuracy(y_true: np.ndarray, y_score: np.ndarray, k: int = 5) -> float:
    topk = np.argpartition(-y_score, k - 1, axis=1)[:, :k]
    correct = (topk == y_true[:, None]).any(axis=1)
    return np.mean(correct)


# ------------------------------------------------------------
#  Demo
# ------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    y_true = np.random.randint(0, 3, size=20)
    y_pred = np.random.randint(0, 3, size=20)
    y_score = np.random.rand(20, 3)

    print("y_true:", y_true)
    print("y_pred:", y_pred)
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

    print(f"\nAccuracy:      {accuracy(y_true, y_pred):.3f}")
    print(f"Precision:     {precision(y_true, y_pred, 'macro'):.3f}")
    print(f"Recall:        {recall(y_true, y_pred, 'macro'):.3f}")
    print(f"F1 Score:      {f1_score(y_true, y_pred, 'macro'):.3f}")
    print(f"TPR:           {tpr((y_true==1).astype(int), (y_pred==1).astype(int)):.3f}")
    print(f"FPR:           {fpr((y_true==1).astype(int), (y_pred==1).astype(int)):.3f}")
    print(f"Specificity:   {specificity((y_true==1).astype(int), (y_pred==1).astype(int)):.3f}")
    print(f"Top-k (k=2):   {top_k_accuracy(y_true, y_score, k=2):.3f}")
