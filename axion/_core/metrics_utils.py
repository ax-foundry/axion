"""
Pure numpy implementations of common metrics.

Replaces scikit-learn dependency for lightweight metric calculations.
"""

import numpy as np


def confusion_matrix_binary(y_true, y_pred):
    """
    Compute confusion matrix values for binary classification.

    Args:
        y_true: Ground truth binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)

    Returns:
        Tuple of (tn, fp, fn, tp) as integers
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def precision_score(y_true, y_pred, zero_division=0.0):
    """
    Compute binary precision: tp / (tp + fp).

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        zero_division: Value to return when there are no positive predictions

    Returns:
        Precision score as float
    """
    _, fp, _, tp = confusion_matrix_binary(y_true, y_pred)
    denom = tp + fp
    return tp / denom if denom > 0 else zero_division


def recall_score(y_true, y_pred, zero_division=0.0):
    """
    Compute binary recall: tp / (tp + fn).

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        zero_division: Value to return when there are no actual positives

    Returns:
        Recall score as float
    """
    _, _, fn, tp = confusion_matrix_binary(y_true, y_pred)
    denom = tp + fn
    return tp / denom if denom > 0 else zero_division


def f1_score(y_true, y_pred, zero_division=0.0):
    """
    Compute binary F1 score: 2 * (precision * recall) / (precision + recall).

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        zero_division: Value to return when precision + recall is zero

    Returns:
        F1 score as float
    """
    p = precision_score(y_true, y_pred, zero_division=0.0)
    r = recall_score(y_true, y_pred, zero_division=0.0)
    denom = p + r
    return 2 * p * r / denom if denom > 0 else zero_division


def cohen_kappa_score(y_true, y_pred):
    """
    Compute Cohen's kappa coefficient for inter-rater agreement.

    Measures agreement between two raters beyond chance agreement.

    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels

    Returns:
        Kappa coefficient as float (-1 to 1, where 1 is perfect agreement)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0

    # Observed agreement
    po = np.sum(y_true == y_pred) / n

    # Expected agreement by chance
    p_true_1 = np.sum(y_true == 1) / n
    p_pred_1 = np.sum(y_pred == 1) / n
    p_true_0 = 1 - p_true_1
    p_pred_0 = 1 - p_pred_1
    pe = (p_true_1 * p_pred_1) + (p_true_0 * p_pred_0)

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def cosine_similarity(X):
    """
    Compute pairwise cosine similarity matrix.

    Args:
        X: 2D array of shape (n_samples, n_features)

    Returns:
        Similarity matrix of shape (n_samples, n_samples)
    """
    X = np.asarray(X)
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    # Handle zero vectors to avoid division by zero
    norm = np.where(norm == 0, 1, norm)
    X_normalized = X / norm
    return X_normalized @ X_normalized.T
