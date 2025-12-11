import json
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    labels = labels.astype(int)
    preds = (probs >= threshold).astype(int)

    metrics = {}
    try:
        metrics['auc'] = float(roc_auc_score(labels, probs))
    except Exception:
        metrics['auc'] = float('nan')

    try:
        metrics['avg_precision'] = float(average_precision_score(labels, probs))
    except Exception:
        metrics['avg_precision'] = float('nan')

    metrics['f1'] = float(f1_score(labels, preds))
    metrics['precision'] = float(precision_score(labels, preds))
    metrics['recall'] = float(recall_score(labels, preds))
    metrics['accuracy'] = float(accuracy_score(labels, preds))
    try:
        metrics['balanced_accuracy'] = float(balanced_accuracy_score(labels, preds))
    except Exception:
        metrics['balanced_accuracy'] = float('nan')
    metrics['threshold_used'] = float(threshold)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    metrics['confusion_matrix'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    return metrics


def find_best_threshold(labels: np.ndarray, probs: np.ndarray, metric: str = 'f1', steps: int = 101) -> Tuple[float, Dict[str, Any]]:
    labels = labels.astype(int)
    best_thr = 0.5
    best_score = -1.0
    best_metrics = {}
    for t in np.linspace(0.0, 1.0, steps):
        m = compute_metrics(labels, probs, threshold=t)
        score = m.get(metric, float('nan'))
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_thr = float(t)
            best_metrics = m
    return best_thr, best_metrics


def find_threshold_for_accuracy(labels: np.ndarray, probs: np.ndarray, target_acc: float = 0.8, steps: int = 201) -> Tuple[float, Dict[str, Any]]:
    """Find a threshold that achieves at least target accuracy; among candidates, maximize F1.

    If no threshold reaches target accuracy, return the one with the highest accuracy.
    """
    labels = labels.astype(int)
    best_thr = 0.5
    best_metrics = {}
    best_acc = -1.0
    best_f1_at_target = -1.0
    chosen_thr = None
    chosen_metrics = None
    for t in np.linspace(0.0, 1.0, steps):
        m = compute_metrics(labels, probs, threshold=t)
        acc = m.get('accuracy', float('nan'))
        f1 = m.get('f1', float('nan'))
        if not np.isnan(acc) and acc > best_acc:
            best_acc = acc
            best_thr = float(t)
            best_metrics = m
        if not np.isnan(acc) and acc >= target_acc:
            if np.isnan(f1):
                f1 = -1.0
            if f1 > best_f1_at_target:
                best_f1_at_target = f1
                chosen_thr = float(t)
                chosen_metrics = m
    if chosen_thr is not None:
        return chosen_thr, chosen_metrics
    return best_thr, best_metrics


def save_metrics_json(metrics: Dict[str, Any], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
