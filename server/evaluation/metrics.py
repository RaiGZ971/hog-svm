def compute_class_metrics(y_true, y_pred):
    """
    Compute per-class TP, FP, FN, precision, recall, F1, and support.

    Returns:
        classes (list): sorted unique class labels
        metrics (dict): {class_label: {tp, fp, fn, precision, recall, f1, support}}
    """
    classes = sorted(set(y_true) | set(y_pred))
    metrics = {}

    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        support = sum(1 for t in y_true if t == cls)

        metrics[cls] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return classes, metrics


def compute_accuracy(y_true, y_pred):
    """
    Returns the fraction of correct predictions.
    """
    if len(y_true) == 0:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def compute_macro_avg(metrics):
    """
    Unweighted average of precision, recall, and F1 across all classes.

    Returns:
        dict with keys: precision, recall, f1
    """
    n = len(metrics)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "precision": sum(m["precision"] for m in metrics.values()) / n,
        "recall":    sum(m["recall"]    for m in metrics.values()) / n,
        "f1":        sum(m["f1"]        for m in metrics.values()) / n,
    }


def compute_weighted_avg(metrics):
    """
    Support-weighted average of precision, recall, and F1 across all classes.

    Returns:
        dict with keys: precision, recall, f1
    """
    total = sum(m["support"] for m in metrics.values())
    if total == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "precision": sum(m["precision"] * m["support"] for m in metrics.values()) / total,
        "recall":    sum(m["recall"]    * m["support"] for m in metrics.values()) / total,
        "f1":        sum(m["f1"]        * m["support"] for m in metrics.values()) / total,
    }
