from evaluation.metrics import (
    compute_class_metrics,
    compute_macro_avg,
    compute_weighted_avg,
)


def print_classification_report(y_true, y_pred):
    """
    Print a per-class precision / recall / F1 / support table,
    followed by macro and weighted averages.

    Args:
        y_true : list of true labels
        y_pred : list of predicted labels
    """
    classes, metrics = compute_class_metrics(y_true, y_pred)
    total_support    = sum(m["support"] for m in metrics.values())

    col_w = max(len(str(c)) for c in classes) + 2

    # ── Header ────────────────────────────────────────────────────────────
    print(
        f"\n{'':>{col_w}}  {'precision':>9}  {'recall':>9}"
        f"  {'f1-score':>9}  {'support':>9}"
    )
    print()

    # ── Per-class rows ────────────────────────────────────────────────────
    for cls in classes:
        m = metrics[cls]
        print(
            f"{str(cls):>{col_w}}  {m['precision']:>9.2f}  {m['recall']:>9.2f}"
            f"  {m['f1']:>9.2f}  {m['support']:>9}"
        )

    # ── Averages ──────────────────────────────────────────────────────────
    macro    = compute_macro_avg(metrics)
    weighted = compute_weighted_avg(metrics)

    print()
    print(
        f"{'macro avg':>{col_w}}  {macro['precision']:>9.2f}  {macro['recall']:>9.2f}"
        f"  {macro['f1']:>9.2f}  {total_support:>9}"
    )
    print(
        f"{'weighted avg':>{col_w}}  {weighted['precision']:>9.2f}  {weighted['recall']:>9.2f}"
        f"  {weighted['f1']:>9.2f}  {total_support:>9}"
    )
    print()
