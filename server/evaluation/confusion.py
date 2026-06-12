import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def build_confusion_matrix(y_true, y_pred, classes):
    """
    Build a confusion matrix as a 2-D list.

    Args:
        y_true  : list of true labels
        y_pred  : list of predicted labels
        classes : ordered list of class labels (defines row/col order)

    Returns:
        cm (list[list[int]]): cm[i][j] = number of samples with
                              true class i predicted as class j
    """
    idx = {cls: i for i, cls in enumerate(classes)}
    n   = len(classes)
    cm  = [[0] * n for _ in range(n)]

    for t, p in zip(y_true, y_pred):
        cm[idx[t]][idx[p]] += 1

    return cm


def plot_confusion_matrix(cm, classes, title="FSL-SVM Confusion Matrix"):
    """
    Plot the confusion matrix as a heatmap using only matplotlib.

    Args:
        cm      : 2-D list from build_confusion_matrix
        classes : ordered list of class labels
        title   : plot title
    """


    n      = len(classes)
    data   = [[cm[i][j] for j in range(n)] for i in range(n)]
    maxval = max(cell for row in data for cell in row) or 1

    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.6)))

    # ── Draw cells ────────────────────────────────────────────────────────
    for i in range(n):
        for j in range(n):
            intensity = data[i][j] / maxval
            bg_color  = (1 - intensity * 0.85, 1 - intensity * 0.85, 1.0)  # blue tint
            ax.add_patch(plt.Rectangle((j, n - i - 1), 1, 1, color=bg_color))
            text_color = "white" if intensity > 0.55 else "black"
            ax.text(
                j + 0.5, n - i - 0.5, str(data[i][j]),
                ha="center", va="center",
                fontsize=max(7, 11 - n // 5),
                color=text_color,
                fontweight="bold" if data[i][j] > 0 else "normal",
            )

    # ── Grid lines ────────────────────────────────────────────────────────
    for k in range(n + 1):
        ax.axhline(k, color="white", linewidth=1.5)
        ax.axvline(k, color="white", linewidth=1.5)

    # ── Axes ──────────────────────────────────────────────────────────────
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_xticklabels(classes, rotation=45, ha="right",
                       fontsize=max(7, 10 - n // 8))
    ax.set_yticklabels(reversed(classes), fontsize=max(7, 10 - n // 8))

    ax.set_xlabel("Predicted", fontsize=11, labelpad=10)
    ax.set_ylabel("Actual",    fontsize=11, labelpad=10)
    ax.set_title(title,        fontsize=13, pad=14)

    plt.tight_layout()
    plt.show()
