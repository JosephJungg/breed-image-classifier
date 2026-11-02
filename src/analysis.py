"""Error analysis on predictions. Pure functions over label arrays."""

from collections import Counter

import numpy as np

from src.data import CATS


def per_class_accuracy(y_true, y_pred, classes):
    """[(name, accuracy, n)] sorted worst first."""
    rows = []
    for cls_id, name in enumerate(classes):
        mask = y_true == cls_id
        if mask.any():
            rows.append((name, float((y_pred[mask] == cls_id).mean()), int(mask.sum())))
    return sorted(rows, key=lambda r: r[1])


def confused_pairs(y_true, y_pred, classes):
    """Counter of (true name, predicted name) over the mistakes."""
    wrong = np.where(y_true != y_pred)[0]
    return Counter((classes[y_true[i]], classes[y_pred[i]]) for i in wrong)


def cat_dog_mixups(y_true, y_pred, classes, cats=CATS):
    wrong = np.where(y_true != y_pred)[0]
    return int(sum((classes[y_true[i]] in cats) != (classes[y_pred[i]] in cats) for i in wrong))


def report(y_true, y_pred, classes, worst=10, top_pairs=10):
    lines = [f"acc {float((y_true == y_pred).mean()):.4f}  "
             f"({int((y_true != y_pred).sum())} mistakes of {len(y_true)})", ""]
    for name, acc, n in per_class_accuracy(y_true, y_pred, classes)[:worst]:
        lines.append(f"{name:30s} {acc:.2f}  (n={n})")
    lines.append("")
    for (t, p), n in confused_pairs(y_true, y_pred, classes).most_common(top_pairs):
        lines.append(f"{n}x  {t:28s} -> {p}")
    lines.append(f"\ncat/dog mix-ups: {cat_dog_mixups(y_true, y_pred, classes)}")
    return "\n".join(lines)
