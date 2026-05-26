import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


def classification_metrics(y_true, y_pred, class_names):
    labels = sorted(class_names.keys())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_accuracy = _per_class_accuracy(matrix, labels, class_names)

    return {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "average_accuracy": float(np.mean([row["accuracy"] for row in per_class_accuracy])),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels)),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
    }


def save_metrics(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_per_class_accuracy(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics["per_class_accuracy"]).to_csv(output_path, index=False)


def _per_class_accuracy(matrix, labels, class_names):
    rows = []
    for index, label in enumerate(labels):
        total = matrix[index].sum()
        correct = matrix[index, index]
        accuracy = 0.0 if total == 0 else correct / total
        rows.append(
            {
                "class_id": int(label),
                "class_name": class_names[int(label)],
                "accuracy": float(accuracy),
                "test_pixels": int(total),
            }
        )
    return rows
