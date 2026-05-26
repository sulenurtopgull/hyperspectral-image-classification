import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_loader import DATASETS, flatten_labeled_pixels, load_dataset
from metrics import classification_metrics, save_metrics, save_per_class_accuracy
from visualization import save_confusion_matrix


def train_random_forest(dataset_name, project_root, test_size, n_estimators, random_state):
    image, ground_truth, config = load_dataset(dataset_name, project_root / "data")
    x, y = flatten_labeled_pixels(image, ground_truth)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = classification_metrics(y_test, y_pred, config["classes"])
    metrics.update(
        {
            "dataset": config["display_name"],
            "model": "RandomForest",
            "test_size": test_size,
            "n_estimators": n_estimators,
            "random_state": random_state,
            "train_pixels": int(len(y_train)),
            "test_pixels": int(len(y_test)),
        }
    )

    prefix = f"{dataset_name}_random_forest"
    metrics_dir = project_root / "results" / "metrics"
    figures_dir = project_root / "results" / "figures"
    models_dir = project_root / "results" / "models"

    save_metrics(metrics, metrics_dir / f"{prefix}_metrics.json")
    save_per_class_accuracy(metrics, metrics_dir / f"{prefix}_per_class_accuracy.csv")
    pd.DataFrame(metrics["confusion_matrix"]).to_csv(
        metrics_dir / f"{prefix}_confusion_matrix.csv",
        index=False,
    )
    save_confusion_matrix(
        metrics["confusion_matrix"],
        metrics["labels"],
        figures_dir / f"{prefix}_confusion_matrix.png",
        f"{config['display_name']} Random Forest",
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / f"{prefix}.joblib")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train a Random Forest hyperspectral classifier.")
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    metrics = train_random_forest(
        dataset_name=args.dataset,
        project_root=args.project_root,
        test_size=args.test_size,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
    )
    print(
        f"{metrics['dataset']} Random Forest: "
        f"OA={metrics['overall_accuracy']:.4f}, "
        f"AA={metrics['average_accuracy']:.4f}, "
        f"Kappa={metrics['kappa']:.4f}"
    )


if __name__ == "__main__":
    main()
