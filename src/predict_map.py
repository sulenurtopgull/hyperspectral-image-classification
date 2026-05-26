import argparse
from pathlib import Path

import joblib
import numpy as np

from data_loader import DATASETS, load_dataset
from visualization import save_classification_map


MODEL_FILES = {
    "svm_pca": "{dataset}_svm_pca.joblib",
    "random_forest": "{dataset}_random_forest.joblib",
}


def predict_classification_map(dataset_name, model_name, project_root, batch_size):
    image, ground_truth, config = load_dataset(dataset_name, project_root / "data")
    model_path = project_root / "results" / "models" / MODEL_FILES[model_name].format(dataset=dataset_name)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    height, width, bands = image.shape
    flat_pixels = image.reshape(-1, bands)
    predictions = np.zeros(flat_pixels.shape[0], dtype=np.int32)

    for start in range(0, flat_pixels.shape[0], batch_size):
        end = start + batch_size
        predictions[start:end] = model.predict(flat_pixels[start:end])

    classification_map = predictions.reshape(height, width)
    # Keep unlabeled/background pixels black to make visual comparison with ground truth easier.
    classification_map[ground_truth == 0] = 0

    output_path = (
        project_root
        / "results"
        / "figures"
        / f"{dataset_name}_{model_name}_classification_map.png"
    )
    save_classification_map(
        classification_map,
        output_path,
        f"{config['display_name']} {display_model_name(model_name)}",
    )
    return output_path


def display_model_name(model_name):
    names = {
        "svm_pca": "SVM + PCA",
        "random_forest": "Random Forest",
    }
    return names[model_name]


def main():
    parser = argparse.ArgumentParser(description="Generate a full-scene classification map.")
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--model", choices=MODEL_FILES.keys(), required=True)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    output_path = predict_classification_map(
        dataset_name=args.dataset,
        model_name=args.model,
        project_root=args.project_root,
        batch_size=args.batch_size,
    )
    print(f"Saved classification map: {output_path}")


if __name__ == "__main__":
    main()
