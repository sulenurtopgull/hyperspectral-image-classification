import argparse
import json
from pathlib import Path

import pandas as pd

from data_loader import DATASETS, class_distribution, load_dataset
from visualization import save_band_image, save_class_distribution, save_ground_truth


def explore_dataset(dataset_name, project_root):
    image, ground_truth, config = load_dataset(dataset_name, project_root / "data")
    distribution = class_distribution(ground_truth, config["classes"])

    summary = {
        "dataset": config["display_name"],
        "image_shape": list(image.shape),
        "ground_truth_shape": list(ground_truth.shape),
        "spectral_bands": int(image.shape[2]),
        "labeled_pixels": int((ground_truth > 0).sum()),
        "num_classes": len(distribution),
    }

    metrics_dir = project_root / "results" / "metrics"
    figures_dir = project_root / "results" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    prefix = dataset_name.lower()
    pd.DataFrame(distribution).to_csv(metrics_dir / f"{prefix}_class_distribution.csv", index=False)
    with open(metrics_dir / f"{prefix}_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    save_band_image(image, figures_dir / f"{prefix}_sample_band.png", config["display_name"])
    save_ground_truth(ground_truth, figures_dir / f"{prefix}_ground_truth.png", config["display_name"])
    save_class_distribution(
        distribution,
        figures_dir / f"{prefix}_class_distribution.png",
        config["display_name"],
    )

    return summary


def main():
    parser = argparse.ArgumentParser(description="Explore hyperspectral benchmark datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS.keys()],
        default="all",
        help="Dataset to explore.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root directory.",
    )
    args = parser.parse_args()

    dataset_names = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    summaries = [explore_dataset(name, args.project_root) for name in dataset_names]

    for summary in summaries:
        print(
            f"{summary['dataset']}: shape={summary['image_shape']}, "
            f"classes={summary['num_classes']}, labeled_pixels={summary['labeled_pixels']}"
        )


if __name__ == "__main__":
    main()
