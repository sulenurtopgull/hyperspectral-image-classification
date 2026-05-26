from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_band_image(image, output_path, title, band_index=None):
    if band_index is None:
        band_index = image.shape[2] // 2

    band = image[:, :, band_index]
    plt.figure(figsize=(7, 6))
    plt.imshow(band, cmap="gray")
    plt.title(f"{title} - Band {band_index}")
    plt.axis("off")
    plt.tight_layout()
    _save(output_path)


def save_ground_truth(ground_truth, output_path, title):
    plt.figure(figsize=(7, 6))
    plt.imshow(ground_truth, cmap="tab20")
    plt.title(f"{title} - Ground Truth")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    _save(output_path)


def save_classification_map(classification_map, output_path, title):
    plt.figure(figsize=(7, 6))
    plt.imshow(classification_map, cmap="tab20")
    plt.title(f"{title} - Classification Map")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    _save(output_path)


def save_class_distribution(distribution, output_path, title):
    class_names = [row["class_name"] for row in distribution]
    counts = [row["pixel_count"] for row in distribution]
    y_positions = np.arange(len(class_names))

    plt.figure(figsize=(10, max(5, len(class_names) * 0.35)))
    plt.barh(y_positions, counts)
    plt.yticks(y_positions, class_names)
    plt.xlabel("Labeled pixel count")
    plt.title(f"{title} - Class Distribution")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    _save(output_path)


def save_confusion_matrix(matrix, labels, output_path, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={"label": "Pixel count"},
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f"{title} - Confusion Matrix")
    plt.tight_layout()
    _save(output_path)


def _save(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
