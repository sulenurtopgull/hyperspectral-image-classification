from pathlib import Path

import numpy as np
from scipy.io import loadmat


DATASETS = {
    "pavia": {
        "image_file": "PaviaU.mat",
        "gt_file": "PaviaU_gt.mat",
        "image_key": "paviaU",
        "gt_key": "paviaU_gt",
        "display_name": "Pavia University",
        "classes": {
            1: "Asphalt",
            2: "Meadows",
            3: "Gravel",
            4: "Trees",
            5: "Painted metal sheets",
            6: "Bare Soil",
            7: "Bitumen",
            8: "Self-Blocking Bricks",
            9: "Shadows",
        },
    },
    "salinas": {
        "image_file": "Salinas_corrected.mat",
        "gt_file": "Salinas_gt.mat",
        "image_key": "salinas_corrected",
        "gt_key": "salinas_gt",
        "display_name": "Salinas",
        "classes": {
            1: "Brocoli_green_weeds_1",
            2: "Brocoli_green_weeds_2",
            3: "Fallow",
            4: "Fallow_rough_plow",
            5: "Fallow_smooth",
            6: "Stubble",
            7: "Celery",
            8: "Grapes_untrained",
            9: "Soil_vinyard_develop",
            10: "Corn_senesced_green_weeds",
            11: "Lettuce_romaine_4wk",
            12: "Lettuce_romaine_5wk",
            13: "Lettuce_romaine_6wk",
            14: "Lettuce_romaine_7wk",
            15: "Vinyard_untrained",
            16: "Vinyard_vertical_trellis",
        },
    },
}


def load_dataset(dataset_name, data_dir):
    if dataset_name not in DATASETS:
        valid_names = ", ".join(DATASETS)
        raise ValueError(f"Unknown dataset '{dataset_name}'. Valid names: {valid_names}")

    config = DATASETS[dataset_name]
    data_path = Path(data_dir)
    image_mat = loadmat(data_path / config["image_file"])
    gt_mat = loadmat(data_path / config["gt_file"])

    image = image_mat[config["image_key"]]
    ground_truth = gt_mat[config["gt_key"]]

    return image.astype(np.float32), ground_truth.astype(np.int32), config


def flatten_labeled_pixels(image, ground_truth):
    mask = ground_truth > 0
    x = image[mask]
    y = ground_truth[mask]
    return x, y


def class_distribution(ground_truth, classes):
    labels, counts = np.unique(ground_truth[ground_truth > 0], return_counts=True)
    rows = []
    for label, count in zip(labels, counts):
        rows.append(
            {
                "class_id": int(label),
                "class_name": classes.get(int(label), f"Class {int(label)}"),
                "pixel_count": int(count),
            }
        )
    return rows
