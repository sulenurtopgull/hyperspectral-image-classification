import argparse
import json
from pathlib import Path

import pandas as pd


def load_experiment_metrics(metrics_dir):
    rows = []
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        with open(path, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        if "model" not in metrics:
            continue

        rows.append(
            {
                "dataset": metrics["dataset"],
                "model": _display_model_name(metrics),
                "overall_accuracy": metrics["overall_accuracy"],
                "average_accuracy": metrics["average_accuracy"],
                "kappa": metrics["kappa"],
                "train_pixels": metrics["train_pixels"],
                "test_pixels": metrics["test_pixels"],
            }
        )

    if not rows:
        raise RuntimeError(f"No experiment metric files found in {metrics_dir}")

    return pd.DataFrame(rows).sort_values(["dataset", "overall_accuracy"], ascending=[True, False])


def save_markdown_summary(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["# Experiment Comparison", ""]
    lines.append(results.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Best Model Per Dataset")
    lines.append("")

    for dataset, group in results.groupby("dataset"):
        best = group.sort_values("overall_accuracy", ascending=False).iloc[0]
        lines.append(
            f"- {dataset}: {best['model']} "
            f"(OA={best['overall_accuracy']:.4f}, AA={best['average_accuracy']:.4f}, Kappa={best['kappa']:.4f})"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _display_model_name(metrics):
    if metrics["model"] == "SVM_RBF":
        return "SVM + PCA"
    return metrics["model"]


def main():
    parser = argparse.ArgumentParser(description="Compare completed experiment metrics.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    metrics_dir = args.project_root / "results" / "metrics"
    results = load_experiment_metrics(metrics_dir)

    csv_path = metrics_dir / "experiment_comparison.csv"
    markdown_path = metrics_dir / "experiment_comparison.md"
    results.to_csv(csv_path, index=False)
    save_markdown_summary(results, markdown_path)

    print(results.to_string(index=False))
    print(f"Saved: {csv_path}")
    print(f"Saved: {markdown_path}")


if __name__ == "__main__":
    main()
