import argparse
import os
from typing import Dict, List, Sequence, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_cnn import CNNClassifier
from train_mlp import MLPClassifier

IMAGE_SIZE = (128, 128)
DATA_ROOT = "data"
MODEL_ROOT = "models"
IMAGE_ROOT = "images"
DEFAULT_BATCH_SIZE = 64

SCENARIOS = [
    {
        "name": "no_overlap",
        "test_classes": [
            "no_overlap_circle",
            "no_overlap_triangle",
            "no_overlap_circle_bw",
            "no_overlap_triangle_bw",
        ],
    },
    {
        "name": "overlap",
        "test_classes": [
            "overlap_circle",
            "overlap_triangle",
            "overlap_circle_bw",
            "overlap_triangle_bw",
        ],
    },
    {
        "name": "no_overlap_to_overlap",
        "test_classes": [
            "overlap_circle",
            "overlap_triangle",
            "overlap_circle_bw",
            "overlap_triangle_bw",
        ],
    },
    {
        "name": "overlap_to_no_overlap",
        "test_classes": [
            "no_overlap_circle",
            "no_overlap_triangle",
            "no_overlap_circle_bw",
            "no_overlap_triangle_bw",
        ],
    },
]

ALL_TEST_CLASSES = [
    "no_overlap_circle",
    "no_overlap_triangle",
    "no_overlap_circle_bw",
    "no_overlap_triangle_bw",
    "overlap_circle",
    "overlap_triangle",
    "overlap_circle_bw",
    "overlap_triangle_bw",
]


def class_to_label(class_name: str) -> int:
    if "_circle" in class_name:
        return 1
    if "_triangle" in class_name:
        return 0
    raise ValueError(f"Unknown class name: {class_name}")


def base_category_name(class_name: str) -> str:
    if class_name.endswith("_circle_bw"):
        return class_name[: -len("_circle_bw")] + "_bw"
    if class_name.endswith("_triangle_bw"):
        return class_name[: -len("_triangle_bw")] + "_bw"
    if class_name.endswith("_circle"):
        return class_name[: -len("_circle")]
    if class_name.endswith("_triangle"):
        return class_name[: -len("_triangle")]
    raise ValueError(f"Unknown class name: {class_name}")


def collect_items(
    split_root: str, class_names: Sequence[str]
) -> List[Tuple[str, int, str]]:
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"Missing split folder: {split_root}")

    items: List[Tuple[str, int, str]] = []
    for class_name in class_names:
        class_dir = os.path.join(split_root, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        label = class_to_label(class_name)
        files = sorted(
            f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not files:
            raise RuntimeError(f"No images found in {class_dir}")

        for fname in files:
            items.append((os.path.join(class_dir, fname), label, class_name))
    return items


def load_shape_dataset_class(model_type: str):
    if model_type == "cnn":
        from train_cnn import ShapeDataset

        return ShapeDataset
    if model_type == "mlp":
        from train_mlp import ShapeDataset

        return ShapeDataset
    raise ValueError(f"Unsupported model type: {model_type}")


def load_model(model_type: str, scenario_name: str, device: torch.device):
    model_dir = os.path.join(MODEL_ROOT, model_type)
    model_path = os.path.join(model_dir, f"{model_type}_{scenario_name}.pt")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    if model_type == "cnn":
        model = CNNClassifier()
    elif model_type == "mlp":
        model = MLPClassifier()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, model_path


def summarize_predictions_by_category(
    model: torch.nn.Module,
    model_type: str,
    items: Sequence[Tuple[str, int, str]],
    category_order: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, List[str], List[int]]:
    shape_dataset = load_shape_dataset_class(model_type)
    dataset = shape_dataset(items, image_size=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    category_counts: Dict[Tuple[str, int], np.ndarray] = {}

    with torch.no_grad():
        for x, y, class_names in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int().cpu().numpy()
            labels = y.int().cpu().numpy()

            for idx, class_name in enumerate(class_names):
                label = int(labels[idx])
                category_name = base_category_name(class_name)
                key = (category_name, label)
                if key not in category_counts:
                    category_counts[key] = np.zeros(2, dtype=np.int64)
                category_counts[key][int(preds[idx])] += 1

    ordered_base_categories: List[str] = []
    for class_name in category_order:
        category_name = base_category_name(class_name)
        if category_name not in ordered_base_categories:
            ordered_base_categories.append(category_name)

    row_keys = [
        (category_name, true_label)
        for category_name in ordered_base_categories
        for true_label in (0, 1)
        if (category_name, true_label) in category_counts
    ]
    counts = np.stack([category_counts[key] for key in row_keys], axis=0)
    row_categories = [category_name for category_name, _ in row_keys]
    true_labels = [true_label for _, true_label in row_keys]
    return counts, row_categories, true_labels


def format_category_label(
    category_name: str,
    true_label: int,
    total_predictions: int,
    show_category_name: bool,
) -> str:
    pretty_name = category_name.replace("_", " ").upper()
    true_text = "Actual: Triangle Above (0)" if true_label == 0 else "Actual: Circle Above (1)"
    category_text = pretty_name if show_category_name else ""
    prefix = f"{category_text}\n" if category_text else ""
    return f"{prefix}{true_text} | n={total_predictions}"


def save_category_heatmap(
    counts: np.ndarray,
    category_names: Sequence[str],
    true_labels: Sequence[int],
    out_path: str,
    title: str,
):
    row_sums = counts.sum(axis=1, keepdims=True)
    col_sums = counts.sum(axis=0)
    percentages = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts, dtype=float),
        where=row_sums != 0,
    ) * 100.0

    fig_height = max(4.8, 1.0 * len(category_names) + 2.2)
    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(figsize=(9.2, fig_height), dpi=180)
        ax.grid(False)
        im = ax.imshow(percentages, cmap="Blues", vmin=0.0, vmax=100.0)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Row-wise Percentage", fontsize=10)
        cbar.ax.grid(False)

        x_labels = [
            f"Triangle Above (0)\nTotal preds: {col_sums[0]}",
            f"Circle Above (1)\nTotal preds: {col_sums[1]}",
        ]
        y_labels = []
        for row_idx, (category_name, true_label, row_total) in enumerate(
            zip(category_names, true_labels, row_sums.ravel())
        ):
            show_category_name = row_idx == 0 or category_names[row_idx - 1] != category_name
            y_labels.append(
                format_category_label(
                    category_name,
                    true_label,
                    int(row_total),
                    show_category_name=show_category_name,
                )
            )

        ax.set_xticks(np.arange(2), labels=x_labels)
        ax.set_yticks(np.arange(len(category_names)), labels=y_labels)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("Test Category", fontsize=10)
        ax.set_title(
            f"{title}\nPrediction accuracy across each test category",
            fontsize=12,
            pad=10,
        )
        ax.tick_params(axis="y", labelsize=8.5, pad=8)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xticks([], minor=True)
        ax.set_yticks([], minor=True)

        for row_idx in range(1, len(category_names)):
            if category_names[row_idx] != category_names[row_idx - 1]:
                ax.axhline(row_idx - 0.5, color="#666666", linewidth=1.2)

        for row_idx in range(counts.shape[0]):
            for col_idx in range(counts.shape[1]):
                is_correct = col_idx == int(true_labels[row_idx])
                if is_correct:
                    ax.add_patch(
                        Rectangle(
                            (col_idx - 0.5, row_idx - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            edgecolor="#111111",
                            linewidth=2.2,
                        )
                    )
                ax.text(
                    col_idx,
                    row_idx,
                    (
                        f"{'Correct' if is_correct else 'Wrong'}\n"
                        f"{counts[row_idx, col_idx]}\n"
                        f"{percentages[row_idx, col_idx]:.1f}%"
                    ),
                    ha="center",
                    va="center",
                    color=("white" if percentages[row_idx, col_idx] >= 50 else "#0b172a"),
                    fontsize=8.5,
                    fontweight="semibold",
                )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def generate_heatmaps(model_type: str, batch_size: int, device: torch.device):
    test_root = os.path.join(DATA_ROOT, "test")
    results = []

    for scenario in SCENARIOS:
        model, model_path = load_model(model_type, scenario["name"], device)
        items = collect_items(test_root, ALL_TEST_CLASSES)
        counts, category_names, true_labels = summarize_predictions_by_category(
            model=model,
            model_type=model_type,
            items=items,
            category_order=ALL_TEST_CLASSES,
            device=device,
            batch_size=batch_size,
        )

        out_path = os.path.join(
            IMAGE_ROOT,
            model_type,
            "category_heatmaps",
            f"{model_type}_{scenario['name']}_test_category_heatmap.png",
        )
        save_category_heatmap(
            counts=counts,
            category_names=category_names,
            true_labels=true_labels,
            out_path=out_path,
            title=f"{model_type.upper()} All Test Categories - {scenario['name']}",
        )

        results.append(
            {
                "scenario": scenario["name"],
                "model_path": model_path,
                "heatmap_path": out_path,
                "test_categories": list(category_names),
            }
        )
        print(f"[{model_type.upper()}][{scenario['name']}] Saved {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=["cnn", "mlp", "all"],
        default="all",
        help="Which saved model family to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_types = ["cnn", "mlp"] if args.model_type == "all" else [args.model_type]

    for model_type in model_types:
        generate_heatmaps(
            model_type=model_type,
            batch_size=args.batch_size,
            device=device,
        )


if __name__ == "__main__":
    main()
