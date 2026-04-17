"""
Evaluate each saved train-pair model against every individual test category and
render train-pair vs. test-category heatmaps.
"""
import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from train_cnn import (
    CNNClassifier,
    DATA_ROOT,
    IMAGE_DIR as CNN_IMAGE_DIR,
    MODEL_DIR as CNN_MODEL_DIR,
    PAIR_RUNS,
    collect_items as collect_cnn_items,
    evaluate as evaluate_cnn,
    list_split_classes as list_cnn_split_classes,
)
from train_mlp import (
    IMAGE_DIR as MLP_IMAGE_DIR,
    MODEL_DIR as MLP_MODEL_DIR,
    MLPClassifier,
    collect_items as collect_mlp_items,
    evaluate as evaluate_mlp,
    list_split_classes as list_mlp_split_classes,
)


def load_checkpoint(model_path: str, device: torch.device) -> Dict:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
    return torch.load(model_path, map_location=device)


def save_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    out_path: str,
    title: str,
):
    fig_width = max(10.0, 1.2 * len(col_labels) + 3.0)
    fig_height = max(5.0, 1.0 * len(row_labels) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": "Accuracy"},
        ax=ax,
    )
    ax.set_xlabel("Test Category")
    ax.set_ylabel("Train Pair")
    ax.set_title(title, pad=10)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_model_family(
    model_name: str,
    model_dir: str,
    image_dir: str,
    model_cls,
    collect_items_fn,
    evaluate_fn,
    list_split_classes_fn,
    batch_size: int,
    device: torch.device,
    run_index: int,
) -> Tuple[np.ndarray, List[Dict]]:
    pair_names = [run_name for run_name, _ in PAIR_RUNS]
    test_categories = list_split_classes_fn(os.path.join(DATA_ROOT, "test"))
    matrix = np.zeros((len(pair_names), len(test_categories)), dtype=np.float32)
    details: List[Dict] = []

    test_items_by_category = {
        class_name: collect_items_fn(os.path.join(DATA_ROOT, "test"), [class_name])
        for class_name in test_categories
    }

    for train_idx, (train_pair_name, train_classes) in enumerate(PAIR_RUNS):
        checkpoint_path = os.path.join(model_dir, f"{model_name}_{train_pair_name}_run{run_index}.pt")
        checkpoint = load_checkpoint(checkpoint_path, device)

        model = model_cls().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        for test_idx, test_category in enumerate(test_categories):
            _loss, test_acc, per_class_acc = evaluate_fn(
                model,
                test_items_by_category[test_category],
                device,
                batch_size,
            )
            matrix[train_idx, test_idx] = test_acc
            details.append(
                {
                    "model_type": model_name,
                    "run_index": run_index,
                    "train_pair": train_pair_name,
                    "train_classes": list(train_classes),
                    "test_category": test_category,
                    "test_acc": test_acc,
                    "test_per_class_acc": per_class_acc,
                    "checkpoint_path": checkpoint_path,
                }
            )

    heatmap_path = os.path.join(image_dir, f"{model_name}_outline_category_heatmap_run{run_index}.png")
    save_heatmap(
        matrix,
        row_labels=pair_names,
        col_labels=test_categories,
        out_path=heatmap_path,
        title=f"{model_name.upper()} Outline Train Pair vs Test Category Accuracy (Run {run_index})",
    )

    summary_path = os.path.join(model_dir, f"outline_category_heatmap_results_run{run_index}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": model_name,
                "run_index": run_index,
                "pair_names": pair_names,
                "test_categories": test_categories,
                "accuracy_matrix": matrix.tolist(),
                "heatmap_path": heatmap_path,
                "results": details,
            },
            f,
            indent=2,
        )

    return matrix, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--run-index", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn_matrix, _ = evaluate_model_family(
        model_name="cnn",
        model_dir=CNN_MODEL_DIR,
        image_dir=CNN_IMAGE_DIR,
        model_cls=CNNClassifier,
        collect_items_fn=collect_cnn_items,
        evaluate_fn=evaluate_cnn,
        list_split_classes_fn=list_cnn_split_classes,
        batch_size=args.batch_size,
        device=device,
        run_index=args.run_index,
    )
    mlp_matrix, _ = evaluate_model_family(
        model_name="mlp",
        model_dir=MLP_MODEL_DIR,
        image_dir=MLP_IMAGE_DIR,
        model_cls=MLPClassifier,
        collect_items_fn=collect_mlp_items,
        evaluate_fn=evaluate_mlp,
        list_split_classes_fn=list_mlp_split_classes,
        batch_size=args.batch_size,
        device=device,
        run_index=args.run_index,
    )

    print(
        json.dumps(
            {
                "cnn_accuracy_matrix": cnn_matrix.tolist(),
                "mlp_accuracy_matrix": mlp_matrix.tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()