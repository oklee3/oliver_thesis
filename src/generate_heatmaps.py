"""
Evaluate each saved train-pair model against every individual test category and
render train-pair vs. test-category heatmaps.
"""
import argparse
import json
import os
import re
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from train_cnn import (
    CNNClassifier,
    DATASET_CONFIGS as CNN_DATASET_CONFIGS,
    PAIR_RUNS,
    collect_items as collect_cnn_items,
    evaluate as evaluate_cnn,
    list_split_classes as list_cnn_split_classes,
)
from train_mlp import (
    DATASET_CONFIGS as MLP_DATASET_CONFIGS,
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
    annot_labels: np.ndarray,
):
    plot_matrix = matrix.T
    plot_annot = annot_labels.T
    fig_width = max(7.0, 1.2 * len(row_labels) + 3.0)
    fig_height = max(10.0, 1.0 * len(col_labels) + 2.5)
    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
        sns.heatmap(
            plot_matrix,
            annot=plot_annot,
            fmt="",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            xticklabels=row_labels,
            yticklabels=col_labels,
            cbar_kws={"label": "Accuracy"},
            ax=ax,
        )
        ax.set_xlabel("Train Pair")
        ax.set_ylabel("Test Category")
        ax.set_title(title, pad=10)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def discover_run_indices(model_name: str, model_dir: str) -> List[int]:
    pair_run_sets: List[set[int]] = []
    for pair_name, _train_classes in PAIR_RUNS:
        pattern = re.compile(rf"^{re.escape(model_name)}_{re.escape(pair_name)}_run(\d+)\.pt$")
        pair_runs = {
            int(match.group(1))
            for fname in os.listdir(model_dir)
            for match in [pattern.match(fname)]
            if match is not None
        }
        if not pair_runs:
            raise FileNotFoundError(
                f"No checkpoints found for {model_name}_{pair_name}_runN.pt in {model_dir}"
            )
        pair_run_sets.append(pair_runs)

    run_indices = sorted(set.intersection(*pair_run_sets))
    if not run_indices:
        raise RuntimeError(f"No shared run indices found across all {model_name} checkpoints in {model_dir}")
    return run_indices


def evaluate_model_family(
    model_name: str,
    dataset_name: str,
    data_root: str,
    model_dir: str,
    image_dir: str,
    model_cls,
    collect_items_fn,
    evaluate_fn,
    list_split_classes_fn,
    batch_size: int,
    device: torch.device,
    run_indices: Sequence[int],
) -> Dict:
    pair_names = [run_name for run_name, _ in PAIR_RUNS]
    test_categories = list_split_classes_fn(os.path.join(data_root, "test"))
    run_matrices: Dict[int, np.ndarray] = {}
    details: List[Dict] = []

    test_items_by_category = {
        class_name: collect_items_fn(os.path.join(data_root, "test"), [class_name])
        for class_name in test_categories
    }

    for run_index in run_indices:
        matrix = np.zeros((len(pair_names), len(test_categories)), dtype=np.float32)
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
                        "dataset_name": dataset_name,
                        "data_root": data_root,
                        "run_index": run_index,
                        "train_pair": train_pair_name,
                        "train_classes": list(train_classes),
                        "test_category": test_category,
                        "test_acc": test_acc,
                        "test_per_class_acc": per_class_acc,
                        "checkpoint_path": checkpoint_path,
                    }
                )
        run_matrices[run_index] = matrix

    stacked_matrices = np.stack([run_matrices[run_index] for run_index in run_indices], axis=0)
    mean_matrix = stacked_matrices.mean(axis=0)
    min_matrix = stacked_matrices.min(axis=0)
    max_matrix = stacked_matrices.max(axis=0)
    range_matrix = max_matrix - min_matrix
    annot_labels = np.empty(mean_matrix.shape, dtype=object)
    for row_idx in range(mean_matrix.shape[0]):
        for col_idx in range(mean_matrix.shape[1]):
            annot_labels[row_idx, col_idx] = f"{min_matrix[row_idx, col_idx]:.3f}-{max_matrix[row_idx, col_idx]:.3f}"

    run_label = f"runs{run_indices[0]}-{run_indices[-1]}" if len(run_indices) > 1 else f"run{run_indices[0]}"
    heatmap_path = os.path.join(image_dir, f"{model_name}_category_heatmap_{run_label}.png")
    save_heatmap(
        mean_matrix,
        row_labels=pair_names,
        col_labels=test_categories,
        out_path=heatmap_path,
        title=(
            f"{model_name.upper()} {dataset_name.replace('_', ' ').title()} "
            f"Train Pair vs Test Category Accuracy Range ({run_label})"
        ),
        annot_labels=annot_labels,
    )

    summary_path = os.path.join(model_dir, f"category_heatmap_results_{run_label}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": model_name,
                "dataset_name": dataset_name,
                "data_root": data_root,
                "run_indices": list(run_indices),
                "pair_names": pair_names,
                "test_categories": test_categories,
                "mean_accuracy_matrix": mean_matrix.tolist(),
                "min_accuracy_matrix": min_matrix.tolist(),
                "max_accuracy_matrix": max_matrix.tolist(),
                "accuracy_range_matrix": range_matrix.tolist(),
                "range_labels": annot_labels.tolist(),
                "run_accuracy_matrices": {
                    str(run_index): run_matrices[run_index].tolist() for run_index in run_indices
                },
                "heatmap_path": heatmap_path,
                "results": details,
            },
            f,
            indent=2,
        )

    return {
        "run_indices": list(run_indices),
        "mean_accuracy_matrix": mean_matrix.tolist(),
        "min_accuracy_matrix": min_matrix.tolist(),
        "max_accuracy_matrix": max_matrix.tolist(),
        "accuracy_range_matrix": range_matrix.tolist(),
        "range_labels": annot_labels.tolist(),
        "heatmap_path": heatmap_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: Dict[str, Dict[str, Dict]] = {"cnn": {}, "mlp": {}}

    for dataset_cfg in CNN_DATASET_CONFIGS:
        run_indices = discover_run_indices("cnn", dataset_cfg["model_dir"])
        results["cnn"][dataset_cfg["name"]] = evaluate_model_family(
            model_name="cnn",
            dataset_name=dataset_cfg["name"],
            data_root=dataset_cfg["data_root"],
            model_dir=dataset_cfg["model_dir"],
            image_dir=dataset_cfg["image_dir"],
            model_cls=CNNClassifier,
            collect_items_fn=collect_cnn_items,
            evaluate_fn=evaluate_cnn,
            list_split_classes_fn=list_cnn_split_classes,
            batch_size=args.batch_size,
            device=device,
            run_indices=run_indices,
        )

    for dataset_cfg in MLP_DATASET_CONFIGS:
        run_indices = discover_run_indices("mlp", dataset_cfg["model_dir"])
        results["mlp"][dataset_cfg["name"]] = evaluate_model_family(
            model_name="mlp",
            dataset_name=dataset_cfg["name"],
            data_root=dataset_cfg["data_root"],
            model_dir=dataset_cfg["model_dir"],
            image_dir=dataset_cfg["image_dir"],
            model_cls=MLPClassifier,
            collect_items_fn=collect_mlp_items,
            evaluate_fn=evaluate_mlp,
            list_split_classes_fn=list_mlp_split_classes,
            batch_size=args.batch_size,
            device=device,
            run_indices=run_indices,
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()