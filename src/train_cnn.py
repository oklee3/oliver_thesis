"""
Train CNN models for each paired category group across the configured datasets.
"""
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = (128, 128)
DATA_ROOT = "data"
MODEL_DIR = "models/cnn"
IMAGE_DIR = "images/cnn"
EXPECTED_NUM_CLASSES = 8
DATASET_CONFIGS = [
    {
        "name": "data",
        "data_root": "data",
        "model_dir": "models/cnn",
        "image_dir": "images/cnn",
        "image_root_label": "images",
        "title_label": "Standard",
    },
    {
        "name": "outline_data",
        "data_root": "outline_data",
        "model_dir": "outline_models/cnn",
        "image_dir": "outline_images/cnn",
        "image_root_label": "outline_images",
        "title_label": "Outline",
    },
]
PAIR_RUNS = [
    ("no_overlap", ["no_overlap_circle", "no_overlap_triangle"]),
    ("no_overlap_bw", ["no_overlap_circle_bw", "no_overlap_triangle_bw"]),
    ("overlap", ["overlap_circle", "overlap_triangle"]),
    ("overlap_bw", ["overlap_circle_bw", "overlap_triangle_bw"]),
]


def class_to_label(class_name: str) -> int:
    # 1: circle above, 0: triangle above
    if "_circle" in class_name:
        return 1
    if "_triangle" in class_name:
        return 0
    raise ValueError(f"Unknown class name: {class_name}")


def collect_items(
    split_root: str, class_names: Optional[Sequence[str]] = None
) -> List[Tuple[str, int, str]]:
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"Missing split folder: {split_root}")

    available_classes = sorted(
        d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d)) and not d.startswith(".")
    )
    if len(available_classes) != EXPECTED_NUM_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_NUM_CLASSES} class folders in {split_root}, found {len(available_classes)}"
        )
    selected_classes = available_classes if class_names is None else list(class_names)
    missing_classes = sorted(c for c in selected_classes if c not in available_classes)
    if missing_classes:
        raise RuntimeError(f"Missing class folders in {split_root}: {', '.join(missing_classes)}")

    items: List[Tuple[str, int, str]] = []
    for class_name in selected_classes:
        class_dir = os.path.join(split_root, class_name)
        label = class_to_label(class_name)
        files = sorted(
            f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not files:
            raise RuntimeError(f"No images found in {class_dir}")
        for fname in files:
            items.append((os.path.join(class_dir, fname), label, class_name))
    return items


def list_split_classes(split_root: str) -> List[str]:
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"Missing split folder: {split_root}")

    classes = sorted(
        d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d)) and not d.startswith(".")
    )
    if len(classes) != EXPECTED_NUM_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_NUM_CLASSES} class folders in {split_root}, found {len(classes)}"
        )
    return classes


def paired_categories(split_root: str) -> List[Tuple[str, List[str]]]:
    available_classes = set(list_split_classes(split_root))
    required_classes = {class_name for _run_name, class_names in PAIR_RUNS for class_name in class_names}
    missing_classes = sorted(required_classes - available_classes)
    if missing_classes:
        raise RuntimeError(f"Missing class folders in {split_root}: {', '.join(missing_classes)}")
    return [(run_name, list(class_names)) for run_name, class_names in PAIR_RUNS]


class ShapeDataset(Dataset):
    def __init__(self, items: Sequence[Tuple[str, int, str]], image_size: Tuple[int, int] = IMAGE_SIZE):
        self.items = list(items)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label, class_name = self.items[idx]
        img = Image.open(path).convert("RGB").resize(self.image_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        x = torch.from_numpy(arr)
        y = torch.tensor(float(label), dtype=torch.float32)
        return x, y, class_name


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x).squeeze(1)


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    seed: int


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_root: str
    model_dir: str
    image_dir: str
    image_root_label: str
    title_label: str


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_progress_interval(num_batches: int) -> int:
    return max(1, num_batches // 10)


def evaluate(
    model: nn.Module,
    items: Sequence[Tuple[str, int, str]],
    device: torch.device,
    batch_size: int,
    loss_fn: Optional[nn.Module] = None,
):
    ds = ShapeDataset(items)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    class_totals = {}
    class_correct = {}

    with torch.no_grad():
        for x, y, class_names in dl:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            if loss_fn is not None:
                total_loss += float(loss_fn(logits, y).item()) * y.numel()
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            total += y.numel()
            correct += (preds == y).sum().item()

            for i, cname in enumerate(class_names):
                class_totals[cname] = class_totals.get(cname, 0) + 1
                class_correct[cname] = class_correct.get(cname, 0) + int(preds[i].item() == y[i].item())

    per_class_acc = {
        cname: class_correct[cname] / class_totals[cname] for cname in sorted(class_totals.keys())
    }
    avg_loss = total_loss / max(total, 1)
    return avg_loss, correct / max(total, 1), per_class_acc


def save_loss_curve(
    history: Dict[str, List[float]],
    out_path: str,
    title: str,
):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
    ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2.0)
    ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2.0)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)
    ax.grid(alpha=0.25)
    ax.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def loss_curve_path_for(dataset_cfg: DatasetConfig, pair_name: str, run_index: int) -> str:
    return os.path.join(
        dataset_cfg.image_dir,
        "loss_curves",
        f"cnn_{pair_name}_run{run_index}_loss_curve.png",
    )


def combine_dataset_loss_curves(
    dataset_cfg: DatasetConfig,
    run_index: int,
    run_order: Sequence[str],
) -> str:
    curve_paths = {
        run_name: loss_curve_path_for(dataset_cfg, run_name, run_index)
        for run_name in run_order
    }
    images: Dict[str, Image.Image] = {}
    missing: List[str] = []
    for run_name, path in curve_paths.items():
        if not os.path.isfile(path):
            missing.append(path)
            continue
        images[run_name] = Image.open(path).convert("RGB")
    if missing:
        missing_list = "\n".join(missing)
        raise FileNotFoundError(f"Missing loss curve images:\n{missing_list}")

    widths = [images[run_name].width for run_name in run_order]
    heights = [images[run_name].height for run_name in run_order]
    tile_width = max(widths)
    tile_height = max(heights)
    padding = 30
    title_height = 50

    canvas = Image.new(
        "RGB",
        (tile_width * 2 + padding * 3, tile_height * 2 + padding * 3 + title_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 15), f"CNN {dataset_cfg.title_label} Loss Curves - Run {run_index}", fill="black")

    for idx, run_name in enumerate(run_order):
        row, col = divmod(idx, 2)
        x = padding + col * (tile_width + padding)
        y = title_height + padding + row * (tile_height + padding)
        image = images[run_name]
        offset_x = x + (tile_width - image.width) // 2
        offset_y = y + (tile_height - image.height) // 2
        canvas.paste(image, (offset_x, offset_y))

    out_path = os.path.join(
        dataset_cfg.image_dir,
        f"cnn_{dataset_cfg.name}_loss_curves_run{run_index}_grid.png",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    return out_path


def train_one_pair(
    dataset_cfg: DatasetConfig,
    pair_name: str,
    class_names: Sequence[str],
    cfg: TrainConfig,
    device: torch.device,
    run_index: int,
):
    print(f"[CNN][{dataset_cfg.name}][{pair_name}][run {run_index}] Starting training...")
    train_items = collect_items(os.path.join(dataset_cfg.data_root, "train"), class_names)
    val_items = collect_items(os.path.join(dataset_cfg.data_root, "val"), class_names)
    test_items = collect_items(os.path.join(dataset_cfg.data_root, "test"), class_names)

    train_dl = DataLoader(
        ShapeDataset(train_items),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    progress_interval = batch_progress_interval(len(train_dl))

    model = CNNClassifier().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_state = None
    best_val_loss = float("inf")
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        example_count = 0
        for batch_idx, (x, y, _class_name) in enumerate(train_dl, start=1):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * y.numel()
            example_count += y.numel()
            if batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == len(train_dl):
                print(
                    f"[CNN][{dataset_cfg.name}][{pair_name}][run {run_index}] "
                    f"Epoch {epoch + 1}/{cfg.epochs} "
                    f"batch {batch_idx}/{len(train_dl)} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

        avg_loss = epoch_loss / max(example_count, 1)
        val_loss, val_acc, _ = evaluate(model, val_items, device, cfg.batch_size, loss_fn=loss_fn)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        print(
            f"[CNN][{dataset_cfg.name}][{pair_name}][run {run_index}] Epoch {epoch + 1}/{cfg.epochs} "
            f"train_loss={avg_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"[CNN][{dataset_cfg.name}][{pair_name}][run {run_index}] "
                f"New best val_loss={best_val_loss:.4f}"
            )

    assert best_state is not None
    model.load_state_dict(best_state)

    loss_curve_path = loss_curve_path_for(dataset_cfg, pair_name, run_index)
    save_loss_curve(
        history,
        loss_curve_path,
        title=f"CNN {dataset_cfg.title_label} Loss Curve - {pair_name} (Run {run_index})",
    )

    test_loss, test_acc, per_class_acc = evaluate(
        model,
        test_items,
        device,
        cfg.batch_size,
        loss_fn=loss_fn,
    )

    os.makedirs(dataset_cfg.model_dir, exist_ok=True)
    model_path = os.path.join(dataset_cfg.model_dir, f"cnn_{pair_name}_run{run_index}.pt")
    torch.save(
        {
            "model_type": "CNNClassifier",
            "dataset_name": dataset_cfg.name,
            "data_root": dataset_cfg.data_root,
            "data_layout": "train_val_test_folders",
            "pair_name": pair_name,
            "run_index": run_index,
            "train_classes": list(class_names),
            "state_dict": model.state_dict(),
            "best_val_loss": best_val_loss,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "test_per_class_acc": per_class_acc,
            "loss_curve_path": loss_curve_path,
            "history": history,
            "image_size": IMAGE_SIZE,
            "label_mapping": {"triangle_above": 0, "circle_above": 1},
        },
        model_path,
    )

    return {
        "pair_name": pair_name,
        "run_index": run_index,
        "dataset_name": dataset_cfg.name,
        "data_root": dataset_cfg.data_root,
        "train_classes": list(class_names),
        "split_layout": "train/val/test",
        "train_count": len(train_items),
        "val_count": len(val_items),
        "test_count": len(test_items),
        "model_path": model_path,
        "epochs": cfg.epochs,
        "best_val_loss": best_val_loss,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "test_per_class_acc": per_class_acc,
        "loss_curve_path": loss_curve_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--start-run-index", type=int, default=1)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[cfg["name"] for cfg in DATASET_CONFIGS],
        choices=[cfg["name"] for cfg in DATASET_CONFIGS],
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_configs = [DatasetConfig(**cfg) for cfg in DATASET_CONFIGS if cfg["name"] in args.datasets]

    all_results = []
    for dataset_cfg in dataset_configs:
        os.makedirs(dataset_cfg.model_dir, exist_ok=True)
        dataset_results = []
        split_root = os.path.join(dataset_cfg.data_root, "train")
        pair_configs = paired_categories(split_root)

        for run_index in range(args.start_run_index, args.start_run_index + args.num_runs):
            run_seed = args.seed + run_index - 1
            set_seed(run_seed)
            cfg = TrainConfig(args.epochs, args.batch_size, args.lr, run_seed)
            run_results = []
            for pair_name, class_names in pair_configs:
                run_results.append(train_one_pair(dataset_cfg, pair_name, class_names, cfg, device, run_index))
            dataset_results.extend(run_results)
            all_results.extend(run_results)

            summary_path = os.path.join(dataset_cfg.model_dir, f"training_summary_run{run_index}.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(run_results, f, indent=2)

            combine_dataset_loss_curves(
                dataset_cfg,
                run_index,
                [pair_name for pair_name, _ in PAIR_RUNS],
            )

        summary_path = os.path.join(dataset_cfg.model_dir, "training_summary_all_runs.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(dataset_results, f, indent=2)

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()