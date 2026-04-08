"""
Train one MLP per paired category group found under data/train.
"""
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = (128, 128)
DATA_ROOT = "outline_data"
MODEL_DIR = "outline_models/mlp"
IMAGE_DIR = "outline_images/mlp"
EXPECTED_NUM_CLASSES = 8
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


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int = 3 * 128 * 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x).squeeze(1)


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    seed: int


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def train_one_pair(run_name: str, class_names: Sequence[str], cfg: TrainConfig, device: torch.device):
    print(f"[MLP][{run_name}] Starting training...")
    train_items = collect_items(os.path.join(DATA_ROOT, "train"), class_names)
    val_items = collect_items(os.path.join(DATA_ROOT, "val"), class_names)
    test_items = collect_items(os.path.join(DATA_ROOT, "test"), class_names)

    train_dl = DataLoader(
        ShapeDataset(train_items),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    model = MLPClassifier().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_state = None
    best_val_loss = float("inf")
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        example_count = 0
        for x, y, _class_name in train_dl:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * y.numel()
            example_count += y.numel()

        avg_loss = epoch_loss / max(example_count, 1)
        val_loss, val_acc, _ = evaluate(model, val_items, device, cfg.batch_size, loss_fn=loss_fn)
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        print(
            f"[MLP][{run_name}] Epoch {epoch + 1}/{cfg.epochs} "
            f"train_loss={avg_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[MLP][{run_name}] New best val_loss={best_val_loss:.4f}")

    assert best_state is not None
    model.load_state_dict(best_state)

    loss_curve_path = os.path.join(IMAGE_DIR, "loss_curves", f"mlp_{run_name}_outline_loss_curve.png")
    save_loss_curve(history, loss_curve_path, title=f"MLP Outline Loss Curve - {run_name}")

    test_loss, test_acc, per_class_acc = evaluate(
        model,
        test_items,
        device,
        cfg.batch_size,
        loss_fn=loss_fn,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"mlp_{run_name}.pt")
    torch.save(
        {
            "model_type": "MLPClassifier",
            "data_layout": "train_val_test_folders",
            "pair_name": run_name,
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
        "pair_name": run_name,
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
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(args.epochs, args.batch_size, args.lr, args.seed)

    results = []
    for run_name, class_names in paired_categories(os.path.join(DATA_ROOT, "train")):
        results.append(train_one_pair(run_name, class_names, cfg, device))

    os.makedirs(MODEL_DIR, exist_ok=True)
    summary_path = os.path.join(MODEL_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
