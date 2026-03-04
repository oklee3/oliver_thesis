'''
4 Models are trained:

[no_overlap] is trained on no_overlap_circle and no_overlap_triangle, and tested on the no_overlap_circle, 
no_overlap_triangle, no_overlap_circle_bw, and no_overlap_triangle_bw.

[overlap] is trained on overlap_circle and overlap_triangle, and tested on overlap_circle, 
overlap_triangle, overlap_circle_bw, and overlap_triangle_bw. 

[no_overlap_to_overlap] is trained on no_overlap_circle and no_overlap_triangle, and tested on overlap_circle, 
overlap_triangle, overlap_circle_bw, and overlap_triangle_bw. 

[overlap_to_no_overlap] is trained on overlap_circle and overlap_triangle, and tested on the no_overlap_circle, 
no_overlap_triangle, no_overlap_circle_bw, and no_overlap_triangle_bw.
'''
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = (128, 128)
DATA_ROOT = "data"
MODEL_DIR = "models/mlp"
IMAGE_DIR = "images/mlp"
EXPECTED_NUM_CLASSES = 8

SCENARIOS = [
    {
        "name": "no_overlap",
        "train_classes": ["no_overlap_circle", "no_overlap_triangle"],
        "test_classes": [
            "no_overlap_circle",
            "no_overlap_triangle",
            "no_overlap_circle_bw",
            "no_overlap_triangle_bw",
        ],
    },
    {
        "name": "overlap",
        "train_classes": ["overlap_circle", "overlap_triangle"],
        "test_classes": [
            "overlap_circle",
            "overlap_triangle",
            "overlap_circle_bw",
            "overlap_triangle_bw",
        ],
    },
    {
        "name": "no_overlap_to_overlap",
        "train_classes": ["no_overlap_circle", "no_overlap_triangle"],
        "test_classes": [
            "overlap_circle",
            "overlap_triangle",
            "overlap_circle_bw",
            "overlap_triangle_bw",
        ],
    },
    {
        "name": "overlap_to_no_overlap",
        "train_classes": ["overlap_circle", "overlap_triangle"],
        "test_classes": [
            "no_overlap_circle",
            "no_overlap_triangle",
            "no_overlap_circle_bw",
            "no_overlap_triangle_bw",
        ],
    },
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
    cm_path: Optional[str] = None,
    title: Optional[str] = None,
):
    ds = ShapeDataset(items)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    total = 0
    correct = 0
    class_totals = {}
    class_correct = {}
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    with torch.no_grad():
        for x, y, class_names in dl:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            total += y.numel()
            correct += (preds == y).sum().item()
            y_true_all.extend(y.int().cpu().tolist())
            y_pred_all.extend(preds.int().cpu().tolist())

            for i, cname in enumerate(class_names):
                class_totals[cname] = class_totals.get(cname, 0) + 1
                class_correct[cname] = class_correct.get(cname, 0) + int(preds[i].item() == y[i].item())

    per_class_acc = {
        cname: class_correct[cname] / class_totals[cname] for cname in sorted(class_totals.keys())
    }
    if cm_path:
        save_confusion_matrix(y_true_all, y_pred_all, cm_path, title)
    return correct / max(total, 1), per_class_acc


def save_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    out_path: str,
    title: Optional[str] = None,
):
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0
    overall_acc = (float(np.trace(cm)) / float(cm.sum()) * 100.0) if cm.sum() else 0.0

    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=180)
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0.0, vmax=100.0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-wise Percentage", fontsize=10)

    class_labels = ["Triangle Above (0)", "Circle Above (1)"]
    ax.set_xticks([0, 1], labels=class_labels)
    ax.set_yticks([0, 1], labels=class_labels)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    header = title if title else "Confusion Matrix"
    ax.set_title(f"{header}\nOverall Accuracy: {overall_acc:.2f}%", fontsize=12, pad=10)

    for i in range(2):
        for j in range(2):
            text = f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=("white" if cm_pct[i, j] >= 50 else "#0b172a"),
                fontsize=10,
                fontweight="semibold",
            )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def train_one_scenario(scenario: dict, cfg: TrainConfig, device: torch.device):
    print(f"[MLP][{scenario['name']}] Starting training...")
    train_items = collect_items(os.path.join(DATA_ROOT, "train"), scenario["train_classes"])
    val_items = collect_items(os.path.join(DATA_ROOT, "val"), scenario["train_classes"])
    test_items = collect_items(os.path.join(DATA_ROOT, "test"), scenario["test_classes"])

    train_dl = DataLoader(
        ShapeDataset(train_items),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    model = MLPClassifier().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_state = None
    best_val_acc = -1.0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for x, y, _class_name in train_dl:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batch_count += 1

        val_acc, _ = evaluate(model, val_items, device, cfg.batch_size)
        avg_loss = epoch_loss / max(batch_count, 1)
        print(
            f"[MLP][{scenario['name']}] Epoch {epoch + 1}/{cfg.epochs} "
            f"loss={avg_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[MLP][{scenario['name']}] New best val_acc={best_val_acc:.4f}")

    assert best_state is not None
    model.load_state_dict(best_state)

    cm_path = os.path.join(IMAGE_DIR, "confusion_matrices", f"mlp_{scenario['name']}_test_cm.png")
    test_acc, per_class_acc = evaluate(
        model,
        test_items,
        device,
        cfg.batch_size,
        cm_path=cm_path,
        title=f"MLP Test - {scenario['name']}",
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"mlp_{scenario['name']}.pt")
    torch.save(
        {
            "model_type": "MLPClassifier",
            "data_layout": "train_val_test_folders",
            "scenario": scenario,
            "state_dict": model.state_dict(),
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "test_per_class_acc": per_class_acc,
            "confusion_matrix_path": cm_path,
            "image_size": IMAGE_SIZE,
            "label_mapping": {"triangle_above": 0, "circle_above": 1},
        },
        model_path,
    )

    return {
        "scenario": scenario["name"],
        "split_layout": "train/val/test",
        "train_count": len(train_items),
        "val_count": len(val_items),
        "test_count": len(test_items),
        "model_path": model_path,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_per_class_acc": per_class_acc,
        "confusion_matrix_path": cm_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig(args.epochs, args.batch_size, args.lr, args.seed)

    results = []
    for scenario in SCENARIOS:
        results.append(train_one_scenario(scenario, cfg, device))

    os.makedirs(MODEL_DIR, exist_ok=True)
    summary_path = os.path.join(MODEL_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
