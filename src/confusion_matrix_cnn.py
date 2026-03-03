import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix


DATA_DIR = Path("data")
MODEL_PATH = Path("models/cnn_model_20_epochs.pth")
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=transform)
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
print("Classes:", class_names)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print(f"Loaded model from {MODEL_PATH}")


def get_predictions(loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(cm, class_names, title, filename):
    fig, ax = plt.subplots(figsize=(7, 7))

    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j],
                    ha="center",
                    va="center",
                    fontsize=10)

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14)

    fig.tight_layout()

    plt.savefig(IMAGES_DIR / f"{filename}.png", dpi=300)
    plt.show()


y_train, train_preds = get_predictions(train_loader)
y_test, test_preds = get_predictions(test_loader)

cm_train = confusion_matrix(y_train, train_preds)
cm_test = confusion_matrix(y_test, test_preds)

plot_confusion_matrix(
    cm_train,
    class_names,
    "Confusion Matrix (Train Set)",
    "confusion_matrix_train_cnn"
)

plot_confusion_matrix(
    cm_test,
    class_names,
    "Confusion Matrix (Test Set)",
    "confusion_matrix_test_cnn"
)