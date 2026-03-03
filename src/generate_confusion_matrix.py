import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import joblib
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR = Path("data")
MODEL_PATH = Path("models/mlp_sklearn_50_epochs.joblib") # change target model here
IMAGE_SIZE = (128, 128)

# load same dataset as in train_mlp
def load_split(split_path):
    X = []
    y = []
    class_names = sorted([d.name for d in split_path.iterdir() if d.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    print(f"Loading {split_path}...")
    print("Class mapping:", class_to_idx)

    for class_name in class_names:
        class_dir = split_path / class_name
        label = class_to_idx[class_name]

        for img_path in class_dir.glob("*.png"):
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMAGE_SIZE)

            img_array = np.array(img) / 255.0
            img_flat = img_array.flatten()

            X.append(img_flat)
            y.append(label)

    return np.array(X), np.array(y), class_to_idx

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

    # save image
    images_dir = Path("images")
    plt.savefig(images_dir / f"{filename}.png", dpi=300)
    plt.show()

# load the model and data
model = joblib.load(MODEL_PATH)
X_train, y_train, class_names = load_split(DATA_DIR / "train")
X_test, y_test, _ = load_split(DATA_DIR / "test")

# create predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# generate confusion matrices
cm_train = confusion_matrix(y_train, train_preds)
cm_test = confusion_matrix(y_test, test_preds)

plot_confusion_matrix(cm_train, class_names,
                      "Confusion Matrix (Train Set)",
                      "confusion_matrix_train")

plot_confusion_matrix(cm_test, class_names,
                      "Confusion Matrix (Test Set)",
                      "confusion_matrix_test")