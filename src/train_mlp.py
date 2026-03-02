import os
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import joblib

# ------------------------
# Config
# ------------------------
DATA_DIR = Path("data")
IMAGE_SIZE = (128, 128)  # your image size
EPOCH_LIST = [5, 10, 20, 50]
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ------------------------
# Function: Load Dataset
# ------------------------
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

            img_array = np.array(img) / 255.0  # normalize
            img_flat = img_array.flatten()     # IMPORTANT for sklearn MLP

            X.append(img_flat)
            y.append(label)

    return np.array(X), np.array(y), class_to_idx

# ------------------------
# Load Data
# ------------------------
X_train, y_train, class_map = load_split(DATA_DIR / "train")
X_val, y_val, _ = load_split(DATA_DIR / "val")
X_test, y_test, _ = load_split(DATA_DIR / "test")

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
print("Test shape:", X_test.shape)

# ------------------------
# Train at Different Epochs
# ------------------------
results = {}

for epochs in EPOCH_LIST:
    print(f"\n===== Training MLP (sklearn) for {epochs} epochs =====")

    model = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        solver="adam",
        max_iter=epochs,
        random_state=42,
        verbose=True
    )

    model.fit(X_train, y_train)

    # SAVE THE MODEL (NEW LINE)
    model_path = MODEL_DIR / f"mlp_sklearn_{epochs}_epochs.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")

    # Validation accuracy
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)

    # Test accuracy
    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)

    results[epochs] = {
        "val_acc": val_acc,
        "test_acc": test_acc
    }

    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

# ------------------------
# Final Results
# ------------------------
print("\n===== Final Results =====")
for ep, metrics in results.items():
    print(f"Epochs: {ep} | Val Acc: {metrics['val_acc']:.4f} | Test Acc: {metrics['test_acc']:.4f}")
