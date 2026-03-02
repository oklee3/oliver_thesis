import numpy as np
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

# load the model and data
model = joblib.load(MODEL_PATH)
X_train, y_train, class_names = load_split(DATA_DIR / "train")
X_test, y_test, _ = load_split(DATA_DIR / "test")

# create predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# generate train and test confusion matrices
cm_train = confusion_matrix(y_train, train_preds)
cm_test = confusion_matrix(y_test, test_preds)

print("\nTRAIN CONFUSION MATRIX:")
print(cm_train)

print("\nTEST CONFUSION MATRIX:")
print(cm_test)


print("\nPER-CLASS ACCURACY (TRAIN):")
for i, class_name in enumerate(class_names):
    class_acc = cm_train[i, i] / cm_train[i].sum()
    print(f"{class_name}: {class_acc:.4f}")

print("\nPER-CLASS ACCURACY (TEST):")
for i, class_name in enumerate(class_names):
    class_acc = cm_test[i, i] / cm_test[i].sum()
    print(f"{class_name}: {class_acc:.4f}")