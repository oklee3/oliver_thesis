import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=transform)
val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=transform)
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
print("Classes:", train_dataset.classes)

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

def evaluate(loader, model, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN(num_classes=len(train_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    train_acc = evaluate(train_loader, model, device)
    val_acc = evaluate(val_loader, model, device)

    print(f"Epoch {epoch+1:2d} | "
        f"Loss: {avg_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f}")

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

test_acc = evaluate(test_loader, model, device)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

save_path = MODEL_DIR / "cnn_model_20_epochs.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")