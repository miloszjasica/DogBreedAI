import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

TRAIN_DIR = "../dog_classifier/data/train"
VAL_DIR   = "../dog_classifier/data/val"
BATCH_SIZE = 32
EPOCHS_HEAD = 5 # train only head
EPOCHS_FINE = 10 # fine-tuning backbone
LR_HEAD = 1e-3
LR_FINE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# DATA
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_ds   = datasets.ImageFolder(VAL_DIR, transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_ds.classes)
print("Classes:", num_classes)

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

# new head
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# HEAD
optimizer = optim.Adam(model.fc.parameters(), lr=LR_HEAD)

print("\nTraining classifier head")
for epoch in range(EPOCHS_HEAD):
    model.train()
    running_loss = 0
    correct = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()

    acc = correct / len(train_ds)
    print(f"[HEAD] Epoch {epoch+1}/{EPOCHS_HEAD} | Loss: {running_loss/len(train_ds):.4f} | Acc: {acc:.4f}")

# FINE TUNING
print("\nFine-tuning last ResNet layers")

for name, param in model.named_parameters():
    if "layer4" in name:
        param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINE)

for epoch in range(EPOCHS_FINE):
    model.train()
    running_loss = 0
    correct = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()

    acc = correct / len(train_ds)
    print(f"[FINE] Epoch {epoch+1}/{EPOCHS_FINE} | Loss: {running_loss/len(train_ds):.4f} | Acc: {acc:.4f}")

os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "../dog_classifier/models/dog_breed_classifier.pth")

print("\nFine-tuning complete. Model saved.")
