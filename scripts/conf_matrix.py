import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

VAL_DIR = "../dog_classifier/data/val"
MODEL_PATH = "../dog_classifier/models/dog_breed_classifier.pth"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = val_dataset.classes
num_classes = len(class_names)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

#collect predictions
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

#plot
conf_matrix = confusion_matrix(y_true, y_pred, normalize="true")

errors = []

for i in range(num_classes):
    for j in range(num_classes):
        if i != j and conf_matrix[i, j] > 0:
            errors.append((conf_matrix[i, j], class_names[i], class_names[j]))

errors.sort(reverse=True)

print("\nTop 30 misses:")
for value, true_cls, pred_cls in errors[:30]:
    print(f"{true_cls} → {pred_cls}: {value:.4f}")
