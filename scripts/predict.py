import torch
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torchvision.models import ResNet18_Weights
from PIL import Image
import os

MODEL_PATH = "../models/dog_breed_finetuned.pth"
IMAGE_PATH = "../examples/benek.jpg" # image to check
TRAIN_DIR = "../dog_classifier/data/train"
TOP_K = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

#Load classes
dummy_dataset = datasets.ImageFolder(TRAIN_DIR)
class_names = sorted(os.listdir(TRAIN_DIR))

#Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

#prediciton
with torch.no_grad():
    outputs = model(input_tensor)
    probs = F.softmax(outputs.cpu(), dim=1)
    top_probs, top_idxs = torch.topk(probs, TOP_K)

print("\nTOP 5 predictions:\n")

for i in range(TOP_K):
    cls = class_names[top_idxs[0][i]]
    conf = top_probs[0][i].item() * 100
    print(f"{i+1}. {cls:40s} {conf:6.2f}%")
