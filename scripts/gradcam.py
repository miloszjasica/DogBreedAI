import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

MODEL_PATH = "../models/dog_breed_finetuned.pth"
IMAGE_PATH = "../examples/benek.jpg" # image to check
NUM_CLASSES = 120
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

#hooks
gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def forward_hook(module, input, output):
    global activations
    activations = output

target_layer = model.layer4[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)

#forward/backward
output = model(input_tensor)
class_idx = output.argmax(dim=1).item()

model.zero_grad(set_to_none=True)
output[0, class_idx].backward()

#grad-cam
weights = gradients.mean(dim=(2, 3), keepdim=True)
cam = (weights * activations).sum(dim=1)
cam = F.relu(cam)
cam = cam.squeeze().cpu().detach().numpy()
cam = cv2.resize(cam, img.size)
cam = (cam - cam.min()) / (cam.max() - cam.min())

img_np = np.array(img)
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

plt.figure(figsize=(8, 8))
plt.imshow(overlay)
plt.axis("off")
plt.title("Grad-CAM")
plt.show()
