from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from torchvision import models, transforms
from PIL import Image
import io
import json

with open("../dog_classifier/models/class_names.json") as f:
    CLASS_NAMES = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(CLASS_NAMES)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("../dog_classifier/models/dog_breed_classifier.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def format_breed(name: str) -> str:
    clean = name.split("-", 1)[1]
    clean = clean.replace("_", " ")
    clean = clean.title()
    return clean

class Prediction(BaseModel):
    breed: str
    probability: float

class PredictionResponse(BaseModel):
    predictions: list[Prediction]

app = FastAPI(title="Dog Breed Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top5_prob, top5_idx = torch.topk(probs, k=5)
    
    predictions = []
    for prob, idx in zip(top5_prob[0], top5_idx[0]):
        predictions.append(
            Prediction(
                breed=format_breed(CLASS_NAMES[idx]),
                probability=round(prob.item(), 4)
            )
        )
    
    return PredictionResponse(predictions=predictions)
