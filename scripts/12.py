import json
from torchvision import datasets

train_dataset = datasets.ImageFolder("../dog_classifier/data/train")
class_names = train_dataset.classes

with open("../dog_classifier/models/class_names.json", "w") as f:
    json.dump(class_names, f)
