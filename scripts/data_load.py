import os
import shutil
from glob import glob
import random

original_data_dir = "../dog_classifier/data/raw/" #my path to original dataset
base_dir = "../dog_classifier/data/"

train_dir = os.path.join(base_dir, "train")
val_dir   = os.path.join(base_dir, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Params
train_ratio = 0.8

for breed_folder in os.listdir(original_data_dir):
    breed_path = os.path.join(original_data_dir, breed_folder)
    if not os.path.isdir(breed_path):
        continue
    
    os.makedirs(os.path.join(train_dir, breed_folder), exist_ok=True)
    os.makedirs(os.path.join(val_dir, breed_folder), exist_ok=True)
    
    images = glob(os.path.join(breed_path, "*.jpg")) + glob(os.path.join(breed_path, "*.png"))
    random.shuffle(images)
    
    split_idx = int(len(images) * train_ratio)
    
    train_images = images[:split_idx]
    val_images   = images[split_idx:]
    
    for img_path in train_images:
        shutil.copy(img_path, os.path.join(train_dir, breed_folder))
    for img_path in val_images:
        shutil.copy(img_path, os.path.join(val_dir, breed_folder))

print("Data load succeded")
