import os
from PIL import Image

DATA_DIRS = [
    "../dog_classifier/data/train",
    "../dog_classifier/data/val"
]

def clean_folder(root):
    removed = 0
    for class_dir in os.listdir(root):
        class_path = os.path.join(root, class_dir)
        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            path = os.path.join(class_path, file)
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                os.remove(path)
                removed += 1
                print(f"removed: {path}")

    print(f"Done. Removed {removed} bad images from {root}")

for d in DATA_DIRS:
    clean_folder(d)
