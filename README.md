# *Dog Breed AI – ML + FastAPI + React*

## Overview
This project includes:

- Pretrained ResNet18 classifier for 120 dog breeds
- End to end system: training -> evaluation -> Grad-CAM explainability -> API -> web UI
- Reproducible scripts for dataset building, fine-tuning & inference

## Dataset
1. Source: https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset
2. Dataset expansion using scraping:

```bash
python scripts/data_scrapper_win_chrome.py      # scrape new images
python scripts/data_scrapped_clean.py           # clean incorrect links / duplicates
python scripts/data_load.py                     # convert dataset to train/val folders
```

## Demo

## Model training

Framework: **PyTorch**  
Base model: **ResNet18 pretrained (ImageNet weights)**

```bash
python scripts/train_model.py
python scripts/finetuning.py
```

## Evaluation

Confusion matrix for 120 breeds: helps visualize misclassifications

```bash
python scripts/conf_matrix.py
```

- Top 10 misclassified breeds:

```bash
Eskimo_dog → Samoyed: 0.3276

Eskimo_dog → Siberian_husky: 0.3276

kuvasz → Great_Pyrenees: 0.3158

miniature_poodle → toy_poodle: 0.3103

collie → Shetland_sheepdog: 0.2931

Lhasa → Shih-Tzu: 0.2923

Greater_Swiss_Mountain_dog → EntleBucher: 0.2833

American_Staffordshire_terrier → Staffordshire_bullterrier: 0.2500

miniature_poodle → standard_poodle: 0.2414

standard_schnauzer → miniature_schnauzer: 0.2373
```

- This analysis helps identify hard examples and potential data issues

# Grad-CAM Visualization
Grad-CAM highlights areas of the image the model focuses on when predicting:

- Useful for understanding model attention
- Can verify where model is looking

```bash
python scripts/gradcam.py
```

## API

FastAPI backend exposes a /predict endpoint:
``` bash
uvicorn scripts.app:app --reload
```
Request example:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@bernanski.jpeg;type=image/jpeg'
  ```
Response example:
```bash
{
  "predictions": [
    {
      "breed": "Bernese Mountain Dog",
      "probability": 0.6122
    },
    {
      "breed": "Gordon Setter",
      "probability": 0.1239
    },
    {
      "breed": "Greater Swiss Mountain Dog",
      "probability": 0.0364
    },
    {
      "breed": "Tibetan Mastiff",
      "probability": 0.036
    },
    {
      "breed": "Appenzeller",
      "probability": 0.0248
    }
  ]
}
```

