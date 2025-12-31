# Overview
This project allows you to:
- Train a ResNet18 model on dog breed images
- Fine-tune the last layers for better accuracy
- Evaluate the model with confusion matrix and top misclassifications
- Visualize Grad-CAM heatmaps to see which parts of the image influence predictions
- Deploy a FastAPI backend for predictions
- Connect a simple React frontend for uploading dog images and displaying results

# Demo


# Dataset
1. Dataset used: https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset
2. To expand the dataset, you can follow these 3 optional steps:

| Script | Description |
|------|--------|
| `/scripts/data_scrapper_win_chrome.py` | Extra web scraping to expand dataset |
| `/scripts/data_scrapped_clean.py` | Cleaning scraped data |
| `/scripts/data_load.py` | Converting raw dataset into `train/val` format (60/40 split) |

# Model training

Framework: **PyTorch**  
Base model: **ResNet18 pretrained (ImageNet weights)**

# Evaluation

Confusion matrix for 120 breeds: helps visualize misclassifications
Top 10 misclassified breeds:

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

# API

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
# Frontend

Built with React
### You can:
- Drag & drop or file upload
- Shows uploaded image
- Displays top 5 predicted breeds with probabilities
