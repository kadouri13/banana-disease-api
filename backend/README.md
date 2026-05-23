# 🍌 Banana Leaf Disease Detection — Backend API

A production-ready FastAPI server that classifies banana leaf diseases using a
soft-voting ensemble of three CNN models (Custom CNN, ResNet50, InceptionV3).

---

## 📁 Project Structure

```
backend/
├── main.py              ← FastAPI application (all routes + logic)
├── export_models.py     ← Helper: save trained models from Kaggle notebook
├── requirements.txt     ← Python dependencies
├── models/              ← Place your .h5 model files here
│   ├── model_custom.h5
│   ├── model_resnet.h5
│   └── model_inception.h5
└── README.md
```

---

## 🚀 Quick Start

### 1 — Export trained models (Kaggle)

Add this cell at the **end of your Kaggle notebook** and run it:

```python
import os, shutil

SAVE_DIR = "/kaggle/working/models"
os.makedirs(SAVE_DIR, exist_ok=True)

model_custom.save(f"{SAVE_DIR}/model_custom.h5")
model_res.save(f"{SAVE_DIR}/model_resnet.h5")
model_inc.save(f"{SAVE_DIR}/model_inception.h5")

print("Models saved to", SAVE_DIR)
```

Download the three `.h5` files from Kaggle and copy them to `backend/models/`.

---

### 2 — Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

---

### 3 — Start the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

| URL | Description |
|-----|-------------|
| `http://localhost:8000/docs` | **Swagger UI** (interactive) |
| `http://localhost:8000/redoc` | ReDoc documentation |
| `http://localhost:8000/health` | Health check |
| `http://localhost:8000/predict` | POST — upload image |
| `http://localhost:8000/classes` | GET — list all classes |

---

## 📡 API Endpoints

### `POST /predict`

Upload a banana leaf image and receive a disease prediction.

**Request:**
- `Content-Type: multipart/form-data`
- Field: `file` — image file (JPG, PNG, WEBP, BMP — max 10 MB)

**Example with `curl`:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/banana_leaf.jpg"
```

**Example with Python `requests`:**

```python
import requests

url = "http://localhost:8000/predict"
with open("banana_leaf.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.json())
```

**Response (200 OK):**

```json
{
  "prediction": "Banana Healthy Leaf",
  "confidence": 0.9423,
  "all_predictions": [
    { "label": "Banana Healthy Leaf",                "confidence": 0.9423 },
    { "label": "Banana Yellow Sigatoka Disease",     "confidence": 0.0312 },
    { "label": "Banana Black Sigatoka Disease",      "confidence": 0.0101 },
    { "label": "Banana Bract Mosaic Virus Disease",  "confidence": 0.0079 },
    { "label": "Banana Insect Pest Disease",         "confidence": 0.0052 },
    { "label": "Banana Moko Disease",               "confidence": 0.0021 },
    { "label": "Banana Panama Disease",             "confidence": 0.0012 }
  ],
  "models_used": ["custom", "resnet", "inception"]
}
```

---

### `GET /health`

Returns server status and loaded models.

```json
{
  "status": "ok",
  "models_loaded": ["custom", "resnet", "inception"],
  "classes": ["Banana Black Sigatoka Disease", "..."]
}
```

---

### `GET /classes`

Returns all detectable disease classes with their indices.

```json
{
  "total": 7,
  "classes": [
    { "index": 0, "label": "Banana Black Sigatoka Disease" },
    { "index": 1, "label": "Banana Bract Mosaic Virus Disease" },
    { "index": 2, "label": "Banana Healthy Leaf" },
    { "index": 3, "label": "Banana Insect Pest Disease" },
    { "index": 4, "label": "Banana Moko Disease" },
    { "index": 5, "label": "Banana Panama Disease" },
    { "index": 6, "label": "Banana Yellow Sigatoka Disease" }
  ]
}
```

---

## 🏥 Disease Classes

| Index | Class Label | Description |
|-------|-------------|-------------|
| 0 | Banana Black Sigatoka Disease | Fungal disease (Mycosphaerella fijiensis) causing dark leaf spots |
| 1 | Banana Bract Mosaic Virus Disease | Viral infection causing mosaic leaf patterns |
| 2 | Banana Healthy Leaf | No disease detected |
| 3 | Banana Insect Pest Disease | Leaf damage from insect pests |
| 4 | Banana Moko Disease | Bacterial wilt (Ralstonia solanacearum) |
| 5 | Banana Panama Disease | Fusarium wilt causing yellowing and wilting |
| 6 | Banana Yellow Sigatoka Disease | Mild fungal leaf disease |

> **Note:** Class indices follow alphabetical order of the dataset folder names,
> matching the mapping assigned by Keras `flow_from_directory`.

---

## 🤖 Ensemble Model Details

| Model | Weight | Description |
|-------|--------|-------------|
| Custom CNN | 20% | Deep custom architecture with BatchNorm + GlobalAveragePooling2D |
| ResNet50 | 40% | Fine-tuned ImageNet pretrained model (last 20 layers unfrozen) |
| InceptionV3 | 40% | Fine-tuned ImageNet pretrained model (last 30 layers unfrozen) |

**Prediction formula:**
```
final_prob = (custom * 0.2) + (resnet * 0.4) + (inception * 0.4)
```

---

## ⚙️ Environment Variables (optional)

You can override defaults using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Bind address |

---

## 🐋 Docker (optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t banana-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models banana-api
```
