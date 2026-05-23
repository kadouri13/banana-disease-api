"""
Banana Leaf Disease Detection API
==================================
FastAPI backend that loads three trained Keras models (Custom CNN, ResNet50, InceptionV3)
and uses soft-voting ensemble to predict banana leaf disease from an uploaded image.

Swagger UI: http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc
"""

import io
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
MAX_FILE_SIZE_MB = 10

# Classes in the order that Keras flow_from_directory assigns them
# (alphabetical order of folder names in the dataset)
CLASS_NAMES = [
    "Banana Black Sigatoka Disease",
    "Banana Bract Mosaic Virus Disease",
    "Banana Healthy Leaf",
    "Banana Insect Pest Disease",
    "Banana Moko Disease",
    "Banana Panama Disease",
    "Banana Yellow Sigatoka Disease",
]

# Soft-voting weights: custom=1.0 (Using only custom to save memory on Render Free Tier)
WEIGHTS = {"custom": 1.0}

# Model file paths (relative to this file; adjust if needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "custom":   os.path.join(BASE_DIR, "models", "model_custom.h5"),
    # Commented out to save memory on Render Free Tier
    # "resnet":   os.path.join(BASE_DIR, "models", "model_resnet.h5"),
    # "inception": os.path.join(BASE_DIR, "models", "model_inception.h5"),
}

# ---------------------------------------------------------------------------
# Model store — loaded once at startup via lifespan
# ---------------------------------------------------------------------------
models: dict = {}


def load_models() -> None:
    """Load all three Keras models from disk."""
    # Import TF here so startup errors are caught cleanly
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Run: pip install tensorflow"
        ) from exc

    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            logger.warning(
                "Model file not found: %s — skipping '%s'. "
                "Place the .h5 file in the models/ directory.",
                path, name,
            )
            continue
        logger.info("Loading model '%s' from %s …", name, path)
        models[name] = load_model(path)
        logger.info("Model '%s' loaded successfully.", name)

    if not models:
        logger.error(
            "No model files found in %s/models/. "
            "Copy model_custom.h5, model_resnet.h5, model_inception.h5 there.",
            BASE_DIR,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    load_models()
    yield
    models.clear()
    logger.info("Models unloaded.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Banana Leaf Disease Detection API",
    description=(
        "## 🍌 Banana Leaf Disease Detection\n\n"
        "Upload a banana leaf image and receive an AI-powered disease diagnosis "
        "using an ensemble of three CNN models (Custom CNN, ResNet50, InceptionV3).\n\n"
        "### Detectable Conditions\n"
        "| Label | Description |\n"
        "|-------|-------------|\n"
        "| Banana Healthy Leaf | No disease detected — leaf is healthy |\n"
        "| Banana Black Sigatoka Disease | Fungal disease causing dark leaf spots |\n"
        "| Banana Yellow Sigatoka Disease | Mild fungal infection with yellow streaks |\n"
        "| Banana Bract Mosaic Virus Disease | Viral infection causing mosaic patterns |\n"
        "| Banana Moko Disease | Bacterial wilt affecting vascular tissue |\n"
        "| Banana Panama Disease | Fusarium wilt causing yellowing and wilting |\n"
        "| Banana Insect Pest Disease | Leaf damage caused by insect pests |\n\n"
        "### Ensemble Strategy\n"
        "Predictions are combined using **soft voting** with the following weights:\n"
        "- Custom CNN → **20%**\n"
        "- ResNet50 → **40%**\n"
        "- InceptionV3 → **40%**\n"
    ),
    version="1.0.0",
    contact={
        "name": "Banana Disease API",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)

# Allow all origins for development — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------
class PredictionResult(BaseModel):
    """Detailed prediction for a single class."""
    label: str = Field(..., example="Banana Healthy Leaf")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, example=0.9423,
        description="Confidence score between 0 and 1",
    )


class PredictionResponse(BaseModel):
    """Full response returned by the /predict endpoint."""
    prediction: str = Field(
        ..., example="Banana Healthy Leaf",
        description="Top predicted class label",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, example=0.9423,
        description="Confidence of the top prediction (0–1)",
    )
    all_predictions: list[PredictionResult] = Field(
        ...,
        description="Confidence scores for all 7 classes, sorted by confidence descending",
    )
    models_used: list[str] = Field(
        ..., example=["custom", "resnet", "inception"],
        description="Names of models that contributed to this prediction",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., example="ok")
    models_loaded: list[str] = Field(
        ..., example=["custom", "resnet", "inception"],
        description="Names of currently loaded models",
    )
    classes: list[str] = Field(
        ..., description="All recognisable class labels",
    )


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str = Field(..., example="Uploaded file is not a valid image.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes to a normalised NumPy array ready for model inference.

    Returns shape (1, 224, 224, 3), values in [0, 1].
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Cannot decode image: {exc}",
        ) from exc

    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


def ensemble_predict(img_array: np.ndarray) -> np.ndarray:
    """
    Run soft-voting ensemble across loaded models.

    Returns a probability vector of shape (num_classes,).
    """
    if not models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No models are loaded. "
                "Place model_custom.h5, model_resnet.h5, and model_inception.h5 "
                "inside the backend/models/ directory and restart the server."
            ),
        )

    combined = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    total_weight = 0.0

    for name, model in models.items():
        weight = WEIGHTS.get(name, 1.0)
        preds = model.predict(img_array, verbose=0)[0]  # (num_classes,)
        combined += preds * weight
        total_weight += weight
        logger.debug("Model '%s' raw top class: %s (%.4f)", name, CLASS_NAMES[np.argmax(preds)], np.max(preds))

    # Normalise in case not all models were loaded
    combined /= total_weight
    return combined


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the API status, which models are currently loaded, and the list of recognisable classes.",
    tags=["Utility"],
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded=list(models.keys()),
        classes=CLASS_NAMES,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Banana Leaf Disease",
    description=(
        "Upload a banana leaf image (JPG, PNG, WEBP, BMP, GIF) or provide an image URL, "
        "and receive a disease diagnosis from the ensemble model.\n\n"
        "**Supported formats:** JPEG · PNG · WEBP · BMP · GIF\n"
        "**Max file size:** 10 MB\n"
        "**Image is automatically resized** to 224 × 224 px before inference."
    ),
    tags=["Inference"],
    responses={
        200: {"description": "Successful prediction"},
        422: {"model": ErrorResponse, "description": "Invalid or unreadable image"},
        503: {"model": ErrorResponse, "description": "No models loaded"},
    },
)
@app.post(
    "/predict/",
    response_model=PredictionResponse,
    include_in_schema=False,
)
async def predict(request: Request) -> PredictionResponse:
    image_bytes = None
    filename = "image"
    file = None
    url = None

    content_type = request.headers.get("content-type", "")

    # 1. Manually parse the body to avoid FastAPI File/Form strictness issues
    if "application/json" in content_type:
        try:
            body = await request.json()
            url = body.get("file") or body.get("url")
        except Exception:
            pass
    else:
        # This handles both multipart/form-data and x-www-form-urlencoded
        try:
            form_data = await request.form()
            file = form_data.get("file")
            url = form_data.get("url")
        except Exception:
            pass

    # 2. Check if user passed a URL in the 'file' field as text in multipart/form-data
    # (very common mistake in Postman when type is left as 'Text')
    if isinstance(file, str) and file.startswith("http"):
        url = file
        file = None

    # 2. Download from URL if provided
    if url:
        import requests
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            image_bytes = resp.content
            filename = url.split("/")[-1]
            content_type = resp.headers.get("Content-Type", "")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to download image from URL: {str(e)}"
            )

    # 3. Otherwise read from uploaded file
    elif file and hasattr(file, 'read'):
        # ── Validate content type ────────────────────────────────────────────────
        allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/gif"}
        if file.content_type and file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Unsupported file type '{file.content_type}'. "
                    f"Allowed: {', '.join(allowed_types)}"
                ),
            )
        image_bytes = await file.read()
        filename = file.filename

    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="You must provide either an image 'file' or a valid image 'url'."
        )

    # ── Enforce size limit ──────────────────────────────────────
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB.",
        )

    # ── Preprocess ───────────────────────────────────────────────────────────
    logger.info("Processing image '%s' (%d bytes)", filename, len(image_bytes))
    img_array = preprocess_image(image_bytes)

    # ── Inference ────────────────────────────────────────────────────────────
    probabilities = ensemble_predict(img_array)

    # ── Build response ───────────────────────────────────────────────────────
    top_idx = int(np.argmax(probabilities))
    top_label = CLASS_NAMES[top_idx]
    top_confidence = float(probabilities[top_idx])

    all_preds = sorted(
        [
            PredictionResult(label=CLASS_NAMES[i], confidence=float(probabilities[i]))
            for i in range(len(CLASS_NAMES))
        ],
        key=lambda x: x.confidence,
        reverse=True,
    )

    logger.info(
        "Prediction: '%s' (%.2f%%) | File: '%s'",
        top_label, top_confidence * 100, file.filename,
    )

    return PredictionResponse(
        prediction=top_label,
        confidence=round(top_confidence, 6),
        all_predictions=all_preds,
        models_used=list(models.keys()),
    )


@app.get(
    "/classes",
    summary="List All Classes",
    description="Returns the list of all disease categories the model can detect.",
    tags=["Utility"],
    response_model=dict,
)
async def list_classes():
    return {
        "total": len(CLASS_NAMES),
        "classes": [
            {"index": i, "label": label}
            for i, label in enumerate(CLASS_NAMES)
        ],
    }
