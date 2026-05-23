"""
Export trained Keras models to the backend/models/ directory.

Run this script ONCE from Kaggle (or wherever you trained the models) so that
the three .h5 files are available for the FastAPI server.

Usage
-----
    python export_models.py

The script expects the following variables to be in scope (i.e. after running
the training notebook cells):
    model_custom  – Custom CNN model
    model_res     – ResNet50 model
    model_inc     – InceptionV3 model

Copy the generated .h5 files to  backend/models/  before starting the API server.
"""

import os

SAVE_DIR = "backend/models"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Saving Custom CNN …")
model_custom.save(os.path.join(SAVE_DIR, "model_custom.h5"))

print("Saving ResNet50 …")
model_res.save(os.path.join(SAVE_DIR, "model_resnet.h5"))

print("Saving InceptionV3 …")
model_inc.save(os.path.join(SAVE_DIR, "model_inception.h5"))

print(f"\nAll models saved to ./{SAVE_DIR}/")
print("Now copy these files to your server and start the API with:")
print("  cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000")
