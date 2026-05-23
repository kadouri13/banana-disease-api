#!/usr/bin/env bash
# start.sh — Render startup script
# Render runs this command: bash start.sh

set -e

echo "=============================="
echo "  Banana Disease API — Startup"
echo "=============================="

# Step 1: Download models only if they are NOT already present
# (Option A: models committed to GitHub → skip download entirely)
# (Option B: models on Google Drive → download now)

MODEL_DIR="models"
MODELS=("model_custom.h5" "model_resnet.h5" "model_inception.h5")
ALL_PRESENT=true

for model in "${MODELS[@]}"; do
    if [ ! -f "$MODEL_DIR/$model" ]; then
        ALL_PRESENT=false
        break
    fi
done

if [ "$ALL_PRESENT" = true ]; then
    echo "✅ All model files found in $MODEL_DIR/ — skipping download."
else
    echo "▶ Step 1/2 — Downloading model files from Google Drive …"
    python download_models.py
fi

# Step 2: Start the FastAPI server
# Render injects the PORT environment variable automatically
echo ""
echo "▶ Starting FastAPI server on port ${PORT:-8000} …"
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
