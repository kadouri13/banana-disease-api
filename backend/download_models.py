"""
Model Downloader for Render Deployment
=======================================
Downloads the three trained .h5 model files from Google Drive
before the FastAPI server starts.

HOW TO USE:
1. Upload your 3 .h5 files to Google Drive and make them publicly shareable
2. Copy each file's ID from the sharing link:
      https://drive.google.com/file/d/<FILE_ID>/view
3. Paste the FILE_IDs below in the MODELS dict
4. This script is called automatically by start.sh on Render startup
"""

import os
import sys

# ─── CONFIGURE THESE ────────────────────────────────────────────────────────
# Replace each "YOUR_FILE_ID_HERE" with the actual Google Drive file ID.
# Get the ID from: File → Share → Copy link
#   https://drive.google.com/file/d/  <FILE_ID>  /view?usp=sharing
MODELS = {
    "model_custom.h5":   "1rZtGHhPjj1IS71XHXCc3ddbUfIQH2x6Z",
    "model_resnet.h5":   "1xqg0CaizWtPf7WvZXvwsPmxCuW4CdOBD",
    "model_inception.h5": "1fhA-pV5Kv0XegP9Nb1V3KUFaM2SybsZU",
}
# ─────────────────────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def download_from_gdrive(file_id: str, dest_path: str) -> None:
    """Download a file from Google Drive using gdown."""
    import gdown  # installed via requirements.txt

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading from Google Drive → {dest_path}")
    gdown.download(url, dest_path, quiet=False, fuzzy=True)


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    all_present = all(
        os.path.exists(os.path.join(MODELS_DIR, fname))
        for fname in MODELS
    )
    if all_present:
        print("✅ All model files already present — skipping download.")
        return

    print("📦 Downloading model files …")
    for filename, file_id in MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)

        if os.path.exists(dest):
            print(f"  ✓ {filename} already exists — skipping.")
            continue

        if file_id.startswith("YOUR_"):
            print(
                f"\n❌ ERROR: You must set the Google Drive file ID for '{filename}' "
                f"in download_models.py before deploying to Render.\n"
            )
            sys.exit(1)

        download_from_gdrive(file_id, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  ✅ {filename} downloaded ({size_mb:.1f} MB)")

    print("\n✅ All models ready.")


if __name__ == "__main__":
    main()
