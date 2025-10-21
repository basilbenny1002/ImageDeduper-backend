# Image Selector Backend

A FastAPI backend that uploads images, groups similar ones using ResNet50 embeddings, and keeps the best per group based on an aesthetics score.

## Project structure

- `main.py` — FastAPI entrypoint, includes API router
- `app/` — Application package
  - `core/config.py` — Central settings (media folder, DB path)
  - `api/routes.py` — HTTP endpoints (upload, process, download, health)
  - `services/image_selector.py` — Model logic (embeddings, similarity, aesthetics, selection)
  - `repositories/embeddings.py` — SQLite wrapper for image embeddings
- `scripts/selector.py` — CLI utility that delegates to the service
- `media/temp/` — Per-user upload folders; processed results in `media/temp/<user_id>/output`
- `media/temp/<user_id>/embeddings.db` — Per-user SQLite DB for cached embeddings (auto-removed after download)

## Requirements

This app uses heavy ML dependencies:
- PyTorch (torch, torchvision)
- Transformers (for the aesthetics model)
- FastAPI + Uvicorn

Install everything into a virtual environment. On Windows PowerShell:

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Notes for PyTorch:
- If `pip install torch torchvision` fails, follow the official instructions at https://pytorch.org/get-started/locally/ and install the CPU or CUDA build appropriate for your system.

The aesthetics model will download from Hugging Face on first use (internet required).

## Run the server

From the project root:

```powershell
. .venv\Scripts\Activate.ps1
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Health check:
- GET `http://localhost:8000/` → `{ "status": "ok" }`

## API usage

1) Upload images (repeat per file):
- POST `http://localhost:8000/upload`
  - form fields: `user_id` (text), `file` (file)

2) Start processing:
- POST `http://localhost:8000/process`
  - form fields: `user_id` (text), `similarity` (float, default 0.87), `use_aesthetics` (bool, default true)
  - Returns immediately and processes in background.

3) Download results (ZIP of best images):
- GET `http://localhost:8000/download/{user_id}`

Output images are also available at `media/temp/<user_id>/output/`.

## Frontend
A ready-to-use frontend for this backend is available here:

- https://github.com/basilbenny1002/image-selector-front-end

It handles file uploads, triggers processing, shows progress, and downloads the results.

## Quick start (local)

```powershell
# 1) Clone the repo
git clone https://github.com/basilbenny1002/ImageDeduper-backend.git
cd ImageDeduper-backend

# 2) Create venv and install dependencies
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# 3) Run the server (API)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then:
- Upload images: POST /upload with form fields user_id and file
- Start processing: POST /process with form fields user_id, similarity, use_aesthetics
- Download ZIP: GET /download/{user_id}

## Local (CLI) quick start

Run the tool directly on folders from your machine:

```powershell
# 1) Clone and enter the project
git clone https://github.com/basilbenny1002/ImageDeduper-backend.git
cd ImageDeduper-backend

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the CLI (default similarity = 0.87)
python scripts/selector.py --input .\path\to\images --output .\out

# Example: custom similarity
python scripts/selector.py --input .\path\to\images --output .\out --similarity 0.92
```

Notes:
- The CLI will prompt you to confirm because the input directory may be moved/deleted during processing. Make a backup first.
- The similarity threshold defaults to 0.87 if you don't set it.

## Troubleshooting
- First run will download model weights; this can take time.
- Large folders: processing runs in a background task; consider adding progress if needed.
- Windows path issues: the code uses `pathlib.Path` to keep paths OS-safe.

## License
SPDX-License-Identifier: MIT — see header in `scripts/selector.py`.

## Project origin
This project is based on by:
- https://github.com/basilbenny1002/Image-Selecter
