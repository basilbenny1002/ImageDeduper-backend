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
- `embeddings.db` — SQLite DB for cached embeddings

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

## Frontend note (index.html)
The current `index.html` uses a WebSocket endpoint `ws://localhost:8000/ws/upload/`, which is not implemented in this refactor. Switch to HTTP uploads and processing:

- Replace WebSocket uploads with `fetch` POST to `/upload` per file using `FormData`.
- Trigger processing via POST `/process`.
- After a short delay, GET `/download/{user_id}` to retrieve the ZIP.

If you want live progress events, we can add:
- A simple in-memory progress endpoint, or
- Server-Sent Events (SSE) stream for real-time updates, or
- A WebSocket channel. Let me know which you prefer.

## CLI usage

You can also run selection directly on folders:

```powershell
. .venv\Scripts\Activate.ps1
python scripts/selector.py --input .\path\to\images --output .\out --similarity 0.87
```

Use `--no-aesthetics` to skip the aesthetics score.

## Troubleshooting
- First run will download model weights; this can take time.
- Large folders: processing runs in a background task; consider adding progress if needed.
- Windows path issues: the code uses `pathlib.Path` to keep paths OS-safe.

## License
SPDX-License-Identifier: MIT — see header in `scripts/selector.py`.
