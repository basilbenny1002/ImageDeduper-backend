
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
import shutil, os
from pathlib import Path
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()
MEDIA_ROOT = Path("media/temp")
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)

progress = {} 

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    user_dir = MEDIA_ROOT / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    with open(user_dir / file.filename, "wb") as f:
        shutil.copyfileobj(file.file, f)
    progress[user_id] = progress.get(user_id, 0) + 1
    return {"message": f"{file.filename} saved"}

@app.get("/progress/{user_id}")
def get_progress(user_id: str):
    return {"progress": progress.get(user_id, 0)}

app.get("/download/{user_id}")
def download_files(user_id: str):
    user_dir = MEDIA_ROOT / user_id
    if not user_dir.exists():
        return {"error": "User directory does not exist"}
    zip_path = MEDIA_ROOT / f"{user_id}_output.zip"
    shutil.make_archive(str(zip_path).replace('.zip', ''), 'zip', user_dir)
    return {"download_url": str(zip_path)}

@app.get("/")
def read_root():
    return JSONResponse(status_code=200, content={"Status": "it works"})