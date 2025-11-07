from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import settings
from app.services.image_selector import ImageSelectorService


router = APIRouter()
selector_service = ImageSelectorService()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...), user_id: str = Form(...)):
    user_dir = settings.MEDIA_ROOT / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    dest = user_dir / file.filename
    try:
        # Increased chunk size from 1MB to 8MB for better performance
        with dest.open("wb") as f:
            while True:
                chunk = await file.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    finally:
        await file.close()
    return {"message": f"{file.filename} saved"}


@router.post("/process")
async def process_images(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    similarity: float = Form(0.87),
    use_aesthetics: bool = Form(True),
):
    input_dir = settings.MEDIA_ROOT / user_id
    if not input_dir.exists():
        raise HTTPException(status_code=404, detail="User directory not found")
    output_dir = input_dir / "output"

    def _run():
        selector_service.choose_best(user_id=user_id, input_dir=input_dir, output_dir=output_dir, similarity=similarity, use_aesthetics=use_aesthetics)

    background_tasks.add_task(_run)
    return {"status": "started"}


@router.get("/download/{user_id}")
async def download_zip(user_id: str, background_tasks: BackgroundTasks):
    user_dir = settings.MEDIA_ROOT / user_id / "output"
    if not user_dir.exists():
        raise HTTPException(status_code=404, detail="No output for this user")
    
    zip_base = settings.MEDIA_ROOT / f"{user_id}_output"
    zip_path = settings.MEDIA_ROOT / f"{user_id}_output.zip"
    
    # Create zip
    from shutil import make_archive, rmtree

    make_archive(str(zip_base), "zip", user_dir)
    
    # After sending the file, delete the zip and the user's media folder
    def _cleanup():
        try:
            if zip_path.exists():
                zip_path.unlink()
        except Exception:
            pass
        try:
            rmtree(settings.MEDIA_ROOT / user_id, ignore_errors=True)
        except Exception:
            pass

    background_tasks.add_task(_cleanup)
    return FileResponse(zip_path, filename=f"{user_id}_output.zip")


@router.get("/progress/{user_id}")
async def get_progress(user_id: str):
    return selector_service.get_progress(user_id)


@router.get("/")
async def health():
    return JSONResponse(status_code=200, content={"status": "ok"})
