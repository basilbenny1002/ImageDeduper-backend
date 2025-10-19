from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Image Selector Backend")
app.include_router(router)