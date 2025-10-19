from pathlib import Path


class Settings:
    # Base directories
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MEDIA_ROOT = BASE_DIR / "media" / "temp"
    MEDIA_ROOT.mkdir(parents=True, exist_ok=True)

    # Database
    DB_PATH = BASE_DIR / "embeddings.db"


settings = Settings()
