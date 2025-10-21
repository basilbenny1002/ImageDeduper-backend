from pathlib import Path


class Settings:
    # Base directories
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MEDIA_ROOT = BASE_DIR / "media" / "temp"
    MEDIA_ROOT.mkdir(parents=True, exist_ok=True)

    # Database: per-user embeddings under their media folder
    def user_db_path(self, user_id: str) -> Path:
        """Return the path to the embeddings DB for a specific user.

        Ensures the parent directory exists.
        """
        db_path = self.MEDIA_ROOT / user_id / "embeddings.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path


settings = Settings()
