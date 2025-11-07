import sqlite3
from pathlib import Path
from typing import Iterable, Tuple, List, Optional
import numpy as np


class EmbeddingsRepository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        # Background tasks may use a different thread; allow cross-thread access
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._cursor = self._conn.cursor()
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                path TEXT PRIMARY KEY,
                embedding BLOB
            )
            """
        )
        # Add index for faster lookups
        self._cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_path ON images(path)"
        )
        self._conn.commit()

    def upsert(self, path: str, embedding: bytes) -> None:
        self._cursor.execute(
            "INSERT OR REPLACE INTO images(path, embedding) VALUES (?, ?)",
            (path, embedding),
        )
        # Don't commit here - let caller batch commits

    def upsert_many(self, items: List[Tuple[str, bytes]]) -> None:
        """Batch insert/update multiple embeddings."""
        self._cursor.executemany(
            "INSERT OR REPLACE INTO images(path, embedding) VALUES (?, ?)",
            items
        )
        # Don't commit here - let caller batch commits

    def commit(self) -> None:
        """Explicitly commit pending changes."""
        self._conn.commit()

    def delete_many(self, paths: Iterable[str]) -> None:
        self._cursor.executemany(
            "DELETE FROM images WHERE path=?", ((p,) for p in paths)
        )
        # Don't commit here - let caller batch commits

    def list_all(self) -> List[Tuple[str, bytes]]:
        self._cursor.execute("SELECT path, embedding FROM images")
        return self._cursor.fetchall()

    def get_all_as_matrix(self) -> Tuple[List[str], Optional[np.ndarray]]:
        """Return all paths and embeddings as a numpy matrix for vectorized operations."""
        rows = self.list_all()
        if not rows:
            return [], None
        paths = [row[0] for row in rows]
        embeddings = np.vstack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
        return paths, embeddings

    def close(self) -> None:
        try:
            self._cursor.close()
        finally:
            self._conn.close()
