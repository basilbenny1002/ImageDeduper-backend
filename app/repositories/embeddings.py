import sqlite3
from pathlib import Path
from typing import Iterable, Tuple, List


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
        self._conn.commit()

    def upsert(self, path: str, embedding: bytes) -> None:
        self._cursor.execute(
            "INSERT OR REPLACE INTO images(path, embedding) VALUES (?, ?)",
            (path, embedding),
        )
        self._conn.commit()

    def delete_many(self, paths: Iterable[str]) -> None:
        self._cursor.executemany(
            "DELETE FROM images WHERE path=?", ((p,) for p in paths)
        )
        self._conn.commit()

    def list_all(self) -> List[Tuple[str, bytes]]:
        self._cursor.execute("SELECT path, embedding FROM images")
        return self._cursor.fetchall()

    def close(self) -> None:
        try:
            self._cursor.close()
        finally:
            self._conn.close()
