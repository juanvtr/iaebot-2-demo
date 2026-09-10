from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Iterable

import duckdb

logger = logging.getLogger(__name__)


class Storage:
    """Persistência para evidências textuais e multimodais do IAEbot 2.0."""

    def __init__(self, motherduck_token: str = "", database: str = "iae_bot_demo"):
        self.backend = "SQLite local"
        self._sqlite_path = Path("data/iae_chat_demo.db")
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        token = (motherduck_token or "").strip()
        if token:
            try:
                # O cliente MotherDuck reconhece oficialmente MOTHERDUCK_TOKEN.
                # Usamos a variável de ambiente em vez de interpolar o token na URL,
                # o que evita problemas de parsing/autenticação no Windows.
                os.environ["MOTHERDUCK_TOKEN"] = token
                self._conn = duckdb.connect(f"md:{database}")
                self.backend = f"MotherDuck ({database})"
                self._create_duckdb_schema()
                return
            except Exception as exc:
                logger.warning("Falha ao conectar ao MotherDuck; usando SQLite local: %s", exc)

        self._conn = sqlite3.connect(self._sqlite_path, check_same_thread=False)
        self._create_sqlite_schema()

    def _create_duckdb_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pdf_data (
                id VARCHAR PRIMARY KEY,
                content VARCHAR NOT NULL,
                filename VARCHAR NOT NULL,
                page_number INTEGER,
                content_type VARCHAR NOT NULL DEFAULT 'text',
                figure_label VARCHAR,
                metadata JSON,
                embedding DOUBLE[]
            )
        """)

    def _create_sqlite_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pdf_data (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                filename TEXT NOT NULL,
                page_number INTEGER,
                content_type TEXT NOT NULL DEFAULT 'text',
                figure_label TEXT,
                metadata TEXT,
                embedding TEXT
            )
        """)
        self._conn.commit()

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM pdf_data").fetchone()[0])

    def insert_many(self, rows: Iterable[dict]) -> int:
        rows = list(rows)
        if not rows:
            return 0

        inserted = 0
        for row in rows:
            values = [
                row["id"],
                row["content"],
                row["filename"],
                row.get("page_number"),
                row.get("content_type", "text"),
                row.get("figure_label"),
                json.dumps(row.get("metadata", {}), ensure_ascii=False),
                row.get("embedding"),
            ]
            if self.backend.startswith("MotherDuck"):
                self._conn.execute("DELETE FROM pdf_data WHERE id = ?", [row["id"]])
                self._conn.execute(
                    "INSERT INTO pdf_data VALUES (?, ?, ?, ?, ?, ?, ?::JSON, ?)",
                    values,
                )
            else:
                if values[-1] is not None:
                    values[-1] = json.dumps(values[-1])
                self._conn.execute(
                    "INSERT OR REPLACE INTO pdf_data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    values,
                )
            inserted += 1

        if not self.backend.startswith("MotherDuck"):
            self._conn.commit()
        return inserted

    def fetch_all(self) -> list[dict]:
        rows = self._conn.execute("""
            SELECT id, content, filename, page_number, content_type, figure_label, metadata, embedding
            FROM pdf_data
        """).fetchall()

        items = []
        for row_id, content, filename, page_number, content_type, figure_label, metadata, embedding in rows:
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}
            if isinstance(embedding, str):
                try:
                    embedding = json.loads(embedding)
                except Exception:
                    embedding = None

            items.append({
                "id": row_id,
                "content": content,
                "source": filename,
                "page_number": page_number,
                "content_type": content_type,
                "figure_label": figure_label,
                "metadata": metadata or {},
                "embedding": list(embedding) if embedding is not None else None,
            })
        return items

    def delete_source(self, filename: str) -> None:
        params = [filename] if self.backend.startswith("MotherDuck") else (filename,)
        self._conn.execute("DELETE FROM pdf_data WHERE filename = ?", params)
        if not self.backend.startswith("MotherDuck"):
            self._conn.commit()
