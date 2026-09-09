from __future__ import annotations

from hashlib import sha1
from io import BytesIO
import re

from pypdf import PdfReader

from .embedding_utils import EmbeddingManager
from .storage import Storage


def split_text(text: str, chunk_size: int = 1050, overlap: int = 160) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def index_document(
    filename: str,
    text: str,
    embedding_manager: EmbeddingManager,
    storage: Storage,
    page_number: int | None = None,
) -> int:
    chunks = split_text(text)
    vectors = embedding_manager.embed_texts(chunks)
    rows = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        rows.append({
            "id": sha1(f"{filename}:{page_number}:{idx}:{chunk[:120]}".encode()).hexdigest(),
            "content": chunk,
            "filename": filename,
            "page_number": page_number,
            "content_type": "text",
            "metadata": {},
            "embedding": vector,
        })
    return storage.insert_many(rows)


def index_pdf_bytes(
    filename: str,
    data: bytes,
    embedding_manager: EmbeddingManager,
    storage: Storage,
) -> int:
    reader = PdfReader(BytesIO(data))
    storage.delete_source(filename)
    total = 0
    for page_no, page in enumerate(reader.pages, start=1):
        total += index_document(
            filename,
            page.extract_text() or "",
            embedding_manager,
            storage,
            page_no,
        )
    return total
