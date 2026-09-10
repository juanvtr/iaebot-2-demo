from __future__ import annotations

from hashlib import sha1
from io import BytesIO
import logging
import os
from pathlib import Path
import re
from typing import Any

import fitz  # PyMuPDF
from ollama import Client
from pypdf import PdfReader

from .embedding_utils import EmbeddingManager
from .storage import Storage

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/uploads")


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


def _make_row(
    filename: str,
    page_number: int | None,
    content_type: str,
    content: str,
    *,
    figure_label: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    basis = f"{filename}:{page_number}:{content_type}:{figure_label}:{content[:180]}"
    return {
        "id": sha1(basis.encode("utf-8")).hexdigest(),
        "content": content,
        "filename": filename,
        "page_number": page_number,
        "content_type": content_type,
        "figure_label": figure_label,
        "metadata": metadata or {},
        "embedding": None,
    }


def _attach_embeddings(rows: list[dict[str, Any]], embedding_manager: EmbeddingManager) -> None:
    if not rows:
        return
    vectors = embedding_manager.embed_texts([row["content"] for row in rows])
    for row, vector in zip(rows, vectors):
        row["embedding"] = vector


def index_document(
    filename: str,
    text: str,
    embedding_manager: EmbeddingManager,
    storage: Storage,
    page_number: int | None = None,
) -> int:
    rows = [
        _make_row(filename, page_number, "text", chunk)
        for chunk in split_text(text)
    ]
    _attach_embeddings(rows, embedding_manager)
    return storage.insert_many(rows)


def _table_to_markdown(table: list[list[Any]]) -> str:
    cleaned = []
    for row in table:
        cleaned.append([
            re.sub(r"\s+", " ", str(cell)).strip() if cell is not None else ""
            for cell in row
        ])
    if not cleaned:
        return ""

    width = max(len(row) for row in cleaned)
    normalized = [row + [""] * (width - len(row)) for row in cleaned]
    header = normalized[0]
    body = normalized[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _extract_tables(page: fitz.Page, filename: str, page_number: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        finder = page.find_tables()
        for idx, table_obj in enumerate(finder.tables, start=1):
            table = table_obj.extract()
            markdown = _table_to_markdown(table)
            if not markdown.strip():
                continue
            content = (
                f"Tabela {idx} detectada na página {page_number} do documento {filename}.\n"
                f"{markdown}"
            )
            rows.append(
                _make_row(
                    filename,
                    page_number,
                    "table",
                    content,
                    metadata={"table_index": idx, "extraction": "pymupdf"},
                )
            )
    except Exception as exc:
        logger.info("Não foi possível extrair tabelas da página %s: %s", page_number, exc)
    return rows


def _figure_label(page_text: str) -> str | None:
    matches = re.findall(r"\bFigura\s+\d+\b", page_text, flags=re.IGNORECASE)
    if not matches:
        return None
    value = matches[-1]
    return value[0].upper() + value[1:]


def _analyze_visual_page(
    page: fitz.Page,
    filename: str,
    page_number: int,
    page_text: str,
) -> dict[str, Any] | None:
    # Para a demo, só chamamos o modelo de visão quando a página contém
    # pelo menos uma imagem raster. Isso evita analisar todas as páginas.
    if not page.get_images(full=True):
        return None

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_VISION_MODEL", "llava")
    api_key = os.getenv("OLLAMA_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

    # Renderizamos a página completa: legenda, eixos e anotações permanecem
    # no mesmo contexto visual do gráfico/imagem.
    pix = page.get_pixmap(matrix=fitz.Matrix(1.7, 1.7), alpha=False)
    png_bytes = pix.tobytes("png")

    prompt = (
        "Você está analisando uma página de um relatório técnico em português. "
        "Descreva SOMENTE informações visuais relevantes presentes em gráficos, figuras, "
        "diagramas ou imagens. Leia valores, unidades, rótulos, eixos, legendas e anotações "
        "visíveis. Se houver um gráfico, destaque máximos, mínimos e valores explicitamente "
        "marcados. Não invente valores. Responda de forma objetiva em português, como uma "
        "evidência que será armazenada em um sistema RAG."
    )

    try:
        client = Client(host=host, headers=headers)
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [png_bytes],
                }
            ],
            options={"temperature": 0.0},
            stream=False,
        )
        try:
            description = response.message.content.strip()
        except AttributeError:
            description = response["message"]["content"].strip()
    except Exception as exc:
        logger.warning("Falha no LLaVA para página %s: %s", page_number, exc)
        return None

    if not description:
        return None

    label = _figure_label(page_text)
    prefix = f"{label}. " if label else ""
    content = (
        f"{prefix}Evidência visual da página {page_number} do documento {filename}. "
        f"{description}"
    )
    return _make_row(
        filename,
        page_number,
        "image",
        content,
        figure_label=label,
        metadata={"vision_model": model, "extraction": "page_render"},
    )


def index_pdf_bytes(
    filename: str,
    data: bytes,
    embedding_manager: EmbeddingManager,
    storage: Storage,
) -> int:
    """Indexa texto, tabelas e evidências visuais de um PDF.

    O PDF original é salvo localmente em data/uploads/. As evidências extraídas
    recebem embeddings e são persistidas no backend configurado (MotherDuck ou SQLite).
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    (UPLOAD_DIR / safe_name).write_bytes(data)

    reader = PdfReader(BytesIO(data))
    visual_doc = fitz.open(stream=data, filetype="pdf")
    rows: list[dict[str, Any]] = []

    for page_no, (text_page, visual_page) in enumerate(
        zip(reader.pages, visual_doc), start=1
    ):
        page_text = text_page.extract_text() or ""

        for chunk in split_text(page_text):
            rows.append(_make_row(safe_name, page_no, "text", chunk))

        rows.extend(_extract_tables(visual_page, safe_name, page_no))

        visual_row = _analyze_visual_page(
            visual_page,
            safe_name,
            page_no,
            page_text,
        )
        if visual_row is not None:
            rows.append(visual_row)

    visual_doc.close()

    if not rows:
        return 0

    _attach_embeddings(rows, embedding_manager)

    # Só substituímos a versão anterior depois que toda a extração terminou.
    storage.delete_source(safe_name)
    return storage.insert_many(rows)
