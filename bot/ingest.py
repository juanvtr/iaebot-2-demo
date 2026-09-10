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


def _vision_client() -> tuple[Client, str]:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_VISION_MODEL", "llava")
    api_key = os.getenv("OLLAMA_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    return Client(host=host, headers=headers), model


def _describe_image(
    image_bytes: bytes,
    *,
    page_number: int,
    filename: str,
    page_text: str,
) -> tuple[str, str] | None:
    client, model = _vision_client()

    context = re.sub(r"\s+", " ", page_text).strip()[:1200]
    prompt = (
        "Analise diretamente a imagem anexada, extraída de um relatório técnico em português. "
        "Transcreva e descreva os dados visuais relevantes com precisão. Leia títulos, eixos, "
        "legendas, unidades, números, porcentagens, máximos, mínimos e anotações. "
        "Se houver uma anotação com um valor numérico, repita esse valor explicitamente na resposta. "
        "Em especial, se aparecer algo como 'deformação máxima registrada', informe exatamente a "
        "porcentagem mostrada. Não estime pela curva e não invente valores. "
        "Responda em português, de forma objetiva, como uma evidência para um sistema RAG.\n\n"
        f"Documento: {filename}\nPágina: {page_number}\n"
        f"Contexto textual da página (apenas para contexto, não substitui a leitura da imagem): {context}"
    )

    try:
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes],
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
    return description, model


def _analyze_visual_page(
    page: fitz.Page,
    filename: str,
    page_number: int,
    page_text: str,
) -> list[dict[str, Any]]:
    """Analisa imagens raster diretamente; usa render da página apenas como fallback.

    Enviar a imagem original ao modelo de visão preserva a resolução de gráficos e
    anotações numéricas pequenas, evitando que elas se percam em um screenshot da página inteira.
    """
    rows: list[dict[str, Any]] = []
    label = _figure_label(page_text)
    images = page.get_images(full=True)

    for image_index, image_info in enumerate(images, start=1):
        xref = image_info[0]
        try:
            extracted = page.parent.extract_image(xref)
            image_bytes = extracted.get("image")
            width = int(extracted.get("width") or 0)
            height = int(extracted.get("height") or 0)
        except Exception as exc:
            logger.info("Falha ao extrair imagem %s da página %s: %s", image_index, page_number, exc)
            continue

        # Evita gastar tempo com logos, ícones e imagens decorativas muito pequenas.
        if not image_bytes or width < 320 or height < 220 or width * height < 120_000:
            continue

        result = _describe_image(
            image_bytes,
            page_number=page_number,
            filename=filename,
            page_text=page_text,
        )
        if result is None:
            continue

        description, model = result
        prefix = f"{label}. " if label else ""
        rows.append(
            _make_row(
                filename,
                page_number,
                "image",
                f"{prefix}Evidência visual da página {page_number} do documento {filename}. {description}",
                figure_label=label,
                metadata={
                    "vision_model": model,
                    "extraction": "embedded_image",
                    "image_index": image_index,
                    "width": width,
                    "height": height,
                },
            )
        )

    if rows:
        return rows

    # Fallback para gráficos vetoriais ou páginas em que a imagem não pôde ser extraída.
    has_visual_hint = bool(images) or bool(page.get_drawings()) or bool(label)
    if not has_visual_hint:
        return []

    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
        rendered = pix.tobytes("png")
    except Exception as exc:
        logger.info("Falha ao renderizar página %s: %s", page_number, exc)
        return []

    result = _describe_image(
        rendered,
        page_number=page_number,
        filename=filename,
        page_text=page_text,
    )
    if result is None:
        return []

    description, model = result
    prefix = f"{label}. " if label else ""
    return [
        _make_row(
            filename,
            page_number,
            "image",
            f"{prefix}Evidência visual da página {page_number} do documento {filename}. {description}",
            figure_label=label,
            metadata={"vision_model": model, "extraction": "page_render_fallback"},
        )
    ]


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
        rows.extend(
            _analyze_visual_page(
                visual_page,
                safe_name,
                page_no,
                page_text,
            )
        )

    visual_doc.close()

    if not rows:
        return 0

    _attach_embeddings(rows, embedding_manager)

    # Só substituímos a versão anterior depois que toda a extração terminou.
    storage.delete_source(safe_name)
    return storage.insert_many(rows)
