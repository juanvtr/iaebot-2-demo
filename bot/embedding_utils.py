from __future__ import annotations

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from .storage import Storage


class EmbeddingManager:
    """Busca semântica MiniLM com suporte a evidências textuais, tabelas e imagens descritas."""

    def __init__(self, storage: Storage, model_name: str):
        self.storage = storage
        self.embedding_model = SentenceTransformer(model_name, device="cpu")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.embedding_model.encode(
            texts,
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [np.asarray(v, dtype=np.float32).tolist() for v in vectors]

    def embed_query(self, query: str) -> np.ndarray:
        return np.asarray(
            self.embedding_model.encode(
                query,
                show_progress_bar=False,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    def fetch_similar_text(
        self,
        query: str,
        top_k: int = 8,
        similarity_threshold: float = 0.4,
    ) -> list[dict]:
        items = self.storage.fetch_all()
        if not items:
            return []

        query_vector = self.embed_query(query)
        missing = [i for i, item in enumerate(items) if item.get("embedding") is None]
        if missing:
            vectors = self.embed_texts([items[i]["content"] for i in missing])
            for i, vector in zip(missing, vectors):
                items[i]["embedding"] = vector

        page_match = re.search(r"p[aá]gina\s+(\d+)", query.lower())
        requested_page = int(page_match.group(1)) if page_match else None
        multimodal_terms = {
            "gráfico", "grafico", "figura", "imagem", "tabela",
            "ensaio", "deformação", "deformacao",
        }
        qterms = set(re.findall(r"[\wÀ-ÿ]+", query.lower()))

        results = []
        for item in items:
            vec = np.asarray(item["embedding"], dtype=np.float32)
            semantic = float(np.dot(query_vector, vec))
            bonus = 0.0
            if requested_page and item.get("page_number") == requested_page:
                bonus += 0.16
            if item.get("content_type") in {"image", "table"} and qterms & multimodal_terms:
                bonus += 0.08
            score = min(1.0, semantic + bonus)
            if score >= similarity_threshold:
                results.append({
                    "content": item["content"],
                    "source": item["source"],
                    "score": score,
                    "page_number": item.get("page_number"),
                    "content_type": item.get("content_type", "text"),
                    "figure_label": item.get("figure_label"),
                    "metadata": item.get("metadata", {}),
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
