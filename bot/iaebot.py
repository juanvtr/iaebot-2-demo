from __future__ import annotations

import logging
from time import perf_counter

from .embedding_utils import EmbeddingManager
from .iaeprompt import Prompt
from .llm import OllamaGenerator, extractive_fallback

logger = logging.getLogger(__name__)


def process_query(
    user_query: str,
    embedding_manager: EmbeddingManager,
    generator: OllamaGenerator,
    top_k: int = 8,
    similarity_threshold: float = 0.4,
) -> dict:
    if not user_query or not user_query.strip():
        return {"answer": "Por favor, insira uma pergunta.", "sources": [], "contexts": []}

    started = perf_counter()
    try:
        contexts = embedding_manager.fetch_similar_text(
            user_query.strip(),
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        if not contexts:
            return {
                "answer": "Com base nos documentos analisados, não encontrei uma resposta para esta pergunta.",
                "sources": [],
                "contexts": [],
                "latency_s": perf_counter() - started,
                "generation_mode": "fallback",
            }

        prompt = Prompt(contexts, user_query.strip()).generate_document_prompt()
        mode = "Ollama"
        try:
            if not generator.configured:
                raise RuntimeError("LLM em nuvem sem chave configurada")
            answer = generator.generate(prompt)
        except Exception as exc:
            logger.warning("Falha de geração no Ollama; usando fallback extrativo: %s", exc)
            answer = extractive_fallback(contexts)
            mode = "fallback extrativo"

        sources = sorted({item["source"] for item in contexts})
        return {
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "latency_s": perf_counter() - started,
            "generation_mode": mode,
        }
    except Exception:
        logger.exception("Erro ao processar consulta")
        return {
            "answer": "Ocorreu um erro ao processar sua pergunta. Tente novamente.",
            "sources": [],
            "contexts": [],
            "latency_s": perf_counter() - started,
            "generation_mode": "erro",
        }
