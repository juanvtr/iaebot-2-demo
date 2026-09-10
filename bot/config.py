from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping


def _get(mapping: Mapping[str, Any] | None, key: str, default: str = "") -> str:
    if mapping is not None:
        try:
            value = mapping.get(key)
            if value is not None:
                return str(value)
        except Exception:
            pass
    return os.getenv(key, default)


@dataclass(frozen=True)
class Settings:
    motherduck_token: str = ""
    motherduck_database: str = "iae_bot_demo"
    ollama_host: str = "http://localhost:11434"
    ollama_api_key: str = ""
    ollama_model: str = "llama3.1"
    ollama_vision_model: str = "llava"
    top_k: int = 8
    similarity_threshold: float = 0.40
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    def from_mapping(cls, secrets: Mapping[str, Any] | None = None) -> "Settings":
        return cls(
            motherduck_token=_get(secrets, "MOTHERDUCK_TOKEN"),
            motherduck_database=_get(secrets, "MOTHERDUCK_DATABASE", "iae_bot_demo"),
            ollama_host=_get(secrets, "OLLAMA_HOST", "http://localhost:11434"),
            ollama_api_key=_get(secrets, "OLLAMA_API_KEY"),
            ollama_model=_get(secrets, "OLLAMA_MODEL", "llama3.1"),
            ollama_vision_model=_get(secrets, "OLLAMA_VISION_MODEL", "llava"),
            top_k=int(_get(secrets, "TOP_K", "8")),
            similarity_threshold=float(_get(secrets, "SIMILARITY_THRESHOLD", "0.40")),
            embedding_model=_get(secrets, "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        )
