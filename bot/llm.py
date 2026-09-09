from __future__ import annotations

import logging
import re

from ollama import Client

logger = logging.getLogger(__name__)


class OllamaGenerator:
    def __init__(self, host: str, model: str, api_key: str = ""):
        self.host = host.rstrip("/")
        self.model = model
        self.api_key = api_key

    @property
    def configured(self) -> bool:
        if self.host.startswith("https://ollama.com"):
            return bool(self.api_key)
        return True

    def generate(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        client = Client(host=self.host, headers=headers)
        response = client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
            stream=False,
        )
        try:
            return response.message.content.strip()
        except AttributeError:
            return response["message"]["content"].strip()


def extractive_fallback(contexts: list[dict]) -> str:
    if not contexts:
        return "Com base nos documentos analisados, não encontrei uma resposta para esta pergunta."

    passages = []
    for item in contexts[:3]:
        sentences = re.split(r"(?<=[.!?])\s+", item["content"].strip())
        excerpt = " ".join(sentences[:2]).strip()
        if excerpt:
            page = f", p. {item['page_number']}" if item.get("page_number") else ""
            passages.append(f"{excerpt} [Fonte: {item['source']}{page}]")
    return "\n\n".join(passages)
