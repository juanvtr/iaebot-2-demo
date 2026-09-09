from __future__ import annotations


class Prompt:
    def __init__(self, context_blocks: list[dict], user_query: str):
        self.context_blocks = context_blocks
        self.user_query = user_query

    def generate_document_prompt(self) -> str:
        blocks = []
        for i, b in enumerate(self.context_blocks, 1):
            loc = f", página {b['page_number']}" if b.get("page_number") else ""
            modality = b.get("content_type", "text")
            figure = f", {b['figure_label']}" if b.get("figure_label") else ""
            blocks.append(
                f"--- Evidência {i} [tipo={modality}; fonte={b.get('source')}{loc}{figure}; score={b.get('score', 0):.3f}] ---\n"
                f"{b.get('content', '')}"
            )
        context = "\n\n".join(blocks)
        return f"""Você é o IAEbot 2.0, assistente para análise de relatórios técnicos. Responda SOMENTE com base nas evidências recuperadas.

REGRAS:
1. Priorize a modalidade pedida pelo usuário. Se ele disser 'conforme o gráfico', priorize evidências do tipo image/figure da página indicada, e não valores de uma tabela diferente.
2. Não invente números ou conclusões.
3. Quando houver página disponível, cite fonte e página: [Fonte: arquivo.pdf, p. N].
4. Seja direto e técnico em português do Brasil.
5. Se as evidências forem insuficientes, responda exatamente: "Com base nos documentos analisados, não encontrei uma resposta para esta pergunta."

EVIDÊNCIAS:
{context}

PERGUNTA: {self.user_query}

RESPOSTA FINAL:""".strip()
