# IAEbot 2.0 — Demo Multimodal

Prova de conceito para a apresentação do **IAEbot 2.0: Análise Multimodal e Extração de Conhecimento de Relatórios Técnicos**.

A primeira fase validou RAG textual. Esta demo mostra a evolução da base de conhecimento para armazenar evidências com **tipo de conteúdo e página**, incluindo texto, tabelas e descrições de elementos visuais.

## Pergunta principal da demo

> Qual foi a deformação máxima registrada no ensaio de tração, conforme o gráfico da página 12?

Resposta esperada: **14,2%**, recuperada da evidência visual da **Figura 4, página 12** do relatório técnico sintético.

## Arquitetura da demo

`PDF → texto/tabela/imagem → descrição/extração → embeddings MiniLM → MotherDuck → busca semântica → LLM → resposta + fonte/página`

> O relatório incluído é sintético e foi criado apenas para demonstração acadêmica. A evidência visual da página 12 foi pré-processada para tornar a demonstração reprodutível no Streamlit Cloud. A etapa de visão computacional online (ex.: LLaVA) é uma evolução prevista do pipeline 2.0.

## Streamlit Secrets

```toml
MOTHERDUCK_TOKEN = "..."
MOTHERDUCK_DATABASE = "iae_bot_demo"
OLLAMA_HOST = "..."
OLLAMA_API_KEY = "..."
OLLAMA_MODEL = "..."
```

Para dados institucionais sensíveis, a arquitetura-alvo preserva execução local.
