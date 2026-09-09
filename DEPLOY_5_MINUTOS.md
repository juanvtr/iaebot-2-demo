# Deploy rápido — Streamlit Community Cloud

1. Crie o app apontando para `juanvtr/iaebot-2-demo` e o arquivo `app.py`.
2. Em **Advanced settings → Secrets**, configure:

```toml
MOTHERDUCK_TOKEN = "..."
MOTHERDUCK_DATABASE = "iae_bot_demo"
OLLAMA_HOST = "https://ollama.com"
OLLAMA_API_KEY = "..."
OLLAMA_MODEL = "gpt-oss:20b"
TOP_K = "8"
SIMILARITY_THRESHOLD = "0.40"
```

3. Abra o app e use a pergunta principal:

> Qual foi a deformação máxima registrada no ensaio de tração, conforme o gráfico da página 12?

Resposta esperada: **14,2%**, citando a página 12 do relatório técnico sintético.

## Observação
A demo em nuvem usa somente dados sintéticos/não sensíveis. A arquitetura científica do projeto preserva a execução local para documentação institucional sensível.
