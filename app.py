from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st

from bot.config import Settings
from bot.embedding_utils import EmbeddingManager
from bot.iaebot import process_query
from bot.ingest import index_document, index_pdf_bytes
from bot.llm import OllamaGenerator
from bot.storage import Storage

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="IAEbot 2.0 — Demo Multimodal",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root { --iae-blue:#0b2e59; --iae-cyan:#0f7fa8; }
.block-container { padding-top: 1.4rem; max-width: 1180px; }
[data-testid="stSidebar"] { border-right: 1px solid #e8eef5; }
.iae-hero {
  padding: 1.25rem 1.35rem; border: 1px solid #e4ebf3; border-radius: 18px;
  background: linear-gradient(120deg,#f8fbff 0%,#ffffff 58%,#f1f7fb 100%);
  margin-bottom: 1rem;
}
.iae-title { color:var(--iae-blue); font-size:2.05rem; font-weight:800; margin:0; }
.iae-sub { color:#526477; margin:.35rem 0 0 0; font-size:1.02rem; }
.badge { display:inline-block; padding:.25rem .55rem; margin:.18rem .18rem .18rem 0; border-radius:999px;
  background:#eef5fb; color:#19486f; border:1px solid #d9e7f3; font-size:.82rem; font-weight:600; }
.small-muted { color:#657587; font-size:.88rem; }
</style>
""",
    unsafe_allow_html=True,
)


def _secret_mapping():
    try:
        return st.secrets
    except Exception:
        return None


@st.cache_resource(show_spinner="Carregando modelo de embeddings e base documental...")
def load_resources():
    settings = Settings.from_mapping(_secret_mapping())
    storage = Storage(settings.motherduck_token, settings.motherduck_database)
    embeddings = EmbeddingManager(storage, settings.embedding_model)
    generator = OllamaGenerator(
        host=settings.ollama_host,
        model=settings.ollama_model,
        api_key=settings.ollama_api_key,
    )
    return settings, storage, embeddings, generator


@st.cache_data
def load_demo_corpus():
    return json.loads(Path("data/demo_corpus.json").read_text(encoding="utf-8"))


def ensure_demo_corpus(storage: Storage, embeddings: EmbeddingManager) -> None:
    if storage.count() > 0:
        return
    with st.spinner("Preparando corpus de demonstração do IAEbot..."):
        for item in load_demo_corpus():
            index_document(item["filename"], item["text"], embeddings, storage)


settings, storage, embedding_manager, generator = load_resources()
ensure_demo_corpus(storage, embedding_manager)

with st.sidebar:
    st.markdown("### ✦ IAEbot 2.0")
    st.caption("Análise multimodal de relatórios técnicos")
    st.divider()

    st.markdown("**Arquitetura da demonstração**")
    st.markdown(f"<span class='badge'>{storage.backend}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='badge'>Ollama · {settings.ollama_model}</span>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>MiniLM-L6-v2</span>", unsafe_allow_html=True)
    st.markdown("<span class='badge'>texto</span><span class='badge'>tabela</span><span class='badge'>imagem</span>", unsafe_allow_html=True)
    st.markdown(
        f"<span class='badge'>top-k {settings.top_k}</span>"
        f"<span class='badge'>threshold {settings.similarity_threshold:.2f}</span>",
        unsafe_allow_html=True,
    )

    st.divider()
    with st.expander("📄 Indexar PDFs na demonstração"):
        st.caption("O upload atual extrai texto por página. A etapa de visão computacional automática é a evolução prevista do IAEbot 2.0.")
        uploads = st.file_uploader("Selecione PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
        if st.button("Indexar arquivos", use_container_width=True, disabled=not uploads):
            total = 0
            with st.spinner("Extraindo texto e gerando embeddings..."):
                for uploaded in uploads:
                    total += index_pdf_bytes(uploaded.name, uploaded.getvalue(), embedding_manager, storage)
            st.success(f"{total} chunks indexados com sucesso.")

    if st.button("Limpar histórico", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Demo acadêmica com dados sintéticos. Para documentação sensível, preserve execução local.")

st.markdown(
    """
<div class="iae-hero">
  <div class="iae-title">IAEbot 2.0 · Consulta multimodal de relatórios técnicos</div>
  <div class="iae-sub">RAG com evidências de texto, tabelas e elementos visuais, com fonte e página rastreáveis.</div>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Evidências", storage.count())
c2.metric("Top-k", settings.top_k)
c3.metric("Limiar", f"{settings.similarity_threshold:.2f}")
c4.metric("Modelo", settings.ollama_model)

st.markdown("**Perguntas sugeridas para a apresentação**")
examples = [
    "Qual foi a deformação máxima registrada no ensaio de tração, conforme o gráfico da página 12?",
    "Qual é o objetivo geral do IAEbot?",
    "Como o sistema reduz alucinações?",
    "Qual modelo de embeddings é utilizado?",
]
cols = st.columns(4)
for col, text in zip(cols, examples):
    if col.button(text, use_container_width=True):
        st.session_state.pending_prompt = text

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            st.caption(message["meta"])

prompt = st.chat_input("Pergunte algo sobre os documentos indexados...")
if not prompt and st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recuperando evidências e construindo a resposta..."):
            result = process_query(
                prompt,
                embedding_manager=embedding_manager,
                generator=generator,
                top_k=settings.top_k,
                similarity_threshold=settings.similarity_threshold,
            )
        st.markdown(result["answer"])
        latency = result.get("latency_s", 0.0)
        mode = result.get("generation_mode", "")
        meta = f"Geração: {mode} · {latency:.2f}s · {len(result.get('contexts', []))} evidências recuperadas"
        st.caption(meta)

        contexts = result.get("contexts", [])
        if contexts:
            with st.expander("🔎 Evidências e fontes recuperadas"):
                for i, item in enumerate(contexts, start=1):
                    loc = f" · p. {item['page_number']}" if item.get("page_number") else ""
                    tipo = item.get("content_type", "text")
                    st.markdown(f"**{i}. {item['source']}{loc} · {tipo} — similaridade {item['score']:.3f}**")
                    st.write(item["content"])
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "meta": meta})

st.divider()
st.markdown(
    "<div class='small-muted'>O IAEbot pode cometer erros. Verifique decisões importantes nas fontes citadas.</div>",
    unsafe_allow_html=True,
)
