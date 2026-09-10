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
    page_title="IAEbot 2.0",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* ---------- Base ---------- */
:root {
  --navy: #0b2e59;
  --blue: #155a8a;
  --cyan: #2d8bb8;
  --ink: #172233;
  --muted: #68788d;
  --line: #e6ecf2;
  --soft: #f6f9fc;
  --soft-blue: #edf5fb;
}

html, body, [class*="css"] {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
  background:
    radial-gradient(circle at 88% 2%, rgba(45,139,184,.07), transparent 26rem),
    #ffffff;
}

.block-container {
  max-width: 1120px;
  padding-top: 2rem;
  padding-bottom: 7rem;
}

[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stToolbar"] {
  visibility: hidden;
  height: 0;
}

#MainMenu, footer {
  visibility: hidden;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
  background: #0b2e59;
  border-right: 0;
}

[data-testid="stSidebar"] * {
  color: #eef5fb;
}

[data-testid="stSidebar"] hr {
  border-color: rgba(255,255,255,.13);
}

[data-testid="stSidebar"] .stCaptionContainer p {
  color: rgba(238,245,251,.68);
}

[data-testid="stSidebar"] button {
  border-radius: 10px;
}

.side-brand {
  display: flex;
  align-items: center;
  gap: .75rem;
  margin: .2rem 0 .3rem 0;
}

.side-logo {
  width: 38px;
  height: 38px;
  border-radius: 12px;
  display: grid;
  place-items: center;
  background: rgba(255,255,255,.12);
  border: 1px solid rgba(255,255,255,.14);
  font-size: 1.05rem;
}

.side-name {
  font-size: 1.1rem;
  font-weight: 750;
  letter-spacing: -.02em;
}

.side-version {
  color: rgba(238,245,251,.64);
  font-size: .78rem;
  margin-top: -.05rem;
}

.status-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: .48rem 0;
  font-size: .84rem;
  border-bottom: 1px solid rgba(255,255,255,.08);
}

.status-value {
  color: #fff;
  font-weight: 650;
}

/* ---------- Main header ---------- */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 2.4rem;
}

.brand-wrap {
  display: flex;
  align-items: center;
  gap: .8rem;
}

.brand-mark {
  width: 44px;
  height: 44px;
  border-radius: 14px;
  display: grid;
  place-items: center;
  color: white;
  background: linear-gradient(145deg, #0b2e59, #1a6b9f);
  box-shadow: 0 8px 24px rgba(11,46,89,.16);
  font-weight: 800;
}

.brand-title {
  margin: 0;
  color: var(--navy);
  font-size: 1.25rem;
  font-weight: 800;
  letter-spacing: -.025em;
}

.brand-subtitle {
  margin-top: .08rem;
  color: var(--muted);
  font-size: .83rem;
}

.online-pill {
  display: inline-flex;
  align-items: center;
  gap: .45rem;
  border: 1px solid #dfe9f1;
  border-radius: 999px;
  padding: .42rem .72rem;
  font-size: .78rem;
  color: #446176;
  background: rgba(255,255,255,.82);
}

.online-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #2e9b66;
  box-shadow: 0 0 0 3px rgba(46,155,102,.10);
}

/* ---------- Hero ---------- */
.hero {
  margin: 1.1rem auto 2.3rem auto;
  max-width: 830px;
  text-align: center;
}

.hero-kicker {
  color: var(--blue);
  text-transform: uppercase;
  letter-spacing: .14em;
  font-weight: 750;
  font-size: .72rem;
  margin-bottom: .65rem;
}

.hero-title {
  color: var(--ink);
  font-size: clamp(2rem, 4vw, 3.1rem);
  line-height: 1.08;
  letter-spacing: -.045em;
  font-weight: 800;
  margin-bottom: .9rem;
}

.hero-copy {
  max-width: 690px;
  margin: 0 auto;
  color: var(--muted);
  font-size: 1rem;
  line-height: 1.65;
}

/* ---------- Chat ---------- */
.chat-shell {
  max-width: 900px;
  margin: 0 auto;
}

[data-testid="stChatMessage"] {
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: .25rem .35rem;
  margin-bottom: .8rem;
  background: #fff;
  box-shadow: 0 6px 22px rgba(28,46,69,.035);
}

[data-testid="stChatMessage"] p {
  line-height: 1.6;
}

[data-testid="stChatInput"] {
  max-width: 900px;
  margin: 0 auto;
}

[data-testid="stChatInput"] textarea {
  border-radius: 16px !important;
}

.source-chip {
  display: inline-block;
  margin: .15rem .35rem .15rem 0;
  padding: .28rem .55rem;
  border-radius: 999px;
  background: var(--soft-blue);
  border: 1px solid #d8e8f3;
  color: #275d80;
  font-size: .76rem;
  font-weight: 650;
}

.empty-note {
  max-width: 720px;
  margin: 2rem auto 0 auto;
  padding: 1rem 1.1rem;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(248,251,253,.86);
  color: var(--muted);
  font-size: .88rem;
  line-height: 1.55;
  text-align: center;
}

.small-muted {
  color: var(--muted);
  font-size: .82rem;
}

@media (max-width: 720px) {
  .block-container { padding-top: 1rem; }
  .topbar { margin-bottom: 1.4rem; }
  .online-pill { display: none; }
  .hero { margin-top: .6rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def _secret_mapping():
    try:
        return st.secrets
    except Exception:
        return None


@st.cache_resource(show_spinner="Inicializando modelos e base de conhecimento...")
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
    with st.spinner("Preparando a base de conhecimento..."):
        for item in load_demo_corpus():
            index_document(item["filename"], item["text"], embeddings, storage)


settings, storage, embedding_manager, generator = load_resources()
ensure_demo_corpus(storage, embedding_manager)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
          <div class="side-logo">✦</div>
          <div>
            <div class="side-name">IAEbot</div>
            <div class="side-version">versão 2.0</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Assistente para consulta de relatórios técnicos")
    st.divider()

    st.markdown("##### Sistema")
    st.markdown(
        f"""
        <div class="status-row"><span>Base</span><span class="status-value">{storage.backend}</span></div>
        <div class="status-row"><span>Modelo</span><span class="status-value">{settings.ollama_model}</span></div>
        <div class="status-row"><span>Evidências</span><span class="status-value">{storage.count()}</span></div>
        <div class="status-row"><span>Recuperação</span><span class="status-value">top-{settings.top_k}</span></div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    with st.expander("Adicionar documentos"):
        st.caption("Indexe PDFs para ampliar a base consultável.")
        uploads = st.file_uploader(
            "Selecione PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if st.button("Indexar documentos", use_container_width=True, disabled=not uploads):
            total = 0
            with st.spinner("Processando documentos..."):
                for uploaded in uploads:
                    total += index_pdf_bytes(
                        uploaded.name,
                        uploaded.getvalue(),
                        embedding_manager,
                        storage,
                    )
            st.success(f"{total} trechos indexados.")

    if st.button("Nova conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("As respostas são geradas a partir das evidências recuperadas na base de conhecimento.")

st.markdown(
    f"""
    <div class="topbar">
      <div class="brand-wrap">
        <div class="brand-mark">IAE</div>
        <div>
          <div class="brand-title">IAEbot 2.0</div>
          <div class="brand-subtitle">Análise multimodal de relatórios técnicos</div>
        </div>
      </div>
      <div class="online-pill"><span class="online-dot"></span>Sistema disponível</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Gestão do conhecimento técnico</div>
          <div class="hero-title">Consulte relatórios com linguagem natural.</div>
          <div class="hero-copy">
            O IAEbot combina recuperação semântica e modelos de linguagem para localizar,
            correlacionar e explicar informações presentes em textos, tabelas e elementos visuais.
          </div>
        </div>
        <div class="empty-note">
          Digite uma pergunta no campo abaixo. Quando houver evidência suficiente, a resposta incluirá
          a origem da informação e a página correspondente.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("meta"):
            st.caption(message["meta"])

prompt = st.chat_input("Pergunte sobre os relatórios indexados...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Analisando os documentos..."):
            result = process_query(
                prompt,
                embedding_manager=embedding_manager,
                generator=generator,
                top_k=settings.top_k,
                similarity_threshold=settings.similarity_threshold,
            )

        st.markdown(result["answer"])

        contexts = result.get("contexts", [])
        latency = result.get("latency_s", 0.0)
        mode = result.get("generation_mode", "")

        if contexts:
            labels = []
            seen = set()
            for item in contexts:
                source = item.get("source", "documento")
                page = item.get("page_number")
                key = (source, page)
                if key in seen:
                    continue
                seen.add(key)
                label = source if not page else f"{source} · p. {page}"
                labels.append(label)

            chips = "".join(f'<span class="source-chip">{label}</span>' for label in labels[:4])
            st.markdown(chips, unsafe_allow_html=True)

            with st.expander("Ver evidências utilizadas"):
                for i, item in enumerate(contexts, start=1):
                    loc = f" · página {item['page_number']}" if item.get("page_number") else ""
                    tipo = item.get("content_type", "text")
                    figura = f" · {item['figure_label']}" if item.get("figure_label") else ""
                    st.markdown(
                        f"**{i}. {item['source']}{loc}{figura}**  \n"
                        f"Tipo: `{tipo}` · similaridade: `{item['score']:.3f}`"
                    )
                    st.write(item["content"])
                    if i < len(contexts):
                        st.divider()

        meta = f"{len(contexts)} evidência(s) · {latency:.2f}s"
        if mode:
            meta += f" · {mode}"
        st.caption(meta)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "meta": meta,
        }
    )

st.markdown("</div>", unsafe_allow_html=True)
