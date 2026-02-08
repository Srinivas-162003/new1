import os
import sys
import tempfile
from pathlib import Path
from typing import List

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import TOP_K_DEFAULT, USE_VISION_DEFAULT
from ingestion.pdf_ingest import ingest_pdfs
from models import DocumentRecord, SectionRecord
from retrieval.vector_store import VectorStore
from utils.cache import compute_file_hash, index_paths
from agent.qa_agent import answer_query


def _save_documents(path: str, documents: List[DocumentRecord]) -> None:
    payload = []
    for doc in documents:
        payload.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "path": doc.path,
                "sections": [
                    {
                        "title": section.title,
                        "page": section.page,
                        "content": section.content,
                        "tables": section.tables,
                        "vision_notes": section.vision_notes,
                    }
                    for section in doc.sections
                ],
            }
        )
    with open(path, "w", encoding="utf-8") as handle:
        import json

        json.dump(payload, handle, indent=2)


def _load_documents(path: str) -> List[DocumentRecord]:
    import json

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    documents: List[DocumentRecord] = []
    for item in payload:
        doc = DocumentRecord(doc_id=item["doc_id"], title=item["title"], path=item["path"], sections=[])
        for section in item.get("sections", []):
            doc.sections.append(
                SectionRecord(
                    title=section["title"],
                    page=section["page"],
                    content=section["content"],
                    tables=section.get("tables", ""),
                    vision_notes=section.get("vision_notes", ""),
                )
            )
        documents.append(doc)

    return documents


st.set_page_config(page_title="Document Q&A Agent", layout="wide")

st.title("Document Q&A Agent")

api_key_present = bool(os.getenv("GEMINI_API_KEY"))
if not api_key_present:
    st.warning("GEMINI_API_KEY is not set. The app will not run until it is configured.")

st.sidebar.header("Settings")
use_vision = st.sidebar.checkbox("Use vision for figures/equations", value=USE_VISION_DEFAULT)
top_k = st.sidebar.slider("Top K", min_value=1, max_value=10, value=TOP_K_DEFAULT)
enable_arxiv = st.sidebar.checkbox("Enable Arxiv tool", value=True)

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if "store" not in st.session_state:
    st.session_state.store = None
    st.session_state.documents = []

if st.button("Build index"):
    if not uploaded_files:
        st.error("Please upload at least one PDF.")
    else:
        temp_dir = tempfile.mkdtemp()
        paths: List[str] = []
        for file in uploaded_files:
            file_path = os.path.join(temp_dir, file.name)
            with open(file_path, "wb") as handle:
                handle.write(file.getbuffer())
            paths.append(file_path)

        cache_key_parts = [compute_file_hash(path) for path in paths]
        cache_key = "_".join(cache_key_parts)
        index_path, docs_path = index_paths("data", cache_key)

        if os.path.exists(index_path) and os.path.exists(docs_path):
            store = VectorStore.load(index_path)
            documents = _load_documents(docs_path)
            st.info("Loaded cached index.")
        else:
            with st.spinner("Ingesting PDFs..."):
                documents, chunks = ingest_pdfs(paths, use_vision=use_vision)
                store = VectorStore()
                store.add(chunks)
            store.save(index_path)
            _save_documents(docs_path, documents)
            st.success("Index built and cached.")

        st.session_state.store = store
        st.session_state.documents = documents

query = st.text_input("Ask a question about your documents")
if st.button("Ask"):
    store = st.session_state.get("store")
    documents = st.session_state.get("documents", [])

    if not store or not documents:
        st.error("Please build the index first.")
    else:
        with st.spinner("Thinking..."):
            answer = answer_query(query, store, documents, enable_arxiv=enable_arxiv, top_k=top_k)

        st.subheader("Answer")
        st.write(answer.answer)

        if answer.citations:
            st.subheader("Citations")
            st.write(", ".join(answer.citations))

        if answer.extra and answer.extra.get("arxiv"):
            st.subheader("Arxiv Results")
            st.write(answer.extra["arxiv"])
