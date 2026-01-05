
import os
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_STORE_DIR,
    AdvancedRAGPipeline,
    EmbeddingManager,
    RAGRetriever,
    VectorStore,
    process_all_documents,
    split_documents,
)
from rag_pipeline.llm import load_groq_llm


load_dotenv()

DATA_DIR = Path(DEFAULT_DATA_DIR)
SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".csv", ".sql"}


def sync_vector_store(vector_store: VectorStore, chunks: Iterable, embeddings, source_files: set[str]) -> None:
    existing_sources = vector_store.list_sources()
    to_delete = existing_sources - source_files
    to_add = source_files - existing_sources

    for src in to_delete:
        vector_store.delete_by_source(src)

    if to_add:
        docs_to_add = []
        embs_to_add = []
        for doc, emb in zip(chunks, embeddings):
            if doc.metadata.get("source_file") in to_add:
                docs_to_add.append(doc)
                embs_to_add.append(emb)
        if docs_to_add:
            vector_store.add_documents(docs_to_add, np.array(embs_to_add))


@st.cache_resource(show_spinner="Building RAG pipeline...")
def build_pipeline_cached(refresh_token: float) -> AdvancedRAGPipeline | None:
    docs = process_all_documents(DATA_DIR)
    if not docs:
        return None

    chunks = split_documents(docs, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
    embedding_manager = EmbeddingManager(model_name=DEFAULT_MODEL_NAME)
    embeddings = embedding_manager.generate_embeddings([doc.page_content for doc in chunks])

    vector_store = VectorStore(collection_name=DEFAULT_COLLECTION_NAME, persist_directory=DEFAULT_VECTOR_STORE_DIR)
    source_files = {doc.metadata.get("source_file") for doc in docs}
    sync_vector_store(vector_store, chunks, embeddings, source_files)

    retriever = RAGRetriever(vector_store=vector_store, embedding_manager=embedding_manager, top_k=DEFAULT_TOP_K)
    llm = load_groq_llm()
    return AdvancedRAGPipeline(retriever, llm)


def list_supported_files() -> list[Path]:
    return [
        p
        for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]


def save_uploaded_files(files) -> list[Path]:
    saved = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        ext = Path(f.name).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue
        target = DATA_DIR / f.name
        with open(target, "wb") as out:
            out.write(f.getbuffer())
        saved.append(target)
    return saved

def parse_think_blocks(text: str):
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    think_blocks = pattern.findall(text or "")
    visible = pattern.sub("", text or "").strip()
    return visible, think_blocks


def delete_files(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except Exception as exc:
            st.warning(f"Could not delete {path.name}: {exc}")


def main() -> None:
    st.set_page_config(page_title="Advanced RAG Chat", layout="wide")
    st.title("Advanced RAG Chat")
    st.caption("Upload or delete documents, sync the vector store, and chat with grounded answers.")

    if "refresh_token" not in st.session_state:
        st.session_state["refresh_token"] = time.time()
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    with st.sidebar:
        st.header("Manage documents")
        uploads = st.file_uploader(
            "Add documents (pdf, txt, md, csv, sql)",
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTS],
            accept_multiple_files=True,
        )
        added = []
        if uploads:
            added = save_uploaded_files(uploads)
            if added:
                st.success(f"Added {len(added)} file(s).")
                st.session_state["refresh_token"] = time.time()

        existing_files = list_supported_files()
        if existing_files:
            to_delete = st.multiselect(
                "Delete selected files",
                options=existing_files,
                format_func=lambda p: str(p.relative_to(DATA_DIR)),
            )
            if st.button("Delete selected"):
                delete_files(to_delete)
                if to_delete:
                    st.success(f"Deleted {len(to_delete)} file(s).")
                    st.session_state["refresh_token"] = time.time()
        else:
            st.info("No documents found yet.")

        if st.button("Rebuild index"):
            st.session_state["refresh_token"] = time.time()

    pipeline = build_pipeline_cached(st.session_state["refresh_token"])
    if not pipeline:
        st.warning("No documents available. Upload supported files to get started.")
        return

    show_think = st.checkbox("Reasoning", value=False, key="show_think_toggle")

    st.divider()
    st.subheader("Chat")
    for turn in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        visible_answer, think_blocks = parse_think_blocks(turn["answer"])
        with st.chat_message("assistant"):
            st.markdown(visible_answer or turn["answer"])
            if turn.get("sources"):
                st.caption("Sources: " + "; ".join(f"{s['source']}" for s in turn["sources"]))
            if show_think and think_blocks:
                with st.expander("Reasoning"):
                    for i, blk in enumerate(think_blocks, 1):
                        st.markdown(f"**Block {i}:**\n\n{blk}")


    prompt = st.chat_input("Ask a question about your documents")
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = pipeline.generate(prompt, score_threshold=DEFAULT_SCORE_THRESHOLD, return_context=False)
            visible_answer, think_blocks = parse_think_blocks(result["answer"])
            st.markdown(visible_answer or result["answer"])
            if result.get("sources"):
                st.caption("Sources: " + "; ".join(f"{s['source']}" for s in result["sources"]))
            if show_think and think_blocks:
                with st.expander("Show reasoning"):
                    for i, blk in enumerate(think_blocks, 1):
                        st.markdown(f"**Block {i}:**\n\n{blk}")

        st.session_state["chat_history"].append(
            {
                "question": prompt,
                "answer": result["answer"],
                "sources": result.get("sources", []),
            }
        )


if __name__ == "__main__":
    main()
