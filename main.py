from pathlib import Path
import os
from dotenv import load_dotenv
import numpy as np

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


def sync_vector_store(vector_store: VectorStore, chunks, embeddings, source_files: set[str]) -> None:
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


def build_pipeline(
    data_dir: Path = DEFAULT_DATA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    persist_directory: Path = DEFAULT_VECTOR_STORE_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    top_k: int = DEFAULT_TOP_K,
) -> AdvancedRAGPipeline | None:
    docs = process_all_documents(data_dir)
    if not docs:
        print("No PDFs found. Add files under the data/ directory and rerun.")
        return None

    chunks = split_documents(docs, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
    embedding_manager = EmbeddingManager(model_name=model_name)
    embeddings = embedding_manager.generate_embeddings([doc.page_content for doc in chunks])

    vector_store = VectorStore(collection_name=collection_name, persist_directory=persist_directory)
    source_files = {doc.metadata.get("source_file") for doc in docs}
    sync_vector_store(vector_store, chunks, embeddings, source_files)

    retriever = RAGRetriever(vector_store=vector_store, embedding_manager=embedding_manager, top_k=top_k)
    llm = load_groq_llm()
    return AdvancedRAGPipeline(retriever, llm)


def main() -> None:
    load_dotenv()

    pipeline = build_pipeline()
    if not pipeline:
        return

    question = os.environ.get(
        "RAG_QUESTION",
        "What is CNN training procedure described under Unsupervised Domain Adaptation?",
    )
    result = pipeline.generate(question, score_threshold=DEFAULT_SCORE_THRESHOLD)
    print(f"Question: {question}\n")
    print(f"Answer:\n{result['answer']}\n")
    if result.get("sources"):
        print("Sources:")
        for src in result["sources"]:
            print(f"- {src['source']} (page {src['page']}) score={src['score']:.3f}")


if __name__ == "__main__":
    main()
