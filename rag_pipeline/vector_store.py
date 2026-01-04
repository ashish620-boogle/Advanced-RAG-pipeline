from pathlib import Path
from typing import Any, List, Set
import uuid
import numpy as np
import chromadb


class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str | Path = Path("data/vector_store")):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "PDF document embedding for RAG",
                "hnsw:space": "cosine",
            },
        )
        print(f"Vector store ready. Collection: {self.collection_name} (count: {self.count()})")

    def add_documents(self, documents: List[Any], embeddings: np.ndarray) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        ids: List[str] = []
        metadatas: List[dict] = []
        document_texts: List[str] = []
        embeddings_list: List[List[float]] = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            document_texts.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=document_texts,
        )
        print(f"Added {len(documents)} documents. Total now: {self.count()}")

    def query(self, query_embedding: np.ndarray, n_results: int = 5) -> dict:
        return self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=n_results)

    def delete_by_source(self, source_file: str) -> None:
        if not self.collection:
            return
        try:
            ids = self.collection.get(where={"source_file": source_file}).get("ids", [])
        except Exception:
            ids = []
        if ids:
            self.collection.delete(ids=ids)
            print(f"Deleted {len(ids)} docs for source_file={source_file}")

    def list_sources(self) -> Set[str]:
        if not self.collection:
            return set()
        try:
            data = self.collection.get(include=["metadatas"])
            metadatas = data.get("metadatas", []) or []
        except Exception:
            return set()
        sources: Set[str] = set()
        def handle_md(md: Any) -> None:
            if isinstance(md, dict):
                src = md.get("source_file")
                if src:
                    sources.add(src)
        for entry in metadatas:
            if not entry:
                continue
            if isinstance(entry, list):
                for md in entry:
                    handle_md(md)
            else:
                handle_md(entry)
        return sources

    def count(self) -> int:
        try:
            return int(self.collection.count()) if self.collection else 0
        except Exception:
            return 0
