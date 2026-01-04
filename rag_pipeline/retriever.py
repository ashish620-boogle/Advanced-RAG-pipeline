from typing import Any, Dict, List
from .vector_store import VectorStore
from .embeddings import EmbeddingManager


class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager, top_k: int = 5):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.top_k = top_k

    def retrieve(self, query: str, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        print(f"Generating embedding for query: {query}")
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        print(f"Searching for top {self.top_k} similar documents...")
        try:
            results = self.vector_store.query(query_embedding=query_embedding, n_results=self.top_k)
        except Exception as exc:
            print(f"Error during retrieval: {exc}")
            return []

        retrieved_docs: List[Dict[str, Any]] = []

        if results.get("documents") and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                distance_value = float(distance)
                similarity_score = 1 / (1 + distance_value)
                if similarity_score < score_threshold:
                    print(f"Skipping document {doc_id} due to low similarity score: {similarity_score}")
                    continue
                retrieved_docs.append(
                    {
                        "id": doc_id,
                        "content": document,
                        "metadata": metadata,
                        "similarity_score": similarity_score,
                        "distance": distance_value,
                        "rank": i + 1,
                    }
                )

            print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
        else:
            print("No documents found.")

        return retrieved_docs
