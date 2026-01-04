from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self._load_model()

    def _load_model(self) -> None:
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device="cpu")
        print(f"Model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
