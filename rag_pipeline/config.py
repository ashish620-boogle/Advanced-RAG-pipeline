from pathlib import Path

DEFAULT_DATA_DIR = Path("data")
DEFAULT_VECTOR_STORE_DIR = DEFAULT_DATA_DIR / "vector_store"
DEFAULT_COLLECTION_NAME = "pdf_documents"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.2


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
