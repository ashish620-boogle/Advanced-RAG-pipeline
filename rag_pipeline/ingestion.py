from pathlib import Path
from typing import Any, Iterable, List

from langchain_core.documents import Document
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".txt": "txt",
    ".md": "markdown",
    ".csv": "csv",
    ".sql": "sql",
}


def _load_file(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
    elif ext in {".txt", ".md", ".sql"}:
        loader = TextLoader(str(path), encoding="utf-8")
    elif ext == ".csv":
        loader = CSVLoader(str(path))
    else:
        return []

    documents = loader.load()
    for doc in documents:
        doc.metadata["source_file"] = path.name
        doc.metadata["file_type"] = SUPPORTED_EXTENSIONS.get(ext, ext.lstrip("."))
    return documents


def process_all_documents(data_directory: str | Path) -> List[Any]:
    """
    Load supported documents (pdf, txt, md, csv, sql) under a directory and attach metadata.
    """
    base_dir = Path(data_directory)
    files = [
        p
        for p in base_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    all_documents: List[Any] = []

    print(
        f"Found {len(files)} supported files to process in {base_dir.resolve()} "
        f"({', '.join(sorted(set(p.suffix.lower().lstrip('.') for p in files))) or 'none'})"
    )

    for file_path in files:
        print(f"Processing: {file_path.name}")
        try:
            docs = _load_file(file_path)
            all_documents.extend(docs)
            print(f"  Loaded {len(docs)} pages/chunks")
        except Exception as exc:
            print(f"  Error processing {file_path.name}: {exc}")

    print(f"Total documents loaded: {len(all_documents)}")
    return all_documents


# Backwards compatibility alias
def process_all_pdfs(data_directory: str | Path) -> List[Any]:
    return process_all_documents(data_directory)
