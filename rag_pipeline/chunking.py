from typing import Any, List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
    if not documents:
        print("No documents to split.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    split_docs = text_splitter.split_documents(documents=documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    return split_docs
