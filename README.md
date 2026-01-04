# RAG Sample Project

This project demonstrates an end-to-end Retrieval Augmented Generation (RAG) pipeline in Python, built with LangChain components, Sentence Transformers for embeddings, and ChromaDB as the persistent vector store. It is organized as a small library (`rag_pipeline/`) plus a runnable orchestrator (`main.py`) that can ingest local documents, chunk and embed them, store them in a vector database, and answer questions with grounding, citations, streaming, conversation history, and summarization.

---

## Conceptual Overview

### Retrieval Augmented Generation (RAG)
RAG pairs a language model with an external knowledge index. Instead of relying on the model’s parametric memory alone, we:
1. **Ingest** raw documents and normalize them into a consistent document schema.
2. **Chunk** documents into manageable passages.
3. **Embed** each chunk into a high-dimensional vector.
4. **Store** vectors in a vector database for fast similarity search.
5. **Retrieve** relevant chunks for a user query.
6. **Generate** an answer grounded in retrieved context, with citations back to sources.

This architecture improves factuality, lets you update knowledge without retraining the model, and enables domain-specific QA, summarization, and chat-style interactions.

### Data Ingestion (Multi-format)
- **Supported types:** PDF, TXT, MD, CSV, SQL.
- **Loader behavior:** Each file is read into a list of LangChain `Document` objects. The pipeline attaches metadata such as `source_file` (original filename) and `file_type` so later stages can filter, trace, and cite sources.
- **Recursive discovery:** All supported files under `data/` are scanned on each run.

### Chunking
- Documents are split with `RecursiveCharacterTextSplitter` into overlapping chunks (defaults: 1000 chars, 200 overlap).
- Overlap preserves context across boundaries; the chunk size balances retrieval recall and prompt length.

### Embeddings
- Uses Sentence Transformers (`all-MiniLM-L6-v2` by default) to convert text chunks into vectors.
- Embeddings are deterministic: same text → same vector (given same model). They are computed once per run for newly added files.

### Vector Store
- **Backend:** ChromaDB persistent store at `data/vector_store/`.
- **Index space:** Cosine distance (`hnsw:space = "cosine"`).
- **Persistence:** The store survives across runs. `main.py` reuses the same collection, adding new documents and deleting those whose source files were removed.
- **Metadata:** Each stored item keeps `source_file`, `file_type`, `doc_index`, `content_length`, etc., enabling filtering and citations.

### Retrieval
- Queries are embedded with the same embedding model.
- Top-k (default 5) nearest neighbors are fetched from Chroma. Distances are mapped to similarity scores in (0, 1] using `1 / (1 + distance)` to avoid negatives.
- A score threshold can filter low-signal hits.

### Generation
- **LLM:** Groq-hosted model (default `qwen/qwen3-32b`) via `langchain_groq.ChatGroq`.
- **Prompting:** Retrieved context is injected into a simple instruction prompt. If no context is found, a fallback message is returned.
- **Advanced pipeline:**
  - Streaming token output.
  - Citations (source file + page if available).
  - Conversation history (stored in memory; trims to a configurable window).
  - Summarization of history or retrieved context.

### Sync Behavior on Each Run
- `main.py` scans `data/`, loads all supported files, splits, and embeds.
- The vector store is **incrementally maintained**:
  - **New files:** chunks are embedded and added.
  - **Deleted files:** their vectors are removed via `source_file` matching.
  - **Unchanged files:** left as-is (content changes are not detected unless filename changes or the store is cleared).

---

## Project Layout
```
rag_pipeline/
  config.py          # Defaults (paths, chunk sizes, model names, thresholds)
  ingestion.py       # Multi-format loaders (pdf, txt, md, csv, sql)
  chunking.py        # Recursive chunking with overlap
  embeddings.py      # SentenceTransformer wrapper
  vector_store.py    # Chroma persistent store helpers, add/delete/list sources
  retriever.py       # Query embedding + similarity scoring
  pipeline.py        # Simple RAG + AdvancedRAGPipeline (streaming, history, summaries)
  llm.py             # Groq LLM loader

main.py              # Orchestrates ingestion → embeddings → vector DB sync → QA
data/                # Place your documents here (pdf/txt/md/csv/sql)
data/vector_store/   # Chroma persistence (auto-created)
.env                 # Holds secrets like GROQ_API_KEY
requirements.txt     # Python deps
```

---

## Configuration & Defaults
- Data directory: `data/`
- Vector store directory: `data/vector_store/`
- Collection name: `pdf_documents`
- Embedding model: `all-MiniLM-L6-v2`
- Chunk size / overlap: 1000 / 200
- Retrieval: top_k = 5, score_threshold default 0.2 (can be overridden in calls)
- LLM: `qwen/qwen3-32b` via Groq (requires `GROQ_API_KEY`)

Adjust defaults in `rag_pipeline/config.py` or by passing args into `build_pipeline` (see `main.py`).

---


## How to Run
1. **Install dependencies** (preferably in a virtualenv):
   ```bash
   pip install -r requirements.txt
   ```
   If you see `ModuleNotFoundError: langchain_core`, ensure the install completed or run `pip install langchain-core`.
2. **Set your Groq key** in `.env`:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. **Add documents** under `data/` (any mix of pdf/txt/md/csv/sql).
4. **Run the pipeline (CLI example)**:
   ```bash
   python main.py
   ```
   - On each run, new files are added, removed files are deleted from the vector store, and existing entries are reused.
   - The script asks the question in `RAG_QUESTION` (env) or defaults to *?What is CNN training procedure described under Unsupervised Domain Adaptation??*.

### Streamlit demo app (chatbot UI)
Live demo: https://ash-advanced-rag-pipeline.streamlit.app/

1. Install deps (as above) and ensure `.env` has `GROQ_API_KEY` if running locally.
2. Launch locally:
   ```bash
   streamlit run app.py
   ```
3. In the sidebar, upload or delete documents (pdf/txt/md/csv/sql) and optionally click **Rebuild index**. The app syncs the vector store to reflect additions/removals.
4. Use the chat box to ask questions. Answers include sources; chat history stays on the page. Streaming is handled by the pipeline?s generate call (non-streaming in UI for simplicity).

## Notes & Customization
- **Refreshing embeddings for changed files:** The current sync detects added/removed files by name. If file contents change but names don’t, delete `data/vector_store/` or change the `collection_name` to rebuild.
- **Streaming / history / summaries:** Use `AdvancedRAGPipeline` methods (`generate`, `stream`, `summarize_history`, `summarize_context`) from `rag_pipeline.pipeline` if you want interactive or chat-style behavior.
- **Different LLM or embedding model:** Change `load_groq_llm` arguments in `rag_pipeline/llm.py` or the `model_name` passed to `EmbeddingManager`.

---

## Quick Code Usage (modular)
```python
from rag_pipeline import process_all_documents, split_documents, EmbeddingManager, VectorStore, RAGRetriever, AdvancedRAGPipeline
from rag_pipeline.llm import load_groq_llm

docs = process_all_documents("data")
chunks = split_documents(docs)
emb_mgr = EmbeddingManager()
embs = emb_mgr.generate_embeddings([c.page_content for c in chunks])

vs = VectorStore()
vs.add_documents(chunks, embs)  # first-time build; later runs can sync like main.py

retriever = RAGRetriever(vs, emb_mgr)
llm = load_groq_llm()
rag = AdvancedRAGPipeline(retriever, llm)

result = rag.generate("Your question here", return_context=True)
print(result["answer"])
print(result["sources"])
```
