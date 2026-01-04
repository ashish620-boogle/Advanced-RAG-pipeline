from typing import Any, Dict, List
from .retriever import RAGRetriever


def rag_simple(query: str, retriever: RAGRetriever, llm) -> str:
    retrieved_docs = retriever.retrieve(query)
    context = "\n\n".join([doc["content"] for doc in retrieved_docs]) if retrieved_docs else ""
    if not context:
        return "I'm sorry, I couldn't find any relevant information to answer your question."

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


def rag_advanced(query: str, retriever: RAGRetriever, llm, score_threshold: float = 0.2, return_context: bool = False) -> Dict[str, Any]:
    retrieved_docs = retriever.retrieve(query, score_threshold=score_threshold)

    if not retrieved_docs:
        empty = {"answer": "I'm sorry, I couldn't find any relevant information to answer your question.", "sources": [], "confidence": 0.0}
        if return_context:
            empty["context"] = ""
        return empty

    context = "\n\n".join([doc["content"] for doc in retrieved_docs])
    sources = [
        {
            "source": doc["metadata"].get("source_file", doc["metadata"].get("doc_index", "unknown")),
            "page": doc["metadata"].get("page", "unknown"),
            "score": doc["similarity_score"],
            "preview": doc["content"][:300] + "...",
        }
        for doc in retrieved_docs
    ]
    confidence = max(doc["similarity_score"] for doc in retrieved_docs)

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = llm.invoke(prompt)

    output = {"answer": getattr(response, "content", str(response)), "sources": sources, "confidence": confidence}
    if return_context:
        output["context"] = context
    return output


class AdvancedRAGPipeline:
    """Structured RAG helper with streaming, citations, history, and summarisation support."""

    def __init__(self, retriever: RAGRetriever, llm, system_prompt: str | None = None, history_limit: int = 6):
        self.retriever = retriever
        self.llm = llm
        self.system_prompt = system_prompt or "You are a helpful assistant that grounds answers in the provided context and cites the matching sources."
        self.history_limit = history_limit
        self.history: List[Dict[str, Any]] = []

    def _append_history(self, question: str, answer: str, sources: List[Dict[str, Any]]) -> None:
        self.history.append({"question": question, "answer": answer, "sources": sources})
        self.history = self.history[-self.history_limit:]

    def _format_history(self) -> str:
        if not self.history:
            return ""
        blocks = []
        for turn in self.history[-self.history_limit:]:
            blocks.append(f"Q: {turn['question']}")
            blocks.append(f"A: {turn['answer']}")
        return "\n".join(blocks)

    def _prepare_context(self, query: str, score_threshold: float):
        docs = self.retriever.retrieve(query, score_threshold=score_threshold)
        if not docs:
            return "", [], 0.0
        context = "\n\n".join([doc["content"] for doc in docs])
        sources = [
            {
                "source": doc["metadata"].get("source_file", doc["metadata"].get("doc_index", "unknown")),
                "page": doc["metadata"].get("page", "unknown"),
                "score": doc["similarity_score"],
                "preview": doc["content"][:200],
            }
            for doc in docs
        ]
        confidence = max(doc["similarity_score"] for doc in docs)
        return context, sources, confidence

    def _build_prompt(self, query: str, context: str, use_history: bool) -> str:
        history_block = self._format_history() if use_history else ""
        sections = [
            self.system_prompt,
            "Use the provided context to answer the question. If the context is insufficient, say you do not know.",
        ]
        if history_block:
            sections.append(f"Conversation history:\n{history_block}")
        if context:
            sections.append(f"Context:\n{context}")
        sections.append(f"Question: {query}")
        sections.append("Answer:")
        return "\n\n".join([part for part in sections if part])

    def generate(self, query: str, score_threshold: float = 0.2, return_context: bool = False, use_history: bool = True) -> Dict[str, Any]:
        context, sources, confidence = self._prepare_context(query, score_threshold)
        if not context:
            fallback = {"answer": "I could not find relevant information in the knowledge base.", "sources": [], "confidence": 0.0}
            if return_context:
                fallback["context"] = ""
            return fallback

        prompt = self._build_prompt(query, context, use_history)
        response = self.llm.invoke(prompt)
        answer_text = getattr(response, "content", str(response))
        self._append_history(query, answer_text, sources)

        result = {"answer": answer_text, "sources": sources, "confidence": confidence}
        if return_context:
            result["context"] = context
        return result

    def stream(self, query: str, score_threshold: float = 0.2, use_history: bool = True) -> Dict[str, Any]:
        context, sources, confidence = self._prepare_context(query, score_threshold)
        if not context:
            def _no_context():
                yield "I could not find relevant information in the knowledge base."
            return {"stream": _no_context(), "sources": [], "confidence": 0.0, "context": ""}

        prompt = self._build_prompt(query, context, use_history)

        def _token_generator():
            collected = []
            for chunk in self.llm.stream(prompt):
                token = getattr(chunk, "content", str(chunk))
                collected.append(token)
                yield token
            final_answer = "".join(collected)
            self._append_history(query, final_answer, sources)

        return {"stream": _token_generator(), "sources": sources, "confidence": confidence, "context": context}

    def summarize_history(self, max_turns: int = 4) -> str:
        if not self.history:
            return "No history yet."
        excerpt = self.history[-max_turns:]
        transcript = "\n".join([f"Q: {turn['question']}\nA: {turn['answer']}" for turn in excerpt])
        prompt = f"Summarize the following conversation in a few sentences:\n{transcript}\n\nSummary:"
        response = self.llm.invoke(prompt)
        return getattr(response, "content", str(response))

    def summarize_context(self, query: str, score_threshold: float = 0.2, max_words: int = 120) -> Dict[str, Any]:
        context, sources, confidence = self._prepare_context(query, score_threshold)
        if not context:
            return {"summary": "No relevant context found to summarize.", "sources": [], "confidence": 0.0}
        prompt = f"Summarize the following context in {max_words} words or less, keeping key details and citations:\n{context}\n\nSummary:"
        response = self.llm.invoke(prompt)
        summary_text = getattr(response, "content", str(response))
        self._append_history(f"Summary request: {query}", summary_text, sources)
        return {"summary": summary_text, "sources": sources, "confidence": confidence}
