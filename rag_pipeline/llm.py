import os
from langchain_groq import ChatGroq


def load_groq_llm(model: str = "qwen/qwen3-32b", temperature: float = 0.1, max_tokens: int = 1024, api_key: str | None = None) -> ChatGroq:
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Set GROQ_API_KEY in your environment (e.g., in .env)")
    return ChatGroq(api_key=key, model=model, temperature=temperature, max_tokens=max_tokens)
