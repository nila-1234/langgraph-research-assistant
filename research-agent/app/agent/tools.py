"""Tools available to the ReAct research agent."""

import os
import re
from typing import List

import litellm
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_litellm import ChatLiteLLM


def _get_llm() -> ChatLiteLLM:
    """Return a configured ChatLiteLLM instance routing through the CMU AI Gateway."""
    return ChatLiteLLM(
        model="openai/wine-gemini-3-flash-preview",
        api_key=os.getenv("LITELLM_API_KEY"),
        api_base=os.getenv("LITELLM_API_BASE"),
    )


_ddg = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web for current information using DuckDuckGo."""
    return _ddg.run(query)


@tool
def search_docs(query: str) -> str:
    """Search indexed local documents for relevant context using RAG."""
    from app.rag.retriever import get_retriever  # lazy import avoids circular dep

    retriever = get_retriever()
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documents found."
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


@tool
def summarize(texts: List[str]) -> str:
    """Summarize a list of text passages into a concise bullet-point list."""
    llm = _get_llm()
    combined = "\n\n---\n\n".join(texts)
    response = llm.invoke(
        f"Summarize the following passages as a concise bullet-point list:\n\n{combined}"
    )
    return response.content


TOOLS = [web_search, search_docs, summarize]
