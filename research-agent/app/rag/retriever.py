"""MMR retriever over the Chroma vector store."""

from langchain_core.vectorstores import VectorStoreRetriever

from app.rag.indexer import get_vectorstore


def get_retriever() -> VectorStoreRetriever:
    """Return an MMR retriever (k=4, fetch_k=20) over the local Chroma store."""
    vs = get_vectorstore()
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20},
    )
