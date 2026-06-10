"""Index sample_docs/ai_overview.txt into the Chroma vector store on first run."""

import os
import sys

from dotenv import load_dotenv

# Allow running from the repo root: python scripts/seed_docs.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.rag.indexer import get_vectorstore, index_file

SAMPLE_DOC = os.path.join(os.path.dirname(__file__), "..", "sample_docs", "ai_overview.txt")


def main() -> None:
    """Index the sample doc if the vector store is empty."""
    vs = get_vectorstore()
    existing = vs._collection.count()
    if existing > 0:
        print(f"Vector store already has {existing} documents. Skipping seed.")
        return

    path = os.path.abspath(SAMPLE_DOC)
    if not os.path.exists(path):
        print(f"Sample doc not found: {path}")
        sys.exit(1)

    count = index_file(path)
    print(f"Indexed {count} chunks from {path}")


if __name__ == "__main__":
    main()
