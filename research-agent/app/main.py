"""FastAPI application — /query, /index, /health endpoints."""

import json
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()  # must happen before any LLM/litellm imports

from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import build_graph
from app.guardrails import validate_input
from app.rag.indexer import get_vectorstore, index_file

app = FastAPI(title="Research Assistant", version="1.0.0")
graph = build_graph()


class QueryRequest(BaseModel):
    question: str
    stream: bool = False


@app.post("/query")
async def query(request: QueryRequest):
    """Run the ReAct agent on a question.

    Returns a JSON response or an SSE stream of token chunks.
    """
    validate_input(request.question)

    initial_state = {
        "messages": [HumanMessage(content=request.question)],
        "sources": [],
        "step_count": 0,
    }

    if not request.stream:
        result = await graph.ainvoke(initial_state)
        return {
            "answer": result["messages"][-1].content,
            "sources": result.get("sources", []),
            "steps": result.get("step_count", 0),
        }

    async def event_generator():
        final_output = {}
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and chunk.content:
                    yield {"event": "token", "data": chunk.content}
            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                final_output = event["data"].get("output", {})

        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "sources": final_output.get("sources", []),
                    "steps": final_output.get("step_count", 0),
                }
            ),
        }

    return EventSourceResponse(event_generator())


@app.post("/index")
async def index(file: UploadFile = File(...)):
    """Upload and index a PDF or .txt file into the Chroma vector store."""
    filename = file.filename or ""
    if not (filename.endswith(".pdf") or filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        chunks = index_file(tmp_path)
    finally:
        os.unlink(tmp_path)

    return {"filename": filename, "chunks_indexed": chunks}


@app.get("/health")
async def health():
    """Return service status and current vector store document count."""
    vs = get_vectorstore()
    count = vs._collection.count()
    return {"status": "ok", "vectorstore_doc_count": count}
