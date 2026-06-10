# Research Assistant Agent

A LangGraph ReAct agent with RAG, web search, and a summarization tool — served via FastAPI.

## Stack

| Component | Library |
|-----------|---------|
| LLM | `ChatLiteLLM` → `gemini/wine-gemini-2.5-lite` via CMU AI Gateway |
| Agent framework | LangGraph (custom ReAct graph) |
| Vector store | Chroma (local, `./chroma_db`) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, runs locally) |
| Web search | DuckDuckGoSearchRun (no API key) |
| API | FastAPI + uvicorn |
| Streaming | SSE via `sse-starlette` |

## Setup

### 1. Create and activate a virtual environment

```bash
cd research-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env` and fill in your keys:

```bash
cp .env .env.local   # optional — .env already has defaults
```

Required variables:

```
LITELLM_API_KEY=<your-key>
LITELLM_API_BASE=https://ai-gateway.andrew.cmu.edu/v1
LANGCHAIN_API_KEY=<your-langsmith-key>
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=research-agent
```

### 3. Seed the vector store

```bash
python scripts/seed_docs.py
```

### 4. Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

## Endpoints

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status":"ok","vectorstore_doc_count":12}
```

### `POST /query` — blocking

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval-augmented generation?", "stream": false}'
```

```json
{
  "answer": "...",
  "sources": ["sample_docs/ai_overview.txt"],
  "steps": 3
}
```

### `POST /query` — streaming (SSE)

```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise recent advances in AI", "stream": true}'
```

Each SSE token arrives as:
```
event: token
data: <chunk>
```

Final event:
```
event: done
data: {"sources": [...], "steps": 4}
```

### `POST /index` — upload a document

```bash
# Upload a PDF
curl -X POST http://localhost:8000/index \
  -F "file=@my_paper.pdf"

# Upload a text file
curl -X POST http://localhost:8000/index \
  -F "file=@notes.txt"
```

```json
{"filename": "my_paper.pdf", "chunks_indexed": 42}
```

## Project layout

```
research-agent/
├── app/
│   ├── main.py            # FastAPI app and endpoints
│   ├── guardrails.py      # Input length + injection checks
│   ├── agent/
│   │   ├── graph.py       # LangGraph ReAct graph
│   │   ├── state.py       # AgentState TypedDict
│   │   └── tools.py       # web_search, search_docs, summarize
│   └── rag/
│       ├── indexer.py     # Chroma + HuggingFace + splitter
│       └── retriever.py   # MMR retriever
├── scripts/
│   └── seed_docs.py       # One-shot indexer for sample docs
├── sample_docs/
│   └── ai_overview.txt
├── chroma_db/             # Created at runtime
├── .env
└── requirements.txt
```

## Guardrails

Requests are rejected with HTTP 400 if the input:
- Exceeds 2000 characters
- Contains prompt-injection patterns (`ignore previous instructions`, `you are now`, `jailbreak`)

## Agent behaviour

The ReAct agent loops through: `agent → tools → agent → …` up to **8 iterations**.
On hitting the limit it returns *"Max iterations reached."* and terminates.
