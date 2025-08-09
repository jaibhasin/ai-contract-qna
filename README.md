# HackRX Intelligent Query–Retrieval API

Production-grade FastAPI application for intelligent document query and retrieval over insurance/legal/HR/compliance documents (PDF, DOCX, Emails).

- Ingestion from HTTP(S) and local file URLs.
- Robust parsing and chunking using LangChain loaders and RecursiveCharacterTextSplitter.
- Embeddings with Google Gemini (text-embedding-004).
- Vector search with FAISS (cosine similarity via inner product on normalized vectors).
- Semantic retrieval, clause selection, and explainable answers with source tracing.

## Architecture Overview
- FastAPI app exposes `POST /api/v1/hackrx/run`.
- Ingestion: downloads or reads local files, supports PDFs, DOCX, EML.
- Chunking: LangChain loaders (PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader) + RecursiveCharacterTextSplitter with `add_start_index` to preserve offsets.
- Embeddings: Gemini `text-embedding-004`.
- Vector DB: FAISS IndexFlatIP over L2-normalized vectors (cosine similarity); metadata stored alongside records.
- Retriever: top-K semantic search with insurance-specific boosts, optional MMR diversity, and lexical augmentation.
- Matcher: sentence-level clause selection with rule-based scoring and query keyword boosts.
- Reasoner: Gemini LLM for final structured answer; falls back to local deterministic composer when no API key.

## Prerequisites
- Python 3.10–3.11 recommended.
- Windows support verified.

## Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration (.env)
The app loads configuration from `.env` automatically at startup.

Key variables:
- TEAM_TOKEN=dev-token
- GOOGLE_API_KEY=your_gemini_api_key  # or GEMINI_API_KEY
- EMBEDDING_MODEL=text-embedding-004
- EMBED_CACHE_TTL=3600                # seconds
- LLM_MODEL=gemini-1.5-flash
- CHUNK_TOKENS=800
- CHUNK_OVERLAP=150
- TOP_K=12
- MAX_CHUNKS_TO_LLM=6
- USE_SUMMARIZER=true
- CROSS_ENCODER_RERANK=false
- DOWNLOAD_MAX_MB=25
- INITIAL_CANDIDATES_MIN=80           # widen early candidate pool for recall
- INITIAL_CANDIDATES_MULT=8
- FILTERED_KEEP=50                    # keep after early filters
- MMR_LAMBDA=0.5                      # diversity vs relevance
- ENABLE_LEXICAL_AUGMENT=true         # add exact-match hits alongside vectors
- LEXICAL_MAX_ADD=10

See `.env` in the repository root for examples.

## Run the API
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Open http://localhost:8000/docs for OpenAPI UI.

## Test with Postman
- Method: POST
- URL: `http://localhost:8000/api/v1/hackrx/run`
- Headers:
  - `Authorization: Bearer dev-token`
  - `Content-Type: application/json`
- Body (raw JSON):
```json
{
  "documents": [
    "file:///C:/Users/NAME/Desktop/api/sample_docs/policy.pdf",
    "file:///C:/Users/NAME/Desktop/api/sample_docs/benefits.docx"
  ],
  "questions": [
    "What is the waiting period for pre-existing diseases and any exclusions?",
    "Is maternity covered and what is the room rent limit?"
  ]
}
```
Tip: Use `file:///C:/...` and URL-encode spaces as `%20`.

## Response Format
- The endpoint returns a simplified response:
- `{"answers": ["string", ...]}` matching the order of input questions.

## Token & Cost Efficiency
- Heavy lifting via embeddings/FAISS, LLM only for final synthesis.
- LangChain splitter uses chunk size ~800 tokens (approx by characters) with overlap to improve recall.
- Matcher trims to clause-level sentences before LLM to reduce tokens.

## Retrieval Accuracy Tuning
- Increase `TOP_K` to fetch more candidates (higher recall, more compute).
- Raise `INITIAL_CANDIDATES_MIN/MULT` if documents are long and varied.
- Lower `MMR_LAMBDA` to favor relevance; raise to diversify.
- Keep `ENABLE_LEXICAL_AUGMENT=true` for exact-match boosts on critical clauses.

## Security
- Bearer token auth via `TEAM_TOKEN`
- URL sanitization and download size limits

## Files
- `app/` contains modules: ingestion, chunking, embeddings, vectordb, retriever, matcher, reasoner, models, utils, main
- `tests/` includes unit/integration tests
- `design.md` summarizes architecture and cost considerations

