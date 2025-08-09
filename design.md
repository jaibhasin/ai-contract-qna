# Design: LLM-Powered Intelligent Query–Retrieval System

## Architecture (ASCII)

Client --> FastAPI `/api/v1/hackrx/run`
            |
            v
     Ingestion (httpx, PyMuPDF, python-docx, email parser)
            |
            v
      Chunker (LangChain loaders + RecursiveCharacterTextSplitter; page-preserving, overlap, IDs)
            |
            v
     Embeddings (Gemini text-embedding-004 with TTL cache)
            |
            v
 Vector DB (FAISS IndexFlatIP over L2-normalized vectors; cosine via inner product)
            |
            v
      Retriever (semantic search + insurance-specific boosts, lexical augmentation, MMR; optional rerank)
            |
            v
     Matcher (clause extraction + rule boosts)
            |
            v
    Reasoner (Gemini prompt -> JSON; local deterministic fallback)
            |
            v
          Response (answers[] aligned to questions)

## Key Modules
- `ingest.py`: Downloads and extracts text with page numbers from PDFs, DOCX, and emails.
- `chunker.py`: Uses LangChain loaders (PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader) and RecursiveCharacterTextSplitter to split into ~800-token chunks (approx by chars) with 150 overlap. Stable chunk IDs via hash.
- `embeddings.py`: EmbeddingClient using Gemini `text-embedding-004` with TTL cache and robust response parsing.
- `vectordb.py`: FAISS-based vector store (IndexFlatIP) with cosine similarity on normalized vectors. Stores provenance metadata and insurance-specific features (numbers, time periods, percentages, priority term counts). Supports lexical augmentation and MMR diversity.
- `retriever.py`: Queries vector store with insurance-specific boosts and optional re-ranking.
- `matcher.py`: Extracts clause-like sentences and scores them with rule-based boosts.
- `reasoner.py`: Builds deterministic RAG prompt and asks Gemini to output strict JSON. Local fallback composes a deterministic answer when LLM is unavailable.
- `main.py`: FastAPI app and endpoint orchestration, auth, and config knobs.

## Explainability & Provenance
- Each clause retains `doc_url`, `page`, `chunk_id`, and offsets in `ChunkMeta`.
- Response includes `sources` array with provenance and retrieval score per clause.
- Reasoner prompts the model to cite clause numbers in reasoning to trace which evidence was used.

## Token/Latency Considerations
- Retrieval-first design: LLM only used for final explanation/formatting.
- `MAX_CHUNKS_TO_LLM` limits clauses passed to LLM.
- `USE_SUMMARIZER` truncates long clauses to keep prompt compact.
- Embedding TTL cache avoids recomputing vectors for unchanged texts.
- Optional `CROSS_ENCODER_RERANK` enables heavier reranking if desired (defaults off to preserve throughput).
- Defaults prioritize accuracy over speed by widening candidate pools and enabling lexical augmentation.

## Configuration
- All configuration is loaded from `.env` at startup. Primary toggles:
  - TEAM_TOKEN
  - GOOGLE_API_KEY / GEMINI_API_KEY
  - EMBEDDING_MODEL (text-embedding-004), EMBED_CACHE_TTL
  - CHUNK_TOKENS, CHUNK_OVERLAP
  - TOP_K, MAX_CHUNKS_TO_LLM, USE_SUMMARIZER, CROSS_ENCODER_RERANK
  - INITIAL_CANDIDATES_MIN, INITIAL_CANDIDATES_MULT, FILTERED_KEEP
  - MMR_LAMBDA
  - ENABLE_LEXICAL_AUGMENT, LEXICAL_MAX_ADD

## Security
- Bearer token auth using `TEAM_TOKEN`.
- URL sanitization and maximum download size `DOWNLOAD_MAX_MB`.

## Extensibility
- Swap `EmbeddingClient` provider/model via env.
- Replace `VectorStore` implementation to use alternative backends if needed.
- Extend `ClauseMatcher` with domain-specific rules or a cross-encoder.
- Add summarizer step before reasoning for additional token savings.

## Testing Strategy
- Unit tests for ingestion, chunking, embeddings.
- Integration test for endpoint using a generated local PDF via `file://` URL.

## API Contract
- Endpoint: `POST /api/v1/hackrx/run`
- Request body:
  - `documents`: string | string[] (blob URLs or local `file:///` URLs)
  - `questions`: string[]
- Response body:
  - `{ "answers": string[] }` where order matches input `questions`.

## Limitations
- Email parsing via `unstructured` may require platform-specific tuning; chunker gracefully falls back to ingestion texts if loader fails.
- FAISS store in this implementation is in-memory and non-persistent across restarts.
