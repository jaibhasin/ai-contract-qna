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
     Embeddings (Gemini first -> SentenceTransformers fallback)
            |
            v
 Vector DB (FAISS IndexFlatIP over L2-normalized vectors; cosine via inner product)
            |
            v
      Retriever (semantic search, optional rerank)
            |
            v
    Matcher (clause extraction + rule boosts)
            |
            v
    Reasoner (Gemini prompt -> JSON; local deterministic fallback)
            |
            v
          Response (answers + sources + metadata)

## Key Modules
- `ingest.py`: Downloads and extracts text with page numbers from PDFs, DOCX, and emails.
- `chunker.py`: Uses LangChain loaders (PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader) and RecursiveCharacterTextSplitter to split into ~800-token chunks (approx by chars) with 150 overlap. Stable chunk IDs via hash.
- `embeddings.py`: EmbeddingClient with TTL cache. Providers: Gemini (first choice) and sentence-transformers fallback.
- `vectordb.py`: FAISS-based vector store (IndexFlatIP) with cosine similarity on normalized vectors. Stores provenance metadata with each record.
- `retriever.py`: Queries vector store, optional re-ranking using fresh embeddings.
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
- Optional `CROSS_ENCODER_RERANK` flag exists to enable heavier reranking if desired (currently embedding-based rerank to keep light).

## Configuration
- All configuration is loaded from `.env` at startup. Primary toggles:
  - CHUNK_TOKENS, CHUNK_OVERLAP
  - TOP_K, MAX_CHUNKS_TO_LLM, USE_SUMMARIZER, CROSS_ENCODER_RERANK
  - EMBEDDING_PROVIDER/EMBEDDING_MODEL (Gemini or sentence-transformers)
  - GOOGLE_API_KEY (or GEMINI_API_KEY) for Gemini

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

## Limitations
- Email parsing via `unstructured` may require platform-specific tuning; chunker gracefully falls back to ingestion texts if loader fails.
- FAISS store in this implementation is in-memory and non-persistent across restarts.
