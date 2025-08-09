# Changelog

All notable changes to this project will be documented in this file.

## 0.2.0 (2025-08-09)
- Switch vector store to FAISS-only (IndexFlatIP with cosine on normalized vectors); removed Pinecone code and dependency.
- Refactor chunker to use LangChain loaders (PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader) and RecursiveCharacterTextSplitter with add_start_index; preserve provenance metadata and stable chunk IDs.
- Make Gemini the first-choice embedding provider via `.env` (EMBEDDING_PROVIDER=gemini); sentence-transformers used as fallback.
- Ensure configuration is loaded strictly from `.env` at startup (load_dotenv in app startup).
- Update README and design docs to reflect current architecture and setup.

## 0.1.0 (2025-08-09)
- Scaffold FastAPI app with `/api/v1/hackrx/run` endpoint
- Implement ingestion for PDF/DOCX/Email with page preservation
- Add chunker with configurable token size and overlap; stable chunk IDs
- Implement embeddings client (Gemini → sentence-transformers → dummy fallback) with TTL cache
- Vector store adapter for Pinecone with in-memory cosine fallback
- Retriever with optional re-ranking; Clause matcher with rule-based boosts
- Reasoner using Gemini with deterministic local fallback producing strict JSON fields
- Tests for ingestion, chunking, embeddings, and endpoint integration
- README and design documentation
