from __future__ import annotations

import time
import os
from typing import List
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pydantic import BaseModel

from .models import RunRequest, RunResponse, RunMetadata, Answer, Source
from .utils import logger, read_env
from .ingest import download_and_extract
from .chunker import chunk_texts
from .embeddings import EmbeddingClient
from .vectordb import VectorStore
from .retriever import Retriever
from .matcher import ClauseMatcher
from .reasoner import Reasoner

app = FastAPI(title="HackRX Intelligent Query–Retrieval API", version="1.0.0")

# Ensure .env is loaded before reading any config
load_dotenv()

TEAM_TOKEN = read_env("TEAM_TOKEN", "dev-token")

# Config knobs (with sane defaults)
CHUNK_TOKENS = int(read_env("CHUNK_TOKENS", "800"))
CHUNK_OVERLAP = int(read_env("CHUNK_OVERLAP", "150"))
TOP_K = int(read_env("TOP_K", "12"))
MAX_CHUNKS_TO_LLM = int(read_env("MAX_CHUNKS_TO_LLM", "6"))
USE_SUMMARIZER = read_env("USE_SUMMARIZER", "true").lower() == "true"
CROSS_ENCODER_RERANK = read_env("CROSS_ENCODER_RERANK", "false").lower() == "true"

# Singletons
embedding_client = EmbeddingClient()
vector_store = VectorStore()
retriever = Retriever(vector_store, embedding_client)
matcher = ClauseMatcher()
reasoner = Reasoner(embedding_client)


class SimpleResponse(BaseModel):
    answers: List[str]

async def clear_vector_store():
    """Background task to clear the vector store."""
    vector_store.clear()
    logger.debug("Vector store cleared after request")

@app.post("/api/v1/hackrx/run")
async def run_endpoint(
    payload: RunRequest, 
    background_tasks: BackgroundTasks,
    authorization: str = Header(None)
):
    start = time.time()
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    if token != TEAM_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

    # Normalize documents list
    doc_urls: List[str] = payload.documents if isinstance(payload.documents, list) else [payload.documents]

    # Ingest and chunk
    pages_by_url = await download_and_extract(doc_urls)
    chunks = chunk_texts(pages_by_url, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP)

    # Embed and upsert
    vector_store.upsert_documents(chunks, embedding_client)

    answers: List[str] = []
    for q in payload.questions:
        try:
            # Retrieve
            retrieved = retriever.search(q, top_k=TOP_K, rerank=CROSS_ENCODER_RERANK)
            # Match clauses and compute confidence
            clauses = matcher.select_clauses(q, retrieved)
            # Limit number to LLM
            top_for_llm = clauses[:MAX_CHUNKS_TO_LLM]
            # Get the answer
            final = await reasoner.answer(q, top_for_llm, use_summarizer=USE_SUMMARIZER)
            answers.append(final["answer"])
        except Exception as e:
            logger.error(f"Error processing question '{q}': {str(e)}")
            answers.append("I couldn't find an answer to this question in the provided documents.")

    # Transform answers to the required format
    formatted_answers = []
    for answer in answers:
        # If answer is a dict (from reasoner), extract the 'answer' field
        if isinstance(answer, dict) and 'answer' in answer:
            formatted_answers.append(answer['answer'])
        else:
            formatted_answers.append(str(answer))
    
    # Schedule vector store clearing to happen after the response is sent
    background_tasks.add_task(clear_vector_store)
    
    # Return the response in the required format
    return {"answers": formatted_answers}
