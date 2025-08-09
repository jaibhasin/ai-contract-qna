from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .chunker import Chunk
from .embeddings import EmbeddingClient
from .vectordb import VectorStore, VectorRecord


@dataclass
class RetrievedChunk(Chunk):
    """A Chunk augmented with retrieval score.
    We reuse Chunk but carry the score field already present.
    """


class Retriever:
    """Semantic search over VectorStore with optional re-ranking.

    Primary ranking uses vector DB scores. Optional rerank recomputes
    cosine similarity using fresh query embedding vs candidate texts
    to stabilize across providers.
    """

    def __init__(self, store: VectorStore, emb: EmbeddingClient) -> None:
        self.store = store
        self.emb = emb

    def search(self, query: str, top_k: int = 10, rerank: bool = False) -> List[Chunk]:
        qv = self.emb.embed_query(query)
        matches = self.store.query(qv, top_k=top_k)
        chunks: List[Chunk] = []
        for rec, score in matches:
            text = rec.metadata.get("text") or rec.metadata.get("text_excerpt", "")
            # Create a pseudo Chunk for provenance downstream
            from .utils import ChunkMeta

            meta = ChunkMeta(
                doc_url=str(rec.metadata.get("doc_url")),
                page=int(rec.metadata.get("page")) if rec.metadata.get("page") is not None else 1,
                chunk_id=str(rec.metadata.get("chunk_id")) or rec.id,
                start=0,
                end=0,
            )
            ch = Chunk(text=text, meta=meta)
            ch.score = float(score)
            chunks.append(ch)
        if rerank and chunks:
            texts = [c.text for c in chunks]
            X = self.emb.embed(texts)
            q = qv.astype(np.float32)
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            qn = q / (np.linalg.norm(q) + 1e-8)
            sims = (Xn @ qn)
            for c, s in zip(chunks, sims):
                c.score = float(s)
            chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks
