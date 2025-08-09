from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from typing_extensions import Literal

from .chunker import Chunk
from .embeddings import EmbeddingClient
from .query_processor import query_processor
from .vectordb import VectorStore, VectorRecord

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk(Chunk):
    """A Chunk augmented with retrieval score.
    We reuse Chunk but carry the score field already present.
    """


class Retriever:
    """Enhanced semantic search over VectorStore with insurance-specific optimizations.
    
    Features:
    - Insurance-specific query preprocessing and expansion
    - Multi-stage retrieval with fallbacks
    - Hybrid search combining vector similarity and term matching
    - Configurable result processing and filtering
    """
    
    def __init__(
        self, 
        store: VectorStore, 
        emb: EmbeddingClient,
        min_score: float = 0.15,
        fallback_min_score: float = 0.10,
        max_retrieval_attempts: int = 2
    ) -> None:
        self.store = store
        self.emb = emb
        self.min_score = min_score
        self.fallback_min_score = fallback_min_score
        self.max_retrieval_attempts = max_retrieval_attempts
        self._query_cache: Dict[Tuple[str, int, bool], List[Chunk]] = {}
    
    def _create_chunk(self, record: VectorRecord, score: float) -> Chunk:
        """Create a Chunk from a VectorRecord with metadata."""
        text = record.metadata.get("text") or record.metadata.get("text_excerpt", "")
        
        # Create a pseudo Chunk for provenance downstream
        from .utils import ChunkMeta
        
        meta = ChunkMeta(
            doc_url=str(record.metadata.get("doc_url") or ""),
            page=int(record.metadata.get("page", 1)),
            chunk_id=str(record.metadata.get("chunk_id") or record.id),
            start=0,
            end=0,
        )
        
        chunk = Chunk(text=text, meta=meta)
        chunk.score = float(score)
        return chunk
    
    def _process_retrieved_chunks(
        self, 
        records_scores: List[Tuple[VectorRecord, float]],
        query: str,
        min_score: float,
        max_chunks: int
    ) -> List[Chunk]:
        """Process and filter retrieved chunks."""
        chunks: List[Chunk] = []
        
        for record, score in records_scores:
            if score < min_score:
                continue
                
            chunk = self._create_chunk(record, score)
            
            # Additional scoring based on term matches
            self._enhance_chunk_score(chunk, query)
            
            chunks.append(chunk)
        
        # Sort by enhanced score
        chunks.sort(key=lambda c: c.score, reverse=True)
        
        return chunks[:max_chunks]
    
    def _enhance_chunk_score(self, chunk: Chunk, query: str) -> None:
        """Enhance chunk score based on term matching and insurance-specific heuristics."""
        text_lower = chunk.text.lower()
        query_terms = set(term.lower() for term in query.split() if len(term) > 2)
        
        # Count matching terms between query and chunk
        chunk_terms = set(term.lower() for term in text_lower.split() if len(term) > 2)
        matching_terms = query_terms.intersection(chunk_terms)
        
        # Boost score based on term matches (up to 0.2)
        term_boost = min(0.2, 0.05 * len(matching_terms))
        
        # Boost for insurance-specific terms
        insurance_terms = {
            'grace period', 'waiting period', 'coverage', 'premium', 'claim',
            'policy', 'exclusion', 'deductible', 'sum insured', 'pre-existing'
        }
        insurance_term_boost = 0.02 * len(insurance_terms.intersection(chunk_terms))
        
        # Apply boosts
        chunk.score += term_boost + insurance_term_boost
    
    def _retrieve_with_fallback(
        self, 
        query: str, 
        top_k: int, 
        min_score: float,
        use_enhanced_search: bool = True
    ) -> List[Chunk]:
        """Retrieve chunks with fallback strategies."""
        cache_key = (query, top_k, use_enhanced_search)
        if cache_key in self._query_cache:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return self._query_cache[cache_key]
        
        logger.info(f"Retrieving for query: {query}")
        
        try:
            # Get query embedding
            qv = self.emb.embed_query(query)
            
            # Try enhanced search first if available
            if use_enhanced_search and hasattr(self.store, 'search_insurance_context'):
                matches = self.store.search_insurance_context(
                    qv, query, k=top_k*2, min_score=min_score
                )
                logger.info(f"Enhanced search returned {len(matches)} results")
            else:
                # Fall back to basic search
                matches = self.store.query(qv, top_k=top_k*2)
                logger.info(f"Basic search returned {len(matches)} results")
            
            # Process and filter results
            chunks = self._process_retrieved_chunks(matches, query, min_score, top_k)
            
            # Cache the results
            self._query_cache[cache_key] = chunks
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._query_cache.clear()
    
    def search(
        self, 
        query: str, 
        top_k: int = 10, 
        rerank: bool = False,
        use_enhanced_search: bool = True
    ) -> List[Chunk]:
        """Search for relevant chunks with insurance-specific optimizations.
        
        Args:
            query: The search query
            top_k: Maximum number of results to return
            rerank: Whether to re-rank results using fresh embeddings
            use_enhanced_search: Whether to use insurance-specific search
            
        Returns:
            List of relevant Chunks with scores
        """
        if not query.strip():
            return []
        
        # Preprocess and expand the query
        processed_query = query_processor.preprocess_query(query)
        logger.debug(f"Original query: {query}")
        logger.debug(f"Processed query: {processed_query}")
        
        # Try retrieval with progressively relaxed constraints
        attempts = 0
        chunks: List[Chunk] = []
        
        while attempts < self.max_retrieval_attempts and not chunks:
            current_min_score = self.min_score if attempts == 0 else self.fallback_min_score
            
            # Try with processed query first, then fall back to original
            search_query = processed_query if attempts < 2 else query
            
            chunks = self._retrieve_with_fallback(
                search_query,
                top_k=top_k * 2,  # Get more chunks initially
                min_score=current_min_score,
                use_enhanced_search=use_enhanced_search and (attempts < 2)
            )
            
            attempts += 1
            
            if not chunks and attempts < self.max_retrieval_attempts:
                logger.warning(
                    f"No results with min_score={current_min_score:.3f}, "
                    f"trying fallback (attempt {attempts + 1}/{self.max_retrieval_attempts})"
                )
        
        # If we still have no chunks, return empty list
        if not chunks:
            logger.warning(f"No results found for query: {query}")
            return []
        
        # Optionally re-rank with fresh embeddings
        if rerank and len(chunks) > 1:
            try:
                texts = [c.text for c in chunks]
                X = self.emb.embed(texts)
                q = self.emb.embed_query(query).astype(np.float32)
                
                # Normalize vectors
                Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
                qn = q / (np.linalg.norm(q) + 1e-8)
                
                # Compute similarities and update scores
                sims = Xn @ qn
                for c, s in zip(chunks, sims):
                    c.score = float(s)
                
                # Re-sort by new scores
                chunks.sort(key=lambda c: c.score, reverse=True)
                
            except Exception as e:
                logger.error(f"Error during re-ranking: {e}")
        
        # Return top-k chunks
        return chunks[:top_k]
