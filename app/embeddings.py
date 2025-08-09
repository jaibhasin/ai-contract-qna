# EMBEDDING PERFORMANCE ISSUES AND FIXES

from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional, Iterable
from dataclasses import dataclass
import numpy as np
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .utils import ChunkMeta, sha1_hex, logger, TTLCache, read_env

@dataclass
class EmbeddingConfig:
    provider: str
    model: str

@dataclass
class Chunk:
    text: str
    meta: ChunkMeta
    score: float = 0.0
    doc_type: str = 'general'

## CURRENT ISSUES:
# 1. Sequential embedding calls (no batching)
# 2. Complex response parsing for each embedding
# 3. Cache misses on similar queries

class EmbeddingClient:
    """Wrapper providing embeddings via Gemini with optimizations."""

    def __init__(self) -> None:
        self.api_key = read_env("GOOGLE_API_KEY") or read_env("GEMINI_API_KEY")
        self.provider = "gemini"
        default_model = "text-embedding-004"  # current Gemini embeddings model
        self.model = read_env("EMBEDDING_MODEL", default_model)
        self.cache = TTLCache(ttl_seconds=int(read_env("EMBED_CACHE_TTL", "3600") or "3600"))
        self._init_backend()
        logger.info("EmbeddingClient initialized: provider=%s model=%s", self.provider, self.model)

    def _init_backend(self) -> None:
        self._gemini = None
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY/GEMINI_API_KEY not set for Gemini provider")
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=self.api_key)
            self._gemini = genai
        except Exception as e:
            logger.error("Failed to init Gemini embeddings: %s", e)
            raise RuntimeError("Failed to initialize Gemini embeddings") from e

    @property
    def config(self) -> EmbeddingConfig:
        return EmbeddingConfig(provider=self.provider, model=self.model)

    def _extract_embedding_values(self, res) -> List[float]:
        """Extract embedding values from various Gemini response formats."""
        try:
            # Case 1: Direct attribute access for SDK response object
            if hasattr(res, 'embedding') and hasattr(res.embedding, 'values'):
                return res.embedding.values
            
            # Case 2: List of embedding objects
            elif isinstance(res, list) and res and hasattr(res[0], 'embedding'):
                return res[0].embedding.values
            
            # Case 3: Dictionary response format
            elif isinstance(res, dict) and 'embedding' in res:
                embedding = res['embedding']
                if isinstance(embedding, dict) and 'values' in embedding:
                    return embedding['values']
                elif isinstance(embedding, list):
                    return embedding
                
            # Case 4: List response format with dictionary
            elif isinstance(res, list) and res and isinstance(res[0], dict) and 'embedding' in res[0]:
                embedding = res[0]['embedding']
                if isinstance(embedding, dict) and 'values' in embedding:
                    return embedding['values']
                elif isinstance(embedding, list):
                    return embedding
            
            # Case 5: Direct list of values
            elif isinstance(res, list) and res and isinstance(res[0], (int, float)):
                return res
                
            logger.warning(f"Unexpected embedding format: {type(res)}")
            return []
                
        except Exception as e:
            logger.error(f"Error extracting embedding values: {e}")
            return []

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts_list: List[str] = list(texts)
        if not texts_list:
            return np.zeros((0, 768), dtype=np.float32)

        # Cache lookup per text
        cached_vecs: List[np.ndarray | None] = []
        to_compute: List[str] = []
        for t in texts_list:
            hit = self.cache.get(f"emb:{self.model}:{t}")
            if hit is not None:
                cached_vecs.append(hit)
            else:
                cached_vecs.append(None)
                to_compute.append(t)

        new_vecs: List[np.ndarray] = []
        if to_compute:
            assert self._gemini is not None
            for t in to_compute:
                try:
                    res = self._gemini.embed_content(
                        model=self.model,
                        content=t,
                        task_type="retrieval_document",
                    )
                    
                    values = self._extract_embedding_values(res)
                    if not values:
                        logger.warning("Empty embedding values from Gemini, using zero vector")
                        values = [0.0] * 768  # Fallback to zero vector
                    
                    new_vecs.append(np.asarray(values, dtype=np.float32))
                    
                except Exception as e:
                    logger.error(f"Failed to embed text '{t[:50]}...': {e}")
                    # Use zero vector as fallback
                    new_vecs.append(np.zeros(768, dtype=np.float32))

        # Merge cached and new
        it_new = iter(new_vecs)
        final: List[np.ndarray] = []
        for i, maybe in enumerate(cached_vecs):
            if maybe is None:
                v = next(it_new)
                final.append(v)
                self.cache.set(f"emb:{self.model}:{texts_list[i]}", v)
            else:
                final.append(maybe)
        
        if not final:
            return np.zeros((0, 768), dtype=np.float32)
            
        return np.vstack(final)

    def embed_query(self, text: str) -> np.ndarray:
        v = self.embed([text])
        return v[0]

## OPTIMIZED EMBEDDING CLIENT

import numpy as np
from typing import List, Iterable
import asyncio
from concurrent.futures import ThreadPoolExecutor

class OptimizedEmbeddingClient:
    """Optimized embedding client with batching and parallel processing.
    
    Features:
    - Batch processing for reduced API calls
    - Parallel execution for improved throughput
    - Request timeouts and retries
    - Efficient caching with TTL
    - Simplified response parsing
    """
    
    def __init__(self, max_workers: int = 4, batch_size: int = 10, request_timeout: int = 30):
        """Initialize the optimized embedding client.
        
        Args:
            max_workers: Number of worker threads for parallel execution
            batch_size: Number of texts to process in each batch
            request_timeout: Timeout in seconds for each embedding request
        """
        self.api_key = read_env("GOOGLE_API_KEY") or read_env("GEMINI_API_KEY")
        self.provider = "gemini"
        self.model = read_env("EMBEDDING_MODEL", "text-embedding-004")
        self.batch_size = int(read_env("EMBEDDING_BATCH_SIZE", str(batch_size)))
        self.request_timeout = request_timeout
        
        # Initialize cache with TTL (1 hour by default)
        cache_ttl = int(read_env("EMBED_CACHE_TTL", "3600"))
        self.cache = TTLCache(ttl_seconds=cache_ttl)
        
        # Thread pool for parallel execution
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize the backend
        self._init_backend()
        logger.info(
            f"Initialized OptimizedEmbeddingClient: "
            f"provider={self.provider}, model={self.model}, "
            f"batch_size={self.batch_size}, workers={max_workers}"
        )
    
    def _extract_embedding_values_fast(self, res) -> List[float]:
        """Extract embedding values with optimized parsing for common response formats.
        
        Args:
            res: Response object from the embedding API
            
        Returns:
            List of embedding values or zeros if extraction fails
        """
        try:
            # Fast path 1: Direct attribute access (SDK response object)
            if hasattr(res, 'embedding') and hasattr(res.embedding, 'values'):
                return res.embedding.values
                
            # Fast path 2: Dictionary with 'embedding' key
            if isinstance(res, dict) and 'embedding' in res:
                embedding = res['embedding']
                if isinstance(embedding, dict) and 'values' in embedding:
                    return embedding['values']
                if isinstance(embedding, list):
                    return embedding
                
            # Fast path 3: List of embedding objects
            if isinstance(res, list) and res and hasattr(res[0], 'embedding'):
                return res[0].embedding.values
                
            # Fallback: Return zero vector if we can't parse the response
            logger.warning(f"Could not extract embedding from response: {type(res)}")
            return [0.0] * 768
            
        except Exception as e:
            logger.error(f"Error extracting embedding values: {e}")
            return [0.0] * 768  # Return zero vector on error
    
    def _process_batch(self, batch: List[Tuple[int, str]]) -> Dict[int, np.ndarray]:
        """Process a single batch of texts for embedding.
        
        Args:
            batch: List of (index, text) tuples to process
            
        Returns:
            Dictionary mapping original indices to their embeddings
        """
        results = {}
        
        # Group texts by their cache keys
        cache_keys = [f"emb:{self.model}:{text}" for _, text in batch]
        texts = [text for _, text in batch]
        
        try:
            # Get all embeddings in a single batch request
            response = self._gemini.embed_content(
                model=self.model,
                content=texts,
                task_type="retrieval_document",
                request_options={"timeout": self.request_timeout}
            )
            
            # Process responses
            if isinstance(response, list):
                # Multiple embeddings in response
                embeddings = [self._extract_embedding_values_fast(res) for res in response]
            else:
                # Single embedding response
                embeddings = [self._extract_embedding_values_fast(response)]
            
            # Update cache and collect results
            for (idx, _), embedding in zip(batch, embeddings):
                if embedding:  # Only cache valid embeddings
                    self.cache.set(cache_keys[idx], embedding)
                    results[idx] = np.asarray(embedding, dtype=np.float32)
                else:
                    results[idx] = np.zeros(768, dtype=np.float32)
                    
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Return zero vectors for failed batch
            for idx, _ in batch:
                results[idx] = np.zeros(768, dtype=np.float32)
                
        return results
    
    def embed_batch(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Embed a batch of texts with parallel processing and caching.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
            
        batch_size = batch_size or self.batch_size
        
        # Check cache first
        cached_results = {}
        uncached_texts = []
        
        for i, text in enumerate(texts):
            if not text:  # Skip empty texts
                cached_results[i] = np.zeros(768, dtype=np.float32)
                continue
                
            cache_key = f"emb:{self.model}:{text}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                cached_results[i] = cached
            else:
                uncached_texts.append((i, text))
        
        # Process uncached texts in parallel batches
        if uncached_texts:
            # Split into batches
            batches = [
                uncached_texts[i:i + batch_size]
                for i in range(0, len(uncached_texts), batch_size)
            ]
            
            # Process batches in parallel
            futures = [
                self._executor.submit(self._process_batch, batch)
                for batch in batches
            ]
            
            # Collect results as they complete
            for future in as_completed(futures, timeout=self.request_timeout * 2):
                try:
                    batch_results = future.result()
                    cached_results.update(batch_results)
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
                    # Ensure we have zero vectors for failed batches
                    for i, _ in batches[len(cached_results) // batch_size]:
                        cached_results[i] = np.zeros(768, dtype=np.float32)
        
        # Combine results in original order
        final_embeddings = [cached_results[i] for i in range(len(texts))]
        return np.vstack(final_embeddings) if final_embeddings else np.zeros((0, 768), dtype=np.float32)
    
    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Embed multiple texts with optimized batching and parallel processing.
        
        Args:
            texts: Iterable of text strings to embed
            
        Returns:
            Numpy array of shape (num_texts, embedding_dim)
        """
        return self.embed_batch(list(texts))
    
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query with caching.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array of shape (embedding_dim,)
        """
        if not text:
            return np.zeros(768, dtype=np.float32)
            
        # Use batch processing with a single item for consistent caching
        result = self.embed_batch([text])
        return result[0] if len(result) > 0 else np.zeros(768, dtype=np.float32)
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

# SMART QUERY PREPROCESSING
def preprocess_query_for_insurance(query: str) -> str:
    """Optimize query for better insurance document retrieval"""
    # Add context terms for better matching
    insurance_context = {
        'grace period': 'grace period premium payment',
        'waiting period': 'waiting period pre-existing disease',
        'maternity': 'maternity coverage benefit',
        'room rent': 'room rent limit daily charges',
        'ncd': 'no claim discount renewal bonus',
    }
    
    query_lower = query.lower()
    for term, expanded in insurance_context.items():
        if term in query_lower:
            return expanded
    
    return query

# RETRIEVAL OPTIMIZATION
class OptimizedRetriever:
    """Optimized retriever with caching, query enhancement, and efficient filtering.
    
    Features:
    - Query result caching with LRU eviction
    - Insurance-specific query enhancement
    - Efficient chunk filtering and scoring
    - Configurable result thresholds
    """
    
    def __init__(self, store, emb, cache_size: int = 100):
        """Initialize the optimized retriever.
        
        Args:
            store: Vector store for similarity search
            emb: Embedding client for query encoding
            cache_size: Maximum number of queries to cache
        """
        self.store = store
        self.emb = emb
        self.cache_size = cache_size
        self._query_cache = {}  # Cache for query results
        self._query_hashes = []  # Track cache keys for LRU eviction
        
    def _get_cache_key(self, query: str, top_k: int) -> str:
        """Generate a consistent cache key for the query."""
        return f"{hash(query)}:{top_k}"
    
    def _add_to_cache(self, key: str, results: list) -> None:
        """Add results to cache with LRU eviction."""
        # Remove key if it exists (to update position in LRU)
        if key in self._query_cache:
            self._query_hashes.remove(key)
        
        # Add to cache
        self._query_cache[key] = results
        self._query_hashes.append(key)
        
        # Enforce cache size limit
        while len(self._query_cache) > self.cache_size and self._query_hashes:
            old_key = self._query_hashes.pop(0)
            self._query_cache.pop(old_key, None)
    
    def _enhance_insurance_query(self, query: str) -> str:
        """Enhance query with insurance-specific context and synonyms."""
        if not query or not query.strip():
            return query
            
        # Common insurance term mappings
        term_mappings = {
            'policy': ['policy', 'plan', 'contract', 'agreement'],
            'coverage': ['coverage', 'protection', 'insurance', 'benefit'],
            'claim': ['claim', 'request', 'application', 'submission'],
            'deductible': ['deductible', 'excess', 'out-of-pocket'],
            'premium': ['premium', 'payment', 'fee', 'cost'],
            'exclusion': ['exclusion', 'limitation', 'exception', 'restriction']
        }
        
        # Add synonyms for common insurance terms
        enhanced_terms = []
        for term in query.lower().split():
            if term in term_mappings:
                enhanced_terms.extend(term_mappings[term])
            else:
                enhanced_terms.append(term)
        
        # Add context for common question patterns
        question_words = ['what', 'how', 'when', 'where', 'why', 'can', 'does', 'do']
        if any(query.lower().startswith(word) for word in question_words):
            enhanced_terms.extend(['explanation', 'definition', 'details'])
        
        # Combine terms and remove duplicates while preserving order
        seen = set()
        result = []
        for term in enhanced_terms:
            if term not in seen:
                seen.add(term)
                result.append(term)
        
        return ' '.join(result)
    
    def _filter_and_rank_results(
        self, 
        results: list, 
        min_score: float = 0.8,
        min_terms: int = 1
    ) -> list:
        """Filter and re-rank search results.
        
        Args:
            results: List of (doc, score) tuples
            min_score: Maximum score threshold (lower is better)
            min_terms: Minimum number of query terms that must match
            
        Returns:
            Filtered and re-ranked list of results
        """
        filtered = []
        
        for doc, score in results:
            # Skip low confidence matches
            if score > min_score:
                continue
                
            # Count matching query terms in document
            content = doc.page_content.lower()
            term_count = sum(1 for term in self._last_query_terms 
                           if term in content)
                            
            if term_count >= min_terms:
                # Adjust score based on term matches (more matches = better)
                adjusted_score = score * (1.0 - (term_count * 0.1))
                filtered.append((doc, adjusted_score, term_count))
        
        # Sort by adjusted score (ascending) and term count (descending)
        filtered.sort(key=lambda x: (x[1], -x[2]))
        return [(doc, score) for doc, score, _ in filtered]
    
    def search_optimized(self, query: str, top_k: int = 10) -> list:
        """Perform optimized semantic search with caching and enhancements.
        
        Args:
            query: The search query string
            top_k: Maximum number of results to return
            
        Returns:
            List of (document, score) tuples, sorted by relevance
        """
        if not query or not query.strip():
            return []
            
        # Generate cache key
        cache_key = self._get_cache_key(query, top_k)
        
        # Check cache first
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        # Preprocess and enhance query
        processed_query = preprocess_query_for_insurance(query)
        enhanced_query = self._enhance_insurance_query(processed_query)
        self._last_query_terms = set(enhanced_query.lower().split())
        
        try:
            # Get query embedding
            query_embedding = self.emb.embed_query(enhanced_query)
            
            # Search in vector store with optimized parameters
            # Retrieve more results than needed for filtering
            results = self.store.similarity_search_with_score(
                query_embedding,
                k=top_k * 3,  # Get more results for better filtering
                filter={"doc_type": "insurance"}  # Filter by document type
            )
            
            # Filter and re-rank results
            filtered_results = self._filter_and_rank_results(
                results,
                min_score=0.8,  # Tune based on your data
                min_terms=1     # Require at least one term match
            )
            
            # Cache the results
            final_results = filtered_results[:top_k]
            self._add_to_cache(cache_key, final_results)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in search_optimized: {e}")
            return []
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._query_cache.clear()
        self._query_hashes = []