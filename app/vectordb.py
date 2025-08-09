from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .chunker import Chunk
from .embeddings import EmbeddingClient
from .utils import logger, read_env


@dataclass
class VectorRecord:
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any]


class VectorStore:
    """FAISS-based vector store with cosine similarity and insurance-specific enhancements.

    This implementation uses faiss IndexFlatIP over L2-normalized vectors to
    approximate cosine similarity. Metadata is stored separately and joined
    on result indices.
    
    Enhanced features:
    - Insurance-specific search with term boosting
    - Better retrieval strategies for policy documents
    - Context-aware result filtering and ranking
    """

    def __init__(self) -> None:
        # FAISS components
        self._faiss_index = None  # lazy init after first upsert
        self._dim: Optional[int] = None
        self._records: List[VectorRecord] = []
        
        # Insurance-specific configuration
        self._insurance_terms = {
            'high_priority': [
                'grace period', 'waiting period', 'pre-existing', 'maternity',
                'exclusion', 'coverage', 'sum insured', 'premium', 'claim'
            ],
            'medium_priority': [
                'deductible', 'copay', 'network', 'benefit', 'rider',
                'policy holder', 'beneficiary', 'ayush', 'room rent'
            ],
            'time_indicators': [
                'days', 'months', 'years', 'period', 'duration'
            ]
        }
        
        logger.info("VectorStore using FAISS (cosine via inner product on normalized vectors)")
        logger.info("Enhanced with insurance-specific search capabilities")

    def upsert_documents(self, chunks: List[Chunk], emb: EmbeddingClient) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vecs = emb.embed(texts)
        # Normalize vectors for cosine similarity
        vecs = vecs.astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        vecs_norm = vecs / norms

        # Lazy init FAISS index
        if self._faiss_index is None:
            try:
                import faiss  # type: ignore

                self._dim = int(vecs_norm.shape[1])
                self._faiss_index = faiss.IndexFlatIP(self._dim)
                logger.info(f"FAISS index initialized with dimension: {self._dim}")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize FAISS: {e}")

        # Add to index and record metadata
        try:
            import faiss  # type: ignore

            self._faiss_index.add(vecs_norm)  # type: ignore[union-attr]
        except Exception as e:
            raise RuntimeError(f"FAISS add failed: {e}")

        for c, v in zip(chunks, vecs_norm):
            # Extract additional metadata for better search
            text_lower = c.text.lower()
            
            # Detect insurance-specific features
            has_numbers = bool(re.search(r'\d+(?:\.\d+)?', c.text))
            has_time_periods = bool(re.search(r'\d+\s*(days?|months?|years?)', text_lower))
            has_percentages = bool(re.search(r'\d+(?:\.\d+)?%', c.text))
            
            # Count high-priority insurance terms
            priority_term_count = sum(1 for term in self._insurance_terms['high_priority'] 
                                    if term in text_lower)
            
            # Enhanced metadata
            enhanced_metadata = {
                "doc_url": c.meta.doc_url,
                "page": c.meta.page,
                "chunk_id": c.meta.chunk_id,
                "start": c.meta.start,
                "end": c.meta.end,
                "text": c.text,
                # Insurance-specific metadata
                "has_numbers": has_numbers,
                "has_time_periods": has_time_periods,
                "has_percentages": has_percentages,
                "priority_term_count": priority_term_count,
                "text_length": len(c.text),
                "doc_type": getattr(c, 'doc_type', 'general')  # From enhanced chunker
            }
            
            self._records.append(
                VectorRecord(
                    id=c.meta.chunk_id,
                    vector=v,
                    metadata=enhanced_metadata,
                )
            )

        logger.info(f"Added {len(chunks)} chunks to vector store. Total records: {len(self._records)}")

    def query(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[VectorRecord, float]]:
        """Legacy query method - redirects to enhanced search."""
        return self.search(query_vec, k=top_k)

    def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> List[Tuple[VectorRecord, float]]:
        """Basic search for similar vectors.

        Args:
            query_embedding: Query vector (assumed to be normalized if needed).
            k: Number of results to return.

        Returns:
            List of (record, score) tuples, sorted by score (descending).
        """
        if not self._records or self._faiss_index is None:
            return []

        # Ensure query is 2D (1, dim) and normalized
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        query = query / (np.linalg.norm(query) + 1e-8)

        # FAISS search returns (distances, indices)
        distances, indices = self._faiss_index.search(query, k=min(k, len(self._records)))
        
        # Convert to (record, score) tuples, handling empty results
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self._records):
                results.append((self._records[idx], float(score)))
        
        return sorted(results, key=lambda x: x[1], reverse=True)

    def search_insurance_context(
        self, 
        query_embedding: np.ndarray, 
        query_text: str,
        k: int = 12,  # Increased default for better coverage
        min_score: float = 0.25,
        boost_factors: Optional[Dict[str, float]] = None
    ) -> List[Tuple[VectorRecord, float]]:
        """Enhanced search optimized for insurance document queries.
        
        Args:
            query_embedding: Query vector
            query_text: Original query text for context analysis
            k: Number of results to return
            min_score: Minimum similarity score threshold
            boost_factors: Custom boost factors for different features
            
        Returns:
            List of (record, score) tuples with enhanced ranking
        """
        if not self._records or self._faiss_index is None:
            logger.warning("No records or index available for search")
            return []

        # Default boost factors
        if boost_factors is None:
            boost_factors = {
                'high_priority_terms': 0.15,
                'time_periods': 0.20,
                'numbers': 0.10,
                'percentages': 0.15,
                'exact_match': 0.25,
                'doc_type_insurance': 0.10
            }

        # Get initial candidates (retrieve more for better filtering)
        initial_k = min(k * 3, len(self._records))
        initial_results = self.search(query_embedding, k=initial_k)
        
        if not initial_results:
            return []
        
        # Apply minimum score filtering
        filtered_results = [
            (record, score) for record, score in initial_results 
            if score >= min_score
        ]
        
        if not filtered_results:
            logger.warning(f"No results above minimum score {min_score}")
            # Lower the threshold and try again
            filtered_results = [
                (record, score) for record, score in initial_results 
                if score >= min_score * 0.7
            ][:k*2]  # Take top candidates even if below threshold

        # Enhanced scoring with insurance-specific boosting
        enhanced_results = []
        query_lower = query_text.lower()
        query_terms = set(query_lower.split())
        
        for record, base_score in filtered_results:
            text_lower = record.metadata['text'].lower()
            enhanced_score = base_score
            boost_reasons = []
            
            # 1. High-priority term matching
            matching_priority_terms = [
                term for term in self._insurance_terms['high_priority'] 
                if term in text_lower and any(word in query_lower for word in term.split())
            ]
            if matching_priority_terms:
                boost = boost_factors['high_priority_terms'] * len(matching_priority_terms)
                enhanced_score += boost
                boost_reasons.append(f"priority_terms({len(matching_priority_terms)})")
            
            # 2. Time period relevance (crucial for insurance)
            if record.metadata.get('has_time_periods', False) and any(
                term in query_lower for term in ['period', 'time', 'days', 'months', 'years', 'when', 'how long']
            ):
                enhanced_score += boost_factors['time_periods']
                boost_reasons.append("time_periods")
            
            # 3. Numerical information (amounts, percentages, counts)
            if 'amount' in query_lower or 'much' in query_lower or 'limit' in query_lower:
                if record.metadata.get('has_numbers', False):
                    enhanced_score += boost_factors['numbers']
                    boost_reasons.append("numbers")
                if record.metadata.get('has_percentages', False):
                    enhanced_score += boost_factors['percentages']
                    boost_reasons.append("percentages")
            
            # 4. Exact phrase matching
            key_phrases = [
                'grace period', 'waiting period', 'pre-existing', 'maternity',
                'room rent', 'sum insured', 'no claim', 'ayush'
            ]
            for phrase in key_phrases:
                if phrase in query_lower and phrase in text_lower:
                    enhanced_score += boost_factors['exact_match']
                    boost_reasons.append(f"exact_match({phrase})")
                    break  # Only boost once for exact matches
            
            # 5. Document type preference
            if record.metadata.get('doc_type') == 'insurance':
                enhanced_score += boost_factors['doc_type_insurance']
                boost_reasons.append("insurance_doc")
            
            # 6. Content quality indicators
            priority_count = record.metadata.get('priority_term_count', 0)
            if priority_count > 2:  # Rich insurance content
                enhanced_score += 0.05
                boost_reasons.append("rich_content")
            
            # Ensure score doesn't exceed 1.0
            enhanced_score = min(1.0, enhanced_score)
            
            # Log boosting for debugging (only for top results)
            if len(enhanced_results) < 5 and boost_reasons:
                logger.debug(f"Boosted chunk {record.id}: {base_score:.3f} -> {enhanced_score:.3f} ({', '.join(boost_reasons)})")
            
            enhanced_results.append((record, enhanced_score))
        
        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply diversity filtering to avoid too many chunks from the same page
        diverse_results = self._apply_diversity_filter(enhanced_results, k)
        
        logger.info(f"Insurance search: {len(initial_results)} initial -> {len(filtered_results)} filtered -> {len(diverse_results)} final")
        return diverse_results

    def _apply_diversity_filter(
        self, 
        results: List[Tuple[VectorRecord, float]], 
        target_k: int
    ) -> List[Tuple[VectorRecord, float]]:
        """Apply diversity filtering to ensure results come from different parts of documents."""
        if len(results) <= target_k:
            return results
        
        # Group by document and page
        page_groups = {}
        for record, score in results:
            page_key = f"{record.metadata['doc_url']}|{record.metadata['page']}"
            if page_key not in page_groups:
                page_groups[page_key] = []
            page_groups[page_key].append((record, score))
        
        # Select best chunks from each page group
        final_results = []
        max_per_page = max(1, target_k // max(1, len(page_groups)))
        
        # First pass: take best chunk(s) from each page
        for page_chunks in page_groups.values():
            page_chunks.sort(key=lambda x: x[1], reverse=True)
            final_results.extend(page_chunks[:max_per_page])
        
        # Second pass: fill remaining slots with highest scoring chunks
        final_results.sort(key=lambda x: x[1], reverse=True)
        remaining_slots = target_k - len(final_results)
        
        if remaining_slots > 0:
            # Add more chunks that weren't selected in first pass
            selected_ids = {r[0].id for r in final_results}
            remaining_chunks = [
                (record, score) for record, score in results 
                if record.id not in selected_ids
            ]
            remaining_chunks.sort(key=lambda x: x[1], reverse=True)
            final_results.extend(remaining_chunks[:remaining_slots])
        
        # Final sort and trim
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:target_k]

    def get_chunk_context(
        self, 
        chunk_id: str, 
        context_window: int = 2
    ) -> List[VectorRecord]:
        """Get surrounding chunks for better context (same document/page)."""
        target_record = None
        for record in self._records:
            if record.id == chunk_id:
                target_record = record
                break
        
        if not target_record:
            return []
        
        # Find chunks from same document and nearby positions
        same_doc_chunks = []
        for record in self._records:
            if (record.metadata['doc_url'] == target_record.metadata['doc_url'] and
                record.metadata['page'] == target_record.metadata['page']):
                same_doc_chunks.append(record)
        
        # Sort by position in document
        same_doc_chunks.sort(key=lambda x: x.metadata['start'])
        
        # Find target position and return context window
        try:
            target_idx = same_doc_chunks.index(target_record)
            start_idx = max(0, target_idx - context_window)
            end_idx = min(len(same_doc_chunks), target_idx + context_window + 1)
            return same_doc_chunks[start_idx:end_idx]
        except ValueError:
            return [target_record]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store contents."""
        if not self._records:
            return {"total_records": 0}
        
        # Document type distribution
        doc_types = {}
        pages_per_doc = {}
        insurance_chunks = 0
        
        for record in self._records:
            doc_type = record.metadata.get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            doc_url = record.metadata['doc_url']
            if doc_url not in pages_per_doc:
                pages_per_doc[doc_url] = set()
            pages_per_doc[doc_url].add(record.metadata['page'])
            
            if record.metadata.get('priority_term_count', 0) > 0:
                insurance_chunks += 1
        
        return {
            "total_records": len(self._records),
            "doc_types": doc_types,
            "documents": len(pages_per_doc),
            "total_pages": sum(len(pages) for pages in pages_per_doc.values()),
            "insurance_rich_chunks": insurance_chunks,
            "dimension": self._dim,
            "avg_chunk_length": np.mean([len(r.metadata['text']) for r in self._records]) if self._records else 0
        }

    def clear(self) -> None:
        """Clear all vectors and reset the index."""
        self._faiss_index = None
        self._dim = None
        self._records = []
        logger.debug("VectorStore cleared")

    # Legacy method aliases for backward compatibility
    def search_similar(self, *args, **kwargs):
        """Legacy alias for search method."""
        return self.search(*args, **kwargs)
    
    def find_similar(self, *args, **kwargs):
        """Legacy alias for search method."""
        return self.search(*args, **kwargs)