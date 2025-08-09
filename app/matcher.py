from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Set, Tuple

from .chunker import Chunk
from .utils import logger


@dataclass
class Clause:
    text: str
    score: float
    source: Chunk


class ClauseMatcher:
    """Enhanced clause matching with insurance-specific optimizations.
    
    Maintains chunk context and applies insurance-specific scoring heuristics
    to improve answer quality for insurance policy questions.
    """
    
    # Pre-compiled regex patterns for insurance-specific terms
    _patterns: Dict[str, Pattern] = {
        'date': re.compile(
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b",
            re.IGNORECASE
        ),
        'exclusion': re.compile(
            r"\b(exclusion|excluded|does not cover|not covered|not included|not applicable|not eligible)\b", 
            re.IGNORECASE
        ),
        'waiting_period': re.compile(
            r"\b(waiting period|waiting time|waiting requirement|after\s+\d+\s+(?:days?|months?|years?)|effective after)\b", 
            re.IGNORECASE
        ),
        'amount': re.compile(
            r"\b(?:Rs\.?\s*)?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\s*(?:rs|inr|rupees?)?\b|\b(?:up to|maximum|max|limit|sum of)\s+(?:Rs\.?\s*)?\d+",
            re.IGNORECASE
        ),
        'coverage': re.compile(
            r"\b(cover|coverage|includes?|including|applicable|eligible|provided|offered|available)\b", 
            re.IGNORECASE
        ),
        'pre_existing': re.compile(
            r"\b(pre[- ]?existing|pre[- ]?existing condition|pre[- ]?existing disease|pre[- ]?existing illness|prior condition)\b",
            re.IGNORECASE
        ),
        'grace_period': re.compile(
            r"\b(grace period|grace days|grace window|late payment window|renewal grace|after due date)\b",
            re.IGNORECASE
        ),
        'claim': re.compile(
            r"\b(claim process|claim procedure|how to claim|claim submission|claim form|claim intimation|claim settlement)\b",
            re.IGNORECASE
        )
    }
    
    # Insurance-specific terms that should boost relevance
    _insurance_terms: Set[str] = {
        'policy', 'premium', 'coverage', 'claim', 'deductible', 'beneficiary',
        'exclusion', 'rider', 'endorsement', 'sum insured', 'pre-existing',
        'waiting period', 'grace period', 'room rent', 'co-payment', 'no claim bonus'
    }
    
    # Question words that indicate the type of information being asked for
    _question_indicators: Dict[str, List[str]] = {
        'time_period': ['how long', 'when', 'what is the duration', 'period', 'timeframe'],
        'amount': ['how much', 'what is the cost', 'what is the amount', 'limit', 'maximum'],
        'coverage': ['is covered', 'does cover', 'what is covered', 'coverage for'],
        'process': ['how to', 'what is the process', 'steps to', 'procedure for']
    }
    
    def __init__(self, min_chunk_length: int = 50, max_chunks: int = 8):
        """Initialize the ClauseMatcher.
        
        Args:
            min_chunk_length: Minimum character length for a chunk to be considered
            max_chunks: Maximum number of chunks to return
        """
        self.min_chunk_length = min_chunk_length
        self.max_chunks = max_chunks
    
    def _detect_question_type(self, question: str) -> List[str]:
        """Detect the type of question to guide scoring."""
        question_lower = question.lower()
        q_types = []
        
        for q_type, indicators in self._question_indicators.items():
            if any(ind in question_lower for ind in indicators):
                q_types.append(q_type)
        
        return q_types or ['general']
    
    def _score_chunk(self, chunk: Chunk, question: str, q_types: List[str]) -> float:
        """Score a chunk based on relevance to the question."""
        if not chunk.text.strip():
            return 0.0
            
        text_lower = chunk.text.lower()
        question_lower = question.lower()
        
        # Start with the base retrieval score
        score = chunk.score
        
        # Check for question terms in the chunk
        q_terms = set(term for term in question_lower.split() if len(term) > 3)
        chunk_terms = set(text_lower.split())
        matching_terms = q_terms.intersection(chunk_terms)
        
        # Boost for matching terms
        score += min(0.3, 0.05 * len(matching_terms))
        
        # Boost for insurance terms
        insurance_term_matches = self._insurance_terms.intersection(chunk_terms)
        score += min(0.2, 0.02 * len(insurance_term_matches))
        
        # Apply pattern-based boosts
        for pattern_name, pattern in self._patterns.items():
            if pattern.search(text_lower):
                # Higher boost for patterns that match the question type
                if (q_types and 
                    ((q_types[0] == 'time_period' and 'period' in pattern_name) or
                     (q_types[0] == 'amount' and 'amount' in pattern_name) or
                     (q_types[0] == 'coverage' and 'coverage' in pattern_name) or
                     (q_types[0] == 'process' and 'claim' in pattern_name))):
                    score += 0.15
                else:
                    score += 0.05
        
        # Penalize very short chunks
        if len(text_lower) < 30:
            score *= 0.7
            
        return max(0.0, min(1.0, score))  # Keep score in [0, 1] range
    
    def _extract_best_snippet(self, chunk: Chunk, question: str, max_snippet_length: int = 300) -> str:
        """Extract the most relevant snippet from a chunk."""
        if len(chunk.text) <= max_snippet_length:
            return chunk.text
            
        # Split into sentences while preserving context
        sentences = re.split(r'(?<=[.!?])\s+', chunk.text)
        if len(sentences) == 1:
            return chunk.text[:max_snippet_length] + '...'
            
        # Find the sentence with the most question terms
        best_sentence = ''
        best_score = -1
        q_terms = set(term.lower() for term in question.split() if len(term) > 3)
        
        for i, sent in enumerate(sentences):
            sent_lower = sent.lower()
            score = sum(1 for term in q_terms if term in sent_lower)
            
            # Boost for patterns
            for pattern in self._patterns.values():
                if pattern.search(sent):
                    score += 2
            
            if score > best_score:
                best_score = score
                best_sentence = sent
        
        # Include surrounding context if possible
        if best_sentence:
            start = max(0, sentences.index(best_sentence) - 1)
            end = min(len(sentences), sentences.index(best_sentence) + 2)
            snippet = ' '.join(sentences[start:end])
            
            if len(snippet) > max_snippet_length:
                snippet = snippet[:max_snippet_length].rsplit(' ', 1)[0] + '...'
                
            return snippet
            
        return chunk.text[:max_snippet_length] + '...'
    
    def select_clauses(
        self, 
        question: str, 
        retrieved: List[Chunk], 
        max_chunks: Optional[int] = None,
        min_score: float = 0.1
    ) -> List[Chunk]:
        """Select the most relevant chunks for answering the question.
        
        Args:
            question: The user's question
            retrieved: List of retrieved chunks with scores
            max_chunks: Maximum number of chunks to return (overrides instance setting)
            min_score: Minimum score threshold for including a chunk
            
        Returns:
            List of selected chunks, ordered by relevance
        """
        if not question or not retrieved:
            return []
            
        max_chunks = max_chunks or self.max_chunks
        q_types = self._detect_question_type(question)
        logger.debug(f"Detected question types: {q_types}")
        
        # Score and filter chunks
        scored_chunks = []
        for chunk in retrieved:
            if not chunk.text.strip():
                continue
                
            # Score the chunk
            score = self._score_chunk(chunk, question, q_types)
            
            # Create a new chunk with the enhanced score
            if score >= min_score:
                enhanced_chunk = Chunk(
                    text=chunk.text,
                    meta=chunk.meta
                )
                enhanced_chunk.score = score
                scored_chunks.append(enhanced_chunk)
        
        # Sort by score and take top N
        scored_chunks.sort(key=lambda c: c.score, reverse=True)
        selected_chunks = scored_chunks[:max_chunks]
        
        # Extract the most relevant snippet from each chunk
        for i, chunk in enumerate(selected_chunks):
            if len(chunk.text) > 300:  # Only process long chunks
                snippet = self._extract_best_snippet(chunk, question)
                if snippet != chunk.text:
                    selected_chunks[i] = Chunk(
                        text=snippet,
                        meta=chunk.meta
                    )
                    selected_chunks[i].score = chunk.score
        
        logger.debug(f"Selected {len(selected_chunks)} chunks with scores: {[c.score for c in selected_chunks]}")
        return selected_chunks
