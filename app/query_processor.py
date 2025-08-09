"""Query preprocessing and expansion for insurance domain.

This module enhances search queries with insurance-specific term expansions
and context to improve retrieval accuracy.
"""
import re
from typing import Dict, List, Set, Optional
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class InsuranceQueryProcessor:
    """Processes and enhances insurance-related search queries.
    
    Handles:
    - Domain-specific term expansion
    - Query type detection
    - Contextual query enhancement
    """
    
    # Insurance-specific term expansions
    INSURANCE_EXPANSIONS = {
        # Core insurance terms
        'policy': ['policy', 'plan', 'contract', 'agreement', 'coverage'],
        'coverage': ['coverage', 'protection', 'insurance', 'benefit', 'inclusion'],
        'claim': ['claim', 'request', 'application', 'submission', 'reimbursement'],
        'premium': ['premium', 'payment', 'installment', 'fee', 'cost'],
        
        # Common insurance-specific terms
        'grace period': ['grace period', 'premium payment due', 'renewal deadline', 'late payment window'],
        'waiting period': ['waiting period', 'initial period', 'exclusion period', 'qualifying period'],
        'maternity': ['maternity', 'pregnancy', 'childbirth', 'delivery', 'prenatal', 'postnatal'],
        'pre-existing': ['pre-existing', 'pre-existing condition', 'pre-existing disease', 'pre-existing illness'],
        'ncd': ['ncd', 'no claim discount', 'no claim bonus', 'renewal discount'],
        'room rent': ['room rent', 'room charges', 'hospital room', 'ward charges'],
        'daycare': ['daycare', 'day care', 'day procedure', 'short stay'],
        'dental': ['dental', 'tooth', 'teeth', 'gum', 'oral'],
        'ophthalmic': ['ophthalmic', 'eye', 'vision', 'cataract', 'lasik'],
        'preventive': ['preventive', 'prevention', 'checkup', 'screening', 'exam'],
        'chronic': ['chronic', 'long-term', 'ongoing', 'persistent'],
        'pre-existing condition': ['pre-existing condition', 'prior condition', 'existing illness', 'previous disease'],
        'sum insured': ['sum insured', 'coverage amount', 'policy limit', 'maximum benefit'],
        'co-payment': ['co-payment', 'co-pay', 'cost sharing', 'patient share'],
        'deductible': ['deductible', 'excess', 'out-of-pocket', 'initial amount'],
        'network hospital': ['network hospital', 'empaneled hospital', 'tie-up hospital', 'preferred provider'],
        'cashless': ['cashless', 'direct billing', 'no payment', 'hospital direct'],
        'pre-authorization': ['pre-authorization', 'pre-approval', 'prior approval', 'treatment authorization']
    }
    
    # Question patterns for intent detection
    QUESTION_PATTERNS = {
        'time_period': [
            r'how (long|much time)',
            r'what is the (?:duration|period|time|length)',
            r'when (?:does|will|can)',
            r'after (?:how many|what period)'
        ],
        'amount': [
            r'how much',
            r'what is the (?:amount|cost|price|sum|limit)',
            r'maximum (?:amount|sum|limit|benefit)',
            r'up to (?:how much|what amount)'
        ],
        'yes_no': [
            r'\b(?:is|are|does|do|can|will|would|could|should|have|has|had|was|were|did|may|might|must|shall|need)\b',
            r'\b(?:is|are|does|do|can|will|would|could|should|have|has|had|was|were|did|may|might|must|shall|need)n\'?t\b',
            r'\b(?:is|are|does|do|can|will|would|could|should|have|has|had|was|were|did|may|might|must|shall|need) not\b'
        ],
        'list': [
            r'what (?:are|is) the',
            r'list of',
            r'name (?:the|some|all)',
            r'which (?:of the|types of|kind of)'
        ]
    }
    
    def __init__(self):
        # Compile regex patterns once
        self._patterns = {
            intent: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for intent, patterns in self.QUESTION_PATTERNS.items()
        }
    
    def detect_query_type(self, query: str) -> List[str]:
        """Detect the type of the query to guide search strategy.
        
        Args:
            query: The user's search query
            
        Returns:
            List of detected query types (can be multiple)
        """
        query_lower = query.lower()
        types = []
        
        for intent, patterns in self._patterns.items():
            if any(pattern.search(query_lower) for pattern in patterns):
                types.append(intent)
        
        return types or ['general']
    
    def expand_insurance_terms(self, query: str) -> str:
        """Expand insurance-specific terms in the query.
        
        Args:
            query: The original search query
            
        Returns:
            Expanded query with additional relevant terms
        """
        if not query or not query.strip():
            return query
            
        query_lower = query.lower()
        expanded_terms = set(query.split())
        
        # Add expansions for matching terms
        for term, expansions in self.INSURANCE_EXPANSIONS.items():
            if term in query_lower:
                expanded_terms.update(expansions)
                logger.debug(f"Expanded term: {term} -> {expansions}")
        
        # If no expansions were added, try partial matches
        if len(expanded_terms) == len(query.split()) and len(query.split()) < 5:
            for term, expansions in self.INSURANCE_EXPANSIONS.items():
                if any(word in query_lower for word in term.split()):
                    expanded_terms.update(expansions)
                    logger.debug(f"Partially matched and expanded: {term}")
        
        # Add context based on query type
        query_types = self.detect_query_type(query)
        if 'time_period' in query_types:
            expanded_terms.update(['duration', 'period', 'timeframe', 'length', 'days', 'months', 'years'])
        if 'amount' in query_types:
            expanded_terms.update(['amount', 'sum', 'limit', 'maximum', 'coverage', 'benefit'])
        
        # Remove very short words that add noise
        expanded_terms = {t for t in expanded_terms if len(t) > 2}
        
        # Combine original query with expanded terms
        result = f"{' '.join(expanded_terms)} {query}"
        
        # Log the expansion for debugging
        logger.debug(f"Query expanded: '{query}' -> '{result}'")
        
        return result.strip()
    
    def preprocess_query(self, query: str) -> str:
        """Main entry point for query preprocessing.
        
        Args:
            query: The original search query
            
        Returns:
            Processed and expanded query
        """
        if not query or not query.strip():
            return query
            
        try:
            # Basic normalization
            query = ' '.join(query.split())
            
            # Expand insurance terms
            expanded = self.expand_insurance_terms(query)
            
            # Remove duplicate words while preserving order
            words = []
            seen = set()
            for word in expanded.split():
                if word not in seen:
                    seen.add(word)
                    words.append(word)
            
            return ' '.join(words)
            
        except Exception as e:
            logger.error(f"Error preprocessing query '{query}': {e}")
            return query  # Return original on error

# Singleton instance for easy import
query_processor = InsuranceQueryProcessor()

# Example usage:
if __name__ == "__main__":
    processor = InsuranceQueryProcessor()
    test_queries = [
        "What is the grace period for premium payment?",
        "Does the policy cover maternity expenses?",
        "What is the waiting period for pre-existing diseases?",
        "How to claim health insurance?",
        "What is the room rent limit?"
    ]
    
    for q in test_queries:
        print(f"\nOriginal: {q}")
        print(f"Expanded: {processor.preprocess_query(q)}")
        print(f"Detected types: {processor.detect_query_type(q)}")
