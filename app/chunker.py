from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional, Set, Callable
from dataclasses import dataclass
import re
import os
import time
from functools import lru_cache, partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Local imports
from .utils import ChunkMeta, sha1_hex, logger

# DOCUMENT TYPE DETECTION WITH CACHING

def _detect_document_type_fast(pages_by_url: Dict[str, List[Tuple[int, str]]]) -> str:
    """Fast document type detection with caching and optimized text sampling.
    
    Args:
        pages_by_url: Dictionary mapping document URLs to page content
        
    Returns:
        Document type as string ('insurance' or 'general')
    """
    # Generate a unique cache key based on document content
    cache_key = _generate_document_fingerprint(pages_by_url)
    
    # Check cache first
    if cache_key in _doc_type_cache:
        return _doc_type_cache[cache_key]
    
    # Sample only first 1000 characters for efficiency
    sample_text = ""
    for url, pages in pages_by_url.items():
        for _, text in pages[:2]:  # Only first 2 pages
            sample_text += text.lower()
            if len(sample_text) > 1000:
                break
        if len(sample_text) > 1000:
            break
    
    # Simple keyword-based detection (faster than regex for this case)
    insurance_indicators = {
        'policy', 'premium', 'coverage', 'claim', 'deductible',
        'beneficiary', 'exclusion', 'rider', 'underwriting', 'endorsement'
    }
    
    # Count insurance terms in sample
    term_count = sum(1 for term in insurance_indicators if term in sample_text)
    doc_type = 'insurance' if term_count >= 3 else 'general'
    
    # Cache the result
    _doc_type_cache[cache_key] = doc_type
    
    # Clean up old cache entries if needed
    if len(_doc_type_cache) > 1000:  # Keep cache size reasonable
        _doc_type_cache.pop(next(iter(_doc_type_cache)), None)
    
    return doc_type

def _generate_document_fingerprint(pages_by_url: Dict[str, List[Tuple[int, str]]]) -> str:
    """Generate a fingerprint for the document based on its content."""
    # Use first 100 chars of first page as fingerprint (fast and good enough for deduplication)
    try:
        first_page = next(iter(pages_by_url.values()))[0][1]
        return sha1_hex(first_page[:100].encode())
    except (StopIteration, IndexError):
        return ""

# PRE-COMPILED REGEX PATTERNS FOR INSURANCE DOCS
_INSURANCE_PATTERNS = [
    # Section headers
    (re.compile(r'(Clause\s+\d+)', re.IGNORECASE), r'\n\n\1'),
    (re.compile(r'(Section\s+[A-Z0-9.]+)'), r'\n\n\1'),
    (re.compile(r'(Article\s+[IVXL]+)'), r'\n\n\1'),
    
    # Common insurance terms that benefit from spacing
    (re.compile(r'(\b(?:coverage|exclusion|endorsement|rider)\s*:)', re.IGNORECASE), 
     r'\n\1'),
     
    # Numbered lists
    (re.compile(r'(\n\s*\d+\.\s)'), r'\n\n\1'),
    
    # Remove excessive newlines
    (re.compile(r'\n{3,}'), '\n\n'),
]

def _preprocess_insurance_text_fast(text: str) -> str:
    """Optimized text preprocessing for insurance documents.
    
    Uses pre-compiled regex patterns and minimizes string operations.
    """
    if not text or not text.strip():
        return text
        
    processed = text
    for pattern, replacement in _INSURANCE_PATTERNS:
        processed = pattern.sub(replacement, processed)
        
    return processed.strip()

# CHUNKING CONFIGURATION
# Optimized for insurance documents
CHUNK_TOKENS: int = 800    # Reduced from 1200 for better accuracy
CHUNK_OVERLAP: int = 100   # Reduced from 200 for less redundancy
CHARS_PER_TOKEN: float = 3.2  # Insurance docs tend to have longer words

# SIMPLIFIED CHUNK QUALITY ASSESSMENT
@dataclass
class Chunk:
    """Represents a chunk of text with metadata."""
    text: str
    meta: ChunkMeta
    score: float = 0.0
    doc_type: str = 'general'
    
    def __post_init__(self):
        # Simple quality check during initialization
        if not self.text.strip():
            self.score = 0.0
        else:
            # Basic quality metrics (simplified from original)
            has_terminator = any(c in self.text for c in '.!?')
            has_insurance_terms = any(term in self.text.lower() for term in 
                                   ('policy', 'coverage', 'claim', 'deductible'))
            self.score = 1.0 if (has_terminator and has_insurance_terms) else 0.5


# Cache for document type detection (URL -> doc_type)
_doc_type_cache = {}

def chunk_texts(
    pages_by_url: Dict[str, List[Tuple[int, str]]],
    chunk_tokens: int = CHUNK_TOKENS,
    overlap: int = CHUNK_OVERLAP,
    parallel: bool = True
) -> List[Chunk]:
    """Optimized chunking of document pages into semantic chunks.
    
    Args:
        pages_by_url: Dictionary mapping document URLs to (page_num, text) tuples
        chunk_tokens: Target number of tokens per chunk (default: 800)
        overlap: Number of tokens to overlap between chunks (default: 100)
        parallel: Whether to process documents in parallel (default: True)
        
    Returns:
        List of Chunk objects with text and metadata
    """
    start_time = time.time()
    
    if not pages_by_url:
        return []
    
    # Process documents in parallel if requested and more than one
    if parallel and len(pages_by_url) > 1:
        with ThreadPoolExecutor() as executor:
            futures = []
            for url, pages in pages_by_url.items():
                future = executor.submit(
                    _process_single_document,
                    {url: pages},
                    chunk_tokens,
                    overlap
                )
                futures.append(future)
            
            # Collect and flatten results
            chunks = []
            for future in as_completed(futures):
                try:
                    chunks.extend(future.result())
                except Exception as e:
                    logger.error(f"Error processing document: {e}")
            
            logger.info(f"Chunked {len(pages_by_url)} documents in "
                      f"{time.time() - start_time:.2f}s")
            return chunks
    
    # Single document processing
    return _process_single_document(pages_by_url, chunk_tokens, overlap)

def _process_single_document(
    pages_by_url: Dict[str, List[Tuple[int, str]]],
    chunk_tokens: int,
    overlap: int
) -> List[Chunk]:
    """Process a single document into chunks."""
    if not pages_by_url:
        return []
    
    # Get document type (with caching)
    doc_type = _detect_document_type_fast(pages_by_url)
    
    # Build documents with optimized preprocessing
    docs = _build_docs_from_pages(pages_by_url, doc_type)
    if not docs:
        return []
    
    # Configure text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap,
        length_function=lambda x: int(len(x) / CHARS_PER_TOKEN)
    )
    
    # Split documents into chunks
    chunks = []
    for doc in docs:
        try:
            # Get chunks from splitter
            split_chunks = splitter.split_documents([doc])
            
            # Convert to our Chunk format
            for i, chunk in enumerate(split_chunks):
                # Create metadata
                meta = ChunkMeta(
                    doc_id=doc.metadata.get('source', ''),
                    page_num=doc.metadata.get('page', 0),
                    chunk_id=i,
                    doc_type=doc_type
                )
                
                # Create chunk with basic quality scoring
                chunk = Chunk(
                    text=chunk.page_content,
                    meta=meta,
                    doc_type=doc_type
                )
                
                # Only keep chunks with some content
                if chunk.text.strip():
                    chunks.append(chunk)
                    
        except Exception as e:
            logger.error(f"Error chunking document {doc.metadata.get('source', '')}: {e}")
    
    return chunks

def _build_docs_from_pages(
    pages_by_url: Dict[str, List[Tuple[int, str]]], 
    doc_type: str
) -> List[Document]:
    """Convert page-level texts to LangChain Documents with optimized preprocessing.
    
    Args:
        pages_by_url: Dictionary of URL to (page_num, text) tuples
        doc_type: Type of document ('insurance' or 'general')
        
    Returns:
        List of preprocessed Document objects
    """
    docs = []
    
    for url, pages in pages_by_url.items():
        for page_num, text in pages:
            # Skip empty or very short texts
            if not text or len(text.strip()) < 10:
                continue
                
            # Create Document with metadata
            metadata = {
                "source": url,
                "page": page_num,
                "start_index": 0,  # Will be updated by splitter
                "end_index": len(text)
            }
            docs.append(Document(page_content=text, metadata=metadata))
    return docs

# Cache document type detection
_doc_type_cache = {}

@lru_cache(maxsize=100)
def _get_separators_cached(doc_type: str) -> List[str]:
    """Cached separator retrieval"""
    if doc_type == 'insurance':
        return ["\n\n\n", "\n\nClause ", "\n\nSection ", "\n\n", "\n", ". ", " ", ""]
    elif doc_type == 'legal':
        return ["\n\n\n", "\n\n(", "\n\nSection ", "\n\n", "\n", ". ", " ", ""]
    else:
        return ["\n\n", "\n", ". ", " ", ""]

def _detect_document_type_fast(pages_by_url: Dict[str, List[Tuple[int, str]]]) -> str:
    """Fast document type detection with caching"""
    # Create cache key from first URL and first page snippet
    first_url = next(iter(pages_by_url.keys()))
    cache_key = first_url
    
    if cache_key in _doc_type_cache:
        return _doc_type_cache[cache_key]
    
    # Sample only first 1000 chars from first page for speed
    sample_text = ""
    for url, pages in pages_by_url.items():
        if pages:
            sample_text = pages[0][1][:1000].lower()
            break
    
    # Simplified detection with fewer terms
    if any(term in sample_text for term in ['policy', 'premium', 'coverage', 'insured']):
        doc_type = 'insurance'
    elif any(term in sample_text for term in ['clause', 'section', 'whereas']):
        doc_type = 'legal'
    else:
        doc_type = 'general'
    
    _doc_type_cache[cache_key] = doc_type
    return doc_type

# Compile regex patterns once
_INSURANCE_PATTERNS = [
    (re.compile(r'(Clause\s+\d+)', re.IGNORECASE), r'\n\n\1'),
    (re.compile(r'(Section\s+\d+)', re.IGNORECASE), r'\n\n\1'),
    # ... other patterns compiled once
]

def _preprocess_insurance_text_fast(text: str) -> str:
    """Fast preprocessing using pre-compiled patterns"""
    # Early exit for small texts
    if len(text) < 200:
        return text
        
    processed = text
    for compiled_pattern, replacement in _INSURANCE_PATTERNS:
        processed = compiled_pattern.sub(replacement, processed)
    
    return processed

def chunk_texts_optimized(
    pages_by_url: Dict[str, List[Tuple[int, str]]],
    chunk_tokens: int = 800,  # Reduced back to reasonable size
    overlap: int = 100,       # Reduced overlap
) -> List[Chunk]:
    """Optimized chunking with performance improvements"""
    if not pages_by_url:
        return []

    # Fast document type detection
    doc_type = _detect_document_type_fast(pages_by_url)
    
    # Use cached separators
    separators = _get_separators_cached(doc_type)
    
    # Reasonable chunk sizes
    chars_per_token = 3.5 if doc_type == 'insurance' else 4.0
    chunk_size_chars = int(chunk_tokens * chars_per_token)
    chunk_overlap_chars = int(overlap * chars_per_token)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_overlap_chars,
        add_start_index=True,
        separators=separators,
        keep_separator=False,  # Faster processing
    )

    # Minimal preprocessing
    if doc_type == 'insurance':
        base_docs = []
        for url, pages in pages_by_url.items():
            for page_num, text in pages:
                # Light preprocessing only
                processed = _preprocess_insurance_text_fast(text)
                doc = Document(page_content=processed, metadata={"source": url, "page": page_num})
                base_docs.append(doc)
    else:
        base_docs = _build_docs_from_pages(pages_by_url)

    split_docs = splitter.split_documents(base_docs)
    
    # Streamlined chunk creation
    chunks = []
    for d in split_docs:
        # Skip quality assessment during chunking
        text = d.page_content.strip()
        if len(text) < 50:  # Simple length filter
            continue

        url = str(d.metadata.get("source", ""))
        page = int(d.metadata.get("page", 1))
        start = int(d.metadata.get("start_index", 0))
        end = start + len(text)

        cid = sha1_hex(f"{url}|{page}|{start}")[:12]  # Simplified ID
        meta = ChunkMeta(doc_url=url, page=page, chunk_id=cid, start=start, end=end)
        chunks.append(Chunk(text=text, meta=meta))

    return chunks