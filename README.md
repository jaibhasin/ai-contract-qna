# HackRX Intelligent Query–Retrieval API

🚀 **Production-grade FastAPI application for intelligent document query and retrieval**

Transform your document analysis workflow with AI-powered search and retrieval over insurance, legal, HR, and compliance documents.

## ✨ Key Features

- 📄 **Multi-format Support**: PDF, DOCX, and Email processing
- 🔍 **Intelligent Search**: Semantic retrieval with Google Gemini embeddings  
- ⚡ **Fast Vector Search**: FAISS-powered similarity search with cosine similarity
- 🎯 **Context-Aware**: Insurance-specific optimizations and clause matching
- 📚 **Source Tracing**: Full provenance tracking with page numbers and document references
- 🛡️ **Enterprise Ready**: Bearer token authentication and configurable security limits

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation) 
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)

## 🚀 Quick Start

### Prerequisites

- **Python 3.10–3.11** (recommended)
- **Google Gemini API Key** (for embeddings and reasoning)
- **Operating System**: Windows, macOS, or Linux

## 📦 Installation

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/jaibhasin/ai-contract-qna.git
cd ai-contract-qna

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
# Required: Authentication
TEAM_TOKEN=your-secure-token-here

# Required: Google Gemini API 
GOOGLE_API_KEY=your_gemini_api_key_here
# Alternative key name also supported:
# GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Start the API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

✅ **Success!** Your API is now running at:
- **API Endpoint**: http://localhost:8000/api/v1/hackrx/run  
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📖 Usage Examples

### Basic Query Example

```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer your-secure-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "https://example.com/policy.pdf",
      "file:///path/to/local/document.pdf"
    ],
    "questions": [
      "What is the waiting period for pre-existing conditions?",
      "What are the coverage limits for outpatient treatment?"
    ]
  }'
```

### Using Local Files

```json
{
  "documents": [
    "file:///home/user/documents/insurance_policy.pdf",
    "file:///home/user/documents/benefits_summary.docx"
  ],
  "questions": [
    "What is the annual deductible amount?",
    "Are dental procedures covered?"
  ]
}
```

### Response Format

```json
{
  "answers": [
    "The waiting period for pre-existing conditions is 24 months from the policy start date, as specified in Section 4.2.",
    "Outpatient treatment is covered up to $5,000 per year with a $50 copayment per visit."
  ]
}
```

## 🏗️ Architecture

The system follows a retrieval-augmented generation (RAG) pipeline:

```
📄 Documents → 🔄 Ingestion → ✂️ Chunking → 🧮 Embeddings → 🗄️ Vector DB
                                                                     ↓
📋 Response ← 🤖 Reasoner ← 🎯 Matcher ← 🔍 Retriever ← 🔎 Semantic Search
```

### Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Ingestion** | PyMuPDF, python-docx, email parser | Extract text from PDFs, DOCX, emails |
| **Chunking** | LangChain RecursiveCharacterTextSplitter | Split documents into ~800-token chunks |
| **Embeddings** | Google Gemini text-embedding-004 | Convert text to vector representations |
| **Vector DB** | FAISS IndexFlatIP | Fast similarity search with cosine similarity |
| **Retriever** | Custom semantic search | Insurance-specific boosts and MMR diversity |
| **Matcher** | Rule-based scoring | Extract relevant clauses with keyword boosts |
| **Reasoner** | Google Gemini LLM | Generate structured answers with source citations |

## ⚙️ Configuration

### Basic Configuration (.env)

The application loads all configuration from a `.env` file at startup.

#### 🔑 Authentication & API Keys
```bash
# Required: Bearer token for API authentication
TEAM_TOKEN=your-secure-token-here

# Required: Google Gemini API key
GOOGLE_API_KEY=your_gemini_api_key_here
```

#### 🤖 AI Model Settings  
```bash
# Embedding model (default: text-embedding-004)
EMBEDDING_MODEL=text-embedding-004

# Language model for reasoning (default: gemini-1.5-flash)  
LLM_MODEL=gemini-1.5-flash

# Embedding cache TTL in seconds (default: 3600)
EMBED_CACHE_TTL=3600
```

#### 📄 Document Processing
```bash
# Chunk size in tokens (default: 800)
CHUNK_TOKENS=800

# Overlap between chunks in tokens (default: 150)  
CHUNK_OVERLAP=150

# Maximum download size in MB (default: 25)
DOWNLOAD_MAX_MB=25
```

#### 🔍 Search & Retrieval
```bash
# Number of top candidates to retrieve (default: 12)
TOP_K=12

# Maximum chunks sent to LLM for final reasoning (default: 6)  
MAX_CHUNKS_TO_LLM=6

# Enable text summarization for long chunks (default: true)
USE_SUMMARIZER=true

# Enable cross-encoder reranking (default: false - for performance)
CROSS_ENCODER_RERANK=false
```

## 🎛️ Advanced Configuration

### Retrieval Accuracy Tuning

For better search results with complex documents:

```bash
# Minimum initial candidates to consider (default: 80)
INITIAL_CANDIDATES_MIN=80

# Multiplier for candidate pool size (default: 8) 
INITIAL_CANDIDATES_MULT=8

# Candidates to keep after early filtering (default: 50)
FILTERED_KEEP=50

# MMR diversity vs relevance balance (0.0=relevance, 1.0=diversity)
MMR_LAMBDA=0.5
```

### Lexical Augmentation

Enhance semantic search with exact-match capabilities:

```bash
# Enable lexical augmentation (default: true)
ENABLE_LEXICAL_AUGMENT=true

# Maximum lexical matches to add (default: 10)  
LEXICAL_MAX_ADD=10
```

### Performance Optimization

- **Higher `TOP_K`**: Better recall, more computation
- **Lower `MMR_LAMBDA`**: Favor relevance over diversity  
- **Enable `LEXICAL_AUGMENT`**: Better exact-match performance
- **Increase `INITIAL_CANDIDATES_MIN`**: Better for long, varied documents

## 🐛 Troubleshooting

### Common Issues

#### 🚫 Authentication Errors
```
401 Unauthorized
```
**Solution**: Check your `TEAM_TOKEN` in the `.env` file and ensure it matches the `Authorization: Bearer <token>` header.

#### 🔑 API Key Issues  
```
Error: Google API key not found
```
**Solution**: Verify `GOOGLE_API_KEY` or `GEMINI_API_KEY` is set in your `.env` file.

#### 📄 File Processing Errors
```
Error: Could not process document
```
**Solutions**:
- Ensure file URLs are properly formatted (`file:///absolute/path`)
- Check file permissions and accessibility
- Verify document format is supported (PDF, DOCX, EML)
- Check `DOWNLOAD_MAX_MB` limit for large files

#### 🔍 Poor Search Results
**Solutions**:
- Increase `TOP_K` for more candidates
- Enable `ENABLE_LEXICAL_AUGMENT=true` for exact matches
- Adjust `MMR_LAMBDA` (lower for more relevance)
- Increase `INITIAL_CANDIDATES_MIN` for complex documents

### Debug Mode

Enable detailed logging:

```bash
export LOG_LEVEL=DEBUG
uvicorn app.main:app --log-level debug --reload
```

## 📁 Project Structure

```
ai-contract-qna/
├── app/                          # Main application package
│   ├── main.py                   # FastAPI app and endpoint orchestration
│   ├── ingest.py                 # Document download and text extraction  
│   ├── chunker.py                # Text splitting with LangChain loaders
│   ├── embeddings.py             # Gemini embedding client with caching
│   ├── vectordb.py               # FAISS vector store with metadata
│   ├── retriever.py              # Semantic search with insurance optimizations
│   ├── matcher.py                # Clause extraction and rule-based scoring
│   ├── reasoner.py               # LLM reasoning with deterministic fallback
│   ├── models.py                 # Pydantic data models
│   └── utils.py                  # Shared utilities and helpers
├── design.md                     # Architecture and design documentation
├── CHANGELOG.md                  # Version history and changes
├── requirements.txt              # Python dependencies
├── runtime.txt                   # Python version specification  
└── README.md                     # This file
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `main.py` | FastAPI application, routing, authentication, configuration |
| `ingest.py` | Downloads and extracts text from PDF, DOCX, email files |  
| `chunker.py` | Document splitting using LangChain with stable chunk IDs |
| `embeddings.py` | Vector embeddings via Gemini with TTL caching |
| `vectordb.py` | FAISS-based similarity search with provenance metadata |
| `retriever.py` | Multi-stage semantic retrieval with insurance optimizations |
| `matcher.py` | Clause-level text extraction with relevance scoring |
| `reasoner.py` | LLM-powered answer generation with source citations |

## 🔒 Security Features

- **🛡️ Bearer Token Authentication**: All API requests require valid authentication
- **📏 Size Limits**: Configurable download limits prevent resource exhaustion  
- **🔗 URL Sanitization**: Input validation and sanitization for file URLs
- **⚡ Rate Limiting**: Built-in protection against abuse (configurable)
- **🔐 API Key Security**: Secure handling of Google Gemini API credentials

## 📚 Additional Resources

- **[Design Documentation](design.md)**: Detailed architecture and implementation notes
- **[API Documentation](http://localhost:8000/docs)**: Interactive OpenAPI documentation (when server is running)
- **[Changelog](CHANGELOG.md)**: Version history and release notes

## 🤝 Support & Contributing

For questions, issues, or contributions:

1. **Issues**: Report bugs and request features via GitHub Issues
2. **Documentation**: Check `design.md` for technical details  
3. **API Reference**: Use `/docs` endpoint for interactive API exploration

---

**Made with ❤️ for intelligent document processing**
