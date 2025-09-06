# Local RAG Assistant - Standalone Software

## Problem Statement

Modern professionals and researchers deal with large volumes of unstructured data requiring:
- AI assistant for domain-specific questions
- Offline capabilities for security and privacy
- Personalized knowledge bases reflecting user expertise
- Cost-effective solutions without cloud APIs

## Solution

**Local RAG Assistant** - A fully offline Retrieval-Augmented Generation system operating on a single computer.

## Key Features

### 🔒 Fully Offline
- Runs locally with llama.cpp using quantized LLM models
- No internet dependency once set up
- Complete data privacy and security

### 🗄️ Local Knowledge Base with Vector Search
- FAISS for high-speed document retrieval
- Processes .txt, .md, and .pdf files into embeddings
- Powered by sentence-transformers

### 🎯 Personalized Answers
- Embeds domain knowledge for contextual responses
- Provides source citations for transparency
- Customizable system prompts

### 🔐 License/Subscription Support
- RSA-based license key generation
- Commercial knowledge sharing capabilities
- Intellectual property protection

### 🖥️ User-Friendly Interfaces
- CLI for power users
- Streamlit web UI for easy interaction
- RESTful API support

### ⚙️ Automation
- Makefile for one-line setup and operations
- Automated model management
- Configuration-driven deployment

## Technical Architecture

```
         ┌───────────────┐
         │ User Queries   │
         └───────┬────────┘
                 │
        ┌────────▼────────┐
        │ RAG Pipeline    │
        │ (Retriever+LLM) │
        └───────┬─────────┘
                │
    ┌───────────▼────────────┐
    │ Vector DB (FAISS)      │  ←── Embeddings (Sentence-Transformers)
    └───────────┬────────────┘
                │
    ┌───────────▼──────────┐
    │ Metadata (SQLite)     │  ←── Document Titles, Paths
    └───────────┬──────────┘
                │
    ┌───────────▼──────────┐
    │ Local LLM (llama.cpp) │  ←── Offline LLaMA/Mistral model
    └───────────────────────┘
```

## Quick Start

### 1. Setup

**Easy Setup** (recommended - interactive setup with prompts):
```bash
python3 setup.py    # Automated setup with model download options
```

**Manual Setup**:
```bash
make setup          # Install dependencies and create environment
make download-model  # Download recommended model
```

### 2. Index Documents

Choose your scanning mode based on your needs:

**Manual Mode** (default - scan only data/ folder):
```bash
# Place your documents in data/
cp /path/to/your/docs/* data/
make ingest         # Process and index documents
```

**Home Directory Mode** (scan user's home folder):
```bash
make ingest-home    # Index all documents from ~/
```

**System-wide Mode** (scan entire system - requires sudo):
```bash
make ingest-system  # Index documents from entire filesystem
# OR use the dedicated script with safety prompts:
sudo python scripts/system_scan.py
```

**Custom Directories** (configure in settings.yaml):
```bash
# Edit config/settings.yaml to set custom directories
make ingest-custom  # Index from configured directories
```

### 3. Run Assistant
```bash
make run            # Start CLI interface
# OR
make web            # Start web interface
# OR  
make api            # Start API server
```

## Requirements

- Python 3.10+
- 8GB+ RAM recommended
- 10GB+ disk space for models
- macOS/Linux/Windows support

## Project Structure

```
rag-persona/
├── README.md                 # This file
├── Makefile                  # Automation commands
├── requirements.txt          # Python dependencies
├── config/                   
│   ├── settings.yaml         # Main configuration
│   └── models.yaml          # Model configurations
├── src/                     
│   ├── __init__.py          
│   ├── core/                # Core RAG implementation
│   │   ├── __init__.py     
│   │   ├── embedder.py      # Document embedding
│   │   ├── retriever.py     # Vector search
│   │   ├── generator.py     # LLM interface
│   │   └── pipeline.py      # Main RAG pipeline
│   ├── data/                # Data processing
│   │   ├── __init__.py     
│   │   ├── loader.py        # Document loading
│   │   ├── chunker.py       # Text chunking
│   │   └── indexer.py       # Index management
│   ├── licensing/           # License system
│   │   ├── __init__.py     
│   │   ├── generator.py     # License generation
│   │   └── validator.py     # License validation
│   ├── interfaces/          # User interfaces
│   │   ├── __init__.py     
│   │   ├── cli.py           # Command line interface
│   │   ├── web.py           # Streamlit web UI
│   │   └── api.py           # FastAPI REST interface
│   └── utils/               # Utilities
│       ├── __init__.py     
│       ├── config.py        # Configuration management
│       ├── logging.py       # Logging setup
│       └── helpers.py       # Helper functions
├── models/                  # LLM models (gitignored)
├── data/                    # Input documents (gitignored)
├── index/                   # Vector index & metadata (gitignored)
├── logs/                    # Application logs (gitignored)
├── licenses/                # License keys (gitignored)
└── scripts/                 # Utility scripts
    └── system_scan.py       # System-wide scanning script
```

## License

MIT License - See LICENSE file for details.

## Support

For issues and questions, please check the documentation or create an issue in the repository.
