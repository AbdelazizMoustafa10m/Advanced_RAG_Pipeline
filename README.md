# Advanced RAG Pipeline

## Overview

**Advanced RAG** is a unified, highly configurable pipeline for Retrieval-Augmented Generation (RAG) that efficiently processes both code and technical documents (e.g., PDFs, DOCX). It leverages advanced document loaders, chunking, enrichment, and vector storage to enable high-quality semantic search and LLM-augmented querying over heterogeneous knowledge bases.

The pipeline is designed for:

- **Efficient processing of large codebases and document repositories**
- **Intelligent document routing and chunking** (using Docling and custom logic)
- **Metadata enrichment for both code and documents**
- **Flexible LLM integration for enrichment and query**
- **Seamless vector storage and retrieval**

## Key Features

- Enhanced document detection with multiple strategies (extension, content analysis, magic numbers)
- Advanced code chunking with Chonkie AST-based parsing and multi-strategy fallback
- Specialized document loaders for PDFs (DoclingReader), code files, markdown, and more
- Document Registry for efficient caching and idempotent processing with accurate path tracking
- Smart skipping of previously processed documents based on modification time
- Document type classification with confidence scoring for accurate file type detection
- Intelligent grouping of document parts for efficient processing and to avoid redundant LLM calls
- Modular embedding system supporting multiple providers (HuggingFace, OpenAI, Cohere, Vertex, Bedrock, Ollama) with batch processing, disk-based caching, and provider fallback
- Modular vector store system supporting ChromaDB, Qdrant (local/cloud), and SimpleVectorStore (in-memory fallback) via configuration and factory pattern
- Multi-level fallback chains for both embedding and vector storage, ensuring robust operation even if dependencies are missing
- Modular pipeline with clear separation of concerns between detectors, processors, enrichers, embedders, and vector stores
- Parallel processing support with optimized batch operations
- Configurable via Python dataclasses and environment variables
- Output includes enriched nodes for both LLM and embedding models with guaranteed visibility of metadata enrichment

## Architecture

```mermaid
[data dir] → [DocumentRegistry] → [EnhancedDirectoryLoader] → [EnhancedDetectorService] → [DocumentTypeRouter]
   └─> [CodeProcessor] (for code)
   └─> [TechnicalDocumentProcessor] (for docs, e.g., PDF)
         └─> [DoclingChunker]
         └─> [DoclingMetadataGenerator]
   └─> [MarkdownProcessor] (for markdown)
   └─> [CSVProcessor] (for spreadsheets)
   └─> [Metadata Enrichers]
→ [EmbedderService] (HuggingFace, OpenAI, Cohere, Ollama, etc.)
→ [Vector Store] (ChromaDB, Qdrant, SimpleVectorStore)
→ [Query Engine]
```

- **main.py**: Entry point. Loads config, initializes pipeline, runs orchestrator, and saves output.
- **pipeline/orchestrator.py**: Core orchestrator that manages loading, routing, processing, and enrichment.
- **core/config.py**: Centralized configuration using dataclasses (for LLMs, loaders, detectors, etc.)
- **processors/**: Contains code and document processors, chunkers, and enrichers.
- **loaders/**: Specialized loaders for directories, code, and documents.
- **detectors/**: Enhanced document detection with multiple strategies and confidence scoring.
- **llm/**: LLM provider integration, prompt templates, and caching.
- **embedders/**: Embedding service with support for multiple providers, batch processing, caching, and provider fallback (see `llamaindex_embedder_service.py`, `embedder_factory.py`).
- **indexing/**: Vector store adapters (ChromaDB, Qdrant, SimpleVectorStore) and query engine, with factory-based selection and fallback (`vector_store.py`).
- **registry/**: Document tracking and status management with accurate path handling.

## Setup

1. **Clone the repository**
2. *(Recommended)* Create and activate a Python virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   - Copy `.env.example` to `.env` and fill in API keys for LLM providers (if needed).

5. **Prepare your data:**
   - Place your code and document files in the `data/` directory (or configure another input directory).

## Usage

Run the main pipeline:

```bash
python main.py
```

- The pipeline processes all files from the input directory, routes and chunks them, enriches with LLMs (if enabled), and saves the results to `node_contents.txt`.
- Output statistics and summaries are printed to the console.

## Configuration

- All configuration is managed via `core/config.py` using Python dataclasses.
- You can customize:
  - Input/output directories
  - Parallelism
  - LLM providers and enrichment options
  - Vector store backend (chroma/qdrant/simple) and all provider-specific options
  - Embedding provider, model, batch size, caching, and advanced settings
  - Code chunking strategies (Chonkie AST, LlamaIndex AST, semantic line, basic line)
  - Document chunking and enrichment strategies
  - Document Registry settings (database path, stalled processing timeout)

## Main Components

- **DocumentRegistry**: Tracks document processing status with accurate path handling to prevent duplicate work
- **EnhancedDirectoryLoader**: Loads files and routes them to specialized loaders based on document type
- **EnhancedDetectorService**: Identifies document types using multiple detection strategies with confidence scoring
- **DoclingReader**: Specialized PDF/document loader and chunker
- **CodeProcessor**: Splits and processes code files with multi-strategy chunking (Chonkie AST, LlamaIndex AST, semantic line, basic line)
- **TechnicalDocumentProcessor**: Handles technical docs (PDF, DOCX, etc.)
- **MarkdownProcessor**: Processes markdown files with specialized chunking
- **Metadata Generators**: Enrich nodes with metadata using LLMs
- **DoclingMetadataFormatter**: Formats complex document metadata with template-based approach to ensure enrichment visibility in both LLM and embedding contexts
- **EmbedderService**: Generates embeddings using configurable providers (HuggingFace, OpenAI, Cohere, Vertex, Bedrock, Ollama, etc.) with batch processing, disk-based caching, and provider fallback. See `embedders/llamaindex_embedder_service.py` and `embedders/embedder_factory.py` for details.
- **Vector Store**: Stores embeddings for semantic search. Supports ChromaDB, Qdrant (local/cloud), and SimpleVectorStore (in-memory fallback) via configuration. See `indexing/vector_store.py`.

## Output

- All processed and enriched nodes are saved to `node_contents.txt` (and variants)
- Includes LLM and embedding model views for each node with complete enrichment information (summaries, titles, questions)
- Consistent formatting of complex metadata through template-based consolidation
- Registry statistics show document counts by type and status
- Detailed listing of all tracked documents with their processing status

## Configuration for Embedding and Vector Store

- All options for embedding and vector store selection are set in `.env` or via `core/config.py` dataclasses.
- Example configuration options:
  - `EMBEDDER_PROVIDER` (huggingface/openai/cohere/ollama/vertex/bedrock)
  - `EMBEDDER_MODEL` (e.g., BAAI/bge-small-en-v1.5)
  - `EMBEDDER_BATCH_SIZE`, `EMBEDDER_CACHE_PATH`, etc.
  - `VECTOR_STORE_ENGINE` (chroma/qdrant/simple)
  - `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_LOCATION`, etc.
- See `.env_example` for all available options and usage patterns.

## Examples

- See `examples/embedder_example.py` for usage of multiple embedding providers with batch/caching.
- See `examples/vector_store_example.py` for vector store configuration and fallback in action.

## Benefits

- **Reliability**: Multi-level fallback for both embedding and vector storage ensures robust operation in all environments.
- **Extensibility**: Easily add new embedding/vector store providers via the factory pattern.
- **Testing**: In-memory vector store and mock embedders make testing easy and dependency-free.
- **Performance**: Batch embedding, caching, and advanced node preparation for efficiency.

## Extending

- Add new document types by extending the EnhancedDetectorService with additional detection strategies
- Add new processors, enrichers, or loaders by extending the respective modules
- Add new embedding providers by extending the LlamaIndexEmbedderService and updating the factory/config
- Add new vector stores by creating a new adapter in `indexing/`, updating the factory/config
- Implement custom detection strategies by following the detector pattern

## License
MIT License

---
