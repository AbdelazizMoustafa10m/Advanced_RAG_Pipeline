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
- Modular pipeline with clear separation of concerns between detectors, processors, and enrichers
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
→ [Vector Store] (Chroma, etc.)
→ [Query Engine]
```

- **main.py**: Entry point. Loads config, initializes pipeline, runs orchestrator, and saves output.
- **pipeline/orchestrator.py**: Core orchestrator that manages loading, routing, processing, and enrichment.
- **core/config.py**: Centralized configuration using dataclasses (for LLMs, loaders, detectors, etc.)
- **processors/**: Contains code and document processors, chunkers, and enrichers.
- **loaders/**: Specialized loaders for directories, code, and documents.
- **detectors/**: Enhanced document detection with multiple strategies and confidence scoring.
- **llm/**: LLM provider integration, prompt templates, and caching.
- **indexing/**: Vector store adapters and query engine.
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
  - Vector store backend
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
- **Vector Store**: Stores embeddings for semantic search (Chroma by default)

## Output

- All processed and enriched nodes are saved to `node_contents.txt` (and variants)
- Includes LLM and embedding model views for each node with complete enrichment information (summaries, titles, questions)
- Consistent formatting of complex metadata through template-based consolidation
- Registry statistics show document counts by type and status
- Detailed listing of all tracked documents with their processing status

## Extending

- Add new document types by extending the EnhancedDetectorService with additional detection strategies
- Add new processors, enrichers, or loaders by extending the respective modules
- Integrate new LLMs or vector stores by updating config and adapters
- Implement custom detection strategies by following the detector pattern

## License
MIT License

---
