# Document Registry Tests

This directory contains tests for the Document Registry system used in the Advanced RAG Pipeline.

## Overview

The Document Registry tracks document processing status through the RAG pipeline and ensures:
- Documents are only processed if they're new or have changed (based on content hash)
- Processing status is properly maintained (pending → processing → completed/failed)
- Metadata like processing time, chunk count, and vector database status is recorded
- Stalled documents can be detected and retried

## Test Structure

- **Unit Tests**: Focus on the Document Registry component in isolation
- **Integration Tests**: Test the Document Registry's interaction with the full pipeline

## Dependencies

These tests require:
- pytest
- pytest-mock
- chromadb (for integration tests)

Install them with:

```bash
pip install pytest pytest-mock chromadb
```

## Running Tests

To run all tests:

```bash
# Run from the project root directory
pytest tests/ --maxfail=1 --disable-warnings
```

To run only unit tests:

```bash
pytest tests/unit/
```

To run only integration tests:

```bash
pytest tests/integration/
```

To run a specific test file:

```bash
pytest tests/unit/test_document_registry.py
```

## Test Coverage

The tests cover:

### Unit Tests:
- Document registration and hash computation
- Status transitions (pending → processing → completed/failed)
- Idempotency (same document registered twice)
- Document update detection
- Metadata handling and updates
- Stalled document detection and reset

### Integration Tests:
- End-to-end document processing and status tracking
- Document update detection and selective reprocessing
- Error handling and recovery
- Vector database status tracking

## Troubleshooting

If tests fail with import errors, make sure you're running from the project root directory with:

```bash
cd /Users/abdelazizmoustafa/Desktop/prj/advanced_rag
pytest tests/
```
