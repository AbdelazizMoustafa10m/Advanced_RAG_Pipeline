# tests/unit/test_document_registry.py

import os
import tempfile
import pytest
import json
import time
import sqlite3
from unittest.mock import MagicMock, patch

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from registry.document_registry import DocumentRegistry
from registry.status import ProcessingStatus
from registry.exceptions import DocumentRegistryError, InvalidStatusError

class TestDocumentRegistry:
    """Unit tests for the DocumentRegistry class."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        os.unlink(path)  # Clean up the temp file
    
    @pytest.fixture
    def registry(self, temp_db_path):
        """Create a DocumentRegistry instance with a temp database."""
        return DocumentRegistry(db_path=temp_db_path)
    
    @pytest.fixture
    def sample_document(self):
        """Sample document content and metadata for testing."""
        return {
            'doc_id': 'test_doc_001',
            'content': 'This is a test document.',
            'document_type': 'document',
            'source': 'test_source',
            'metadata': {
                'document_type': 'document',
                'language': 'english',
                'processor': 'TestProcessor',
                'vector_status': 'pending'
            }
        }
    
    def test_initialization(self, registry, temp_db_path):
        """Test that the registry initializes correctly and creates the database."""
        assert os.path.exists(temp_db_path)
        
        # Check that the schema was created
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
            assert cursor.fetchone() is not None
    
    def test_register_document(self, registry, sample_document):
        """Test registering a new document."""
        doc_id = registry.register_document(
            doc_id=sample_document['doc_id'],
            content=sample_document['content'],
            document_type=sample_document['document_type'],
            source=sample_document['source'],
            metadata=sample_document['metadata']
        )
        
        assert doc_id == sample_document['doc_id']
        
        # Verify the document was registered properly
        status = registry.get_document_status(doc_id)
        assert status is not None
        assert status['status'] == ProcessingStatus.PENDING.value
        assert status['document_type'] == sample_document['document_type']
        assert status['metadata']['language'] == 'english'
    
    def test_register_document_with_hash(self, registry, sample_document):
        """Test registering a document with a pre-computed hash."""
        content_hash = registry.compute_hash(sample_document['content'])
        
        doc_id = registry.register_document(
            doc_id=sample_document['doc_id'],
            content_hash=content_hash,
            document_type=sample_document['document_type'],
            source=sample_document['source'],
            metadata=sample_document['metadata']
        )
        
        assert doc_id == sample_document['doc_id']
        
        # Verify the document was registered properly
        status = registry.get_document_status(doc_id)
        assert status is not None
        assert status['content_hash'] == content_hash
    
    def test_update_status_valid_transitions(self, registry, sample_document):
        """Test valid status transitions."""
        # Register the document (PENDING state)
        registry.register_document(
            doc_id=sample_document['doc_id'],
            content=sample_document['content'],
            document_type=sample_document['document_type']
        )
        
        # PENDING → PROCESSING (valid)
        registry.update_status(sample_document['doc_id'], ProcessingStatus.PROCESSING)
        status = registry.get_document_status(sample_document['doc_id'])
        assert status['status'] == ProcessingStatus.PROCESSING.value
        
        # PROCESSING → COMPLETED (valid)
        registry.update_status(sample_document['doc_id'], ProcessingStatus.COMPLETED)
        status = registry.get_document_status(sample_document['doc_id'])
        assert status['status'] == ProcessingStatus.COMPLETED.value
        
        # Register another document for FAILED path
        doc_id_2 = 'test_doc_002'
        registry.register_document(
            doc_id=doc_id_2,
            content=sample_document['content'],
            document_type=sample_document['document_type']
        )
        
        # PENDING → PROCESSING → FAILED (valid)
        registry.update_status(doc_id_2, ProcessingStatus.PROCESSING)
        registry.update_status(doc_id_2, ProcessingStatus.FAILED, error_message="Test error")
        status = registry.get_document_status(doc_id_2)
        assert status['status'] == ProcessingStatus.FAILED.value
        assert status['error_message'] == "Test error"
    
    def test_update_metadata(self, registry, sample_document):
        """Test updating document metadata."""
        # Register the document
        registry.register_document(
            doc_id=sample_document['doc_id'],
            content=sample_document['content'],
            metadata=sample_document['metadata']
        )
        
        # Update the status with new metadata
        new_metadata = {
            'chunk_count': 5,
            'vector_status': 'completed',
            'processing_time': 1.25
        }
        
        registry.update_status(
            sample_document['doc_id'], 
            ProcessingStatus.COMPLETED,
            metadata_updates=new_metadata
        )
        
        # Verify metadata was updated properly
        status = registry.get_document_status(sample_document['doc_id'])
        assert status['metadata']['chunk_count'] == 5
        assert status['metadata']['vector_status'] == 'completed'
        assert status['metadata']['processing_time'] == 1.25
        # Original metadata should still be there
        assert status['metadata']['language'] == 'english'
    
    def test_idempotency(self, registry, sample_document):
        """Test that registering the same document twice doesn't create duplicates."""
        # Register document first time
        registry.register_document(
            doc_id=sample_document['doc_id'],
            content=sample_document['content'],
            metadata=sample_document['metadata']
        )
        
        # Update state to COMPLETED
        registry.update_status(sample_document['doc_id'], ProcessingStatus.PROCESSING)
        registry.update_status(
            sample_document['doc_id'], 
            ProcessingStatus.COMPLETED,
            metadata_updates={'vector_status': 'completed'}
        )
        
        # Get the status and hash
        status_before = registry.get_document_status(sample_document['doc_id'])
        
        # Register the same document again (same content = same hash)
        registry.register_document(
            doc_id=sample_document['doc_id'],
            content=sample_document['content'],
            metadata=sample_document['metadata']
        )
        
        # Check that status didn't change (still COMPLETED)
        status_after = registry.get_document_status(sample_document['doc_id'])
        assert status_after['status'] == status_before['status']
        assert status_after['content_hash'] == status_before['content_hash']
        assert status_after['metadata']['vector_status'] == 'completed'
        
        # Now register with different content (should trigger re-processing)
        registry.register_document(
            doc_id=sample_document['doc_id'],
            content=sample_document['content'] + " Updated content.",
            metadata=sample_document['metadata']
        )
        
        # Check that status changed back to PENDING
        status_updated = registry.get_document_status(sample_document['doc_id'])
        assert status_updated['status'] == ProcessingStatus.PENDING.value
        assert status_updated['content_hash'] != status_before['content_hash']  # Hash should change
    
    def test_should_process(self, registry, sample_document):
        """Test the logic for determining if a document should be processed."""
        # Register and complete a document
        registry.register_document(
            doc_id=sample_document['doc_id'],
            content=sample_document['content']
        )
        registry.update_status(sample_document['doc_id'], ProcessingStatus.PROCESSING)
        registry.update_status(sample_document['doc_id'], ProcessingStatus.COMPLETED)
        
        # Shouldn't process a completed document with same content
        assert not registry.should_process(
            sample_document['doc_id'],
            content=sample_document['content']
        )
        
        # Should process if hash has changed
        assert registry.should_process(
            sample_document['doc_id'],
            content=sample_document['content'] + " Modified content."
        )
        
        # Register another document and fail it
        doc_id_2 = 'test_doc_002'
        registry.register_document(
            doc_id=doc_id_2,
            content=sample_document['content']
        )
        registry.update_status(doc_id_2, ProcessingStatus.PROCESSING)
        registry.update_status(doc_id_2, ProcessingStatus.FAILED, error_message="Test error")
        
        # Should process a failed document even with same content
        assert registry.should_process(
            doc_id_2,
            content=sample_document['content']
        )
        
        # Should always process a document not in the registry
        assert registry.should_process(
            'non_existent_doc',
            content=sample_document['content']
        )
        
        # Can force reprocessing with force_reprocess=True
        assert registry.should_process(
            sample_document['doc_id'],
            content=sample_document['content'],
            force_reprocess=True
        )
    
    def test_get_documents_by_status(self, registry):
        """Test retrieving documents by status."""
        # Register documents with different statuses
        registry.register_document(doc_id='doc1', content='Content 1')
        registry.register_document(doc_id='doc2', content='Content 2')
        registry.register_document(doc_id='doc3', content='Content 3')
        
        registry.update_status('doc2', ProcessingStatus.PROCESSING)
        registry.update_status('doc3', ProcessingStatus.PROCESSING)
        registry.update_status('doc3', ProcessingStatus.COMPLETED)
        
        # Retrieve documents by status
        pending_docs = registry.get_documents_by_status(ProcessingStatus.PENDING)
        processing_docs = registry.get_documents_by_status(ProcessingStatus.PROCESSING)
        completed_docs = registry.get_documents_by_status(ProcessingStatus.COMPLETED)
        
        assert len(pending_docs) == 1
        assert pending_docs[0]['doc_id'] == 'doc1'
        
        assert len(processing_docs) == 1
        assert processing_docs[0]['doc_id'] == 'doc2'
        
        assert len(completed_docs) == 1
        assert completed_docs[0]['doc_id'] == 'doc3'
    
    def test_reset_stalled_processing(self, registry):
        """Test resetting documents stuck in processing state."""
        # Register documents
        registry.register_document(doc_id='doc1', content='Content 1')
        registry.register_document(doc_id='doc2', content='Content 2')
        
        # Set to PROCESSING state
        registry.update_status('doc1', ProcessingStatus.PROCESSING)
        registry.update_status('doc2', ProcessingStatus.PROCESSING)
        
        # Store the current time for reference
        current_real_time = time.time()
        
        # Mock the time so we can control the timestamp
        with patch('time.time') as mock_time:
            # Set current time to be 2 hours later - use a concrete numeric value
            mock_time.return_value = current_real_time + 7200  # 2 hours in seconds
            
            # Reset stalled documents (default timeout is 1 hour)
            reset_count = registry.reset_stalled_processing()
            
            assert reset_count == 2  # Should reset both documents
            
            # Verify both docs are now FAILED
            doc1 = registry.get_document_status('doc1')
            doc2 = registry.get_document_status('doc2')
            
            assert doc1['status'] == ProcessingStatus.FAILED.value
            assert doc2['status'] == ProcessingStatus.FAILED.value
            assert "timeout" in doc1['error_message'].lower()
    
    def test_delete_document(self, registry, sample_document):
        """Test deleting a document from the registry."""
        # Register a document
        registry.register_document(
            doc_id=sample_document['doc_id'],
            content=sample_document['content']
        )
        
        # Verify it exists
        assert registry.get_document_status(sample_document['doc_id']) is not None
        
        # Delete it
        result = registry.delete_document(sample_document['doc_id'])
        assert result is True
        
        # Verify it no longer exists
        assert registry.get_document_status(sample_document['doc_id']) is None
        
        # Attempt to delete non-existent document
        result = registry.delete_document('non_existent_doc')
        assert result is False
    
    def test_get_processing_stats(self, registry):
        """Test getting processing statistics."""
        # Register documents with different statuses
        registry.register_document(doc_id='doc1', content='Content 1')
        registry.register_document(doc_id='doc2', content='Content 2')
        registry.register_document(doc_id='doc3', content='Content 3')
        registry.register_document(doc_id='doc4', content='Content 4')
        
        registry.update_status('doc2', ProcessingStatus.PROCESSING)
        registry.update_status('doc3', ProcessingStatus.COMPLETED)
        registry.update_status('doc4', ProcessingStatus.FAILED, error_message="Test error")
        
        # Get stats
        stats = registry.get_processing_stats()
        
        assert stats[ProcessingStatus.PENDING.value] == 1
        assert stats[ProcessingStatus.PROCESSING.value] == 1
        assert stats[ProcessingStatus.COMPLETED.value] == 1
        assert stats[ProcessingStatus.FAILED.value] == 1
    
    def test_metadata_integrity(self, registry, sample_document):
        """Test that metadata is properly stored and retrieved."""
        # Define metadata with all required fields
        metadata = {
            'document_type': 'document',
            'language': 'english',
            'processor': 'TestProcessor',
            'vector_status': 'pending',
            'source_uri': 'test://source/uri',
            'ingestion_timestamp': time.time(),
            'chunk_count': 0
        }
        
        # Register document with metadata
        registry.register_document(
            doc_id=sample_document['doc_id'],
            content=sample_document['content'],
            metadata=metadata
        )
        
        # Retrieve and verify metadata
        status = registry.get_document_status(sample_document['doc_id'])
        
        for key, value in metadata.items():
            assert key in status['metadata']
            assert status['metadata'][key] == value
