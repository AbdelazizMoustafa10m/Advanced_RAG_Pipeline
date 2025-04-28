# --- registry/document_registry.py ---

import sqlite3
import hashlib
import json
import time
import os
from pathlib import Path
from enum import Enum
from typing import Dict, Optional, List, Any, Union

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentRegistry:
    def __init__(self, db_path: str = "./document_registry.db"):
        """Initialize the document registry with SQLite database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._initialize_db()
    
    def _initialize_db(self):
        """Create the database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    source TEXT,
                    status TEXT NOT NULL,
                    document_type TEXT,
                    last_processed REAL,
                    error_message TEXT,
                    metadata TEXT
                )
            ''')
            # Create indexes for fast lookups
            conn.execute('CREATE INDEX IF NOT EXISTS idx_content_hash ON documents(content_hash)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON documents(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_document_type ON documents(document_type)')
            conn.commit()
    
    def compute_hash(self, content: Union[str, bytes]) -> str:
        """Compute a hash of the document content.
        
        Args:
            content: Document content as string or bytes
            
        Returns:
            SHA-256 hash of the content
        """
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()
    
    def register_document(self, 
                         doc_id: str, 
                         content: Optional[Union[str, bytes]] = None, 
                         content_hash: Optional[str] = None, 
                         source: Optional[str] = None,
                         document_type: Optional[str] = None, 
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Register a document in the registry.
        
        Args:
            doc_id: Unique document identifier (e.g., file path, database ID)
            content: Document content (optional if content_hash is provided)
            content_hash: Pre-computed content hash (optional if content is provided)
            source: Source information if different from doc_id
            document_type: Type of document (e.g., "code", "document")
            metadata: Additional metadata to store with the document
            
        Returns:
            doc_id of the registered document
            
        Raises:
            ValueError: If neither content nor content_hash is provided
        """
        if content is None and content_hash is None:
            raise ValueError("Either content or content_hash must be provided")
        
        if content_hash is None:
            content_hash = self.compute_hash(content)
            
        with sqlite3.connect(self.db_path) as conn:
            # Check if document exists
            existing = conn.execute(
                'SELECT status, content_hash FROM documents WHERE doc_id = ?', 
                (doc_id,)
            ).fetchone()
            
            metadata_json = json.dumps(metadata) if metadata else None
            timestamp = time.time()
            
            if existing:
                status, existing_hash = existing
                
                # Update only if hash changed or not completed successfully
                if existing_hash != content_hash or status != ProcessingStatus.COMPLETED.value:
                    conn.execute(
                        '''UPDATE documents 
                           SET content_hash = ?, source = ?, document_type = ?,
                               status = ?, last_processed = ?,
                               metadata = ?, error_message = NULL
                           WHERE doc_id = ?''',
                        (content_hash, source, document_type, ProcessingStatus.PENDING.value, 
                         timestamp, metadata_json, doc_id)
                    )
            else:
                # Insert new document
                conn.execute(
                    '''INSERT INTO documents 
                       (doc_id, content_hash, source, document_type, status, 
                        last_processed, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (doc_id, content_hash, source, document_type, ProcessingStatus.PENDING.value, 
                     timestamp, metadata_json)
                )
            conn.commit()
        return doc_id
    
    def update_status(self, 
                     doc_id: str, 
                     status: ProcessingStatus, 
                     error_message: Optional[str] = None,
                     metadata_updates: Optional[Dict[str, Any]] = None) -> bool:
        """Update the processing status and optionally add error message or update metadata.
        
        Args:
            doc_id: Unique document identifier
            status: New processing status
            error_message: Error message (for FAILED status)
            metadata_updates: Additional metadata to update/add

        Returns:
            True if document status was updated, False if document not found
        """
        with sqlite3.connect(self.db_path) as conn:
            # First check if document exists and get current metadata
            result = conn.execute(
                'SELECT metadata FROM documents WHERE doc_id = ?',
                (doc_id,)
            ).fetchone()
            
            if not result:
                return False
                
            # Update metadata if needed
            metadata_json = result[0]
            if metadata_updates and metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                    metadata.update(metadata_updates)
                    metadata_json = json.dumps(metadata)
                except json.JSONDecodeError:
                    metadata_json = json.dumps(metadata_updates)
            elif metadata_updates:
                metadata_json = json.dumps(metadata_updates)
                
            # Update status and metadata
            conn.execute(
                '''UPDATE documents 
                   SET status = ?, last_processed = ?, error_message = ?,
                       metadata = ?
                   WHERE doc_id = ?''',
                (status.value, time.time(), error_message, metadata_json, doc_id)
            )
            conn.commit()
            return True
    
    def should_process(self, 
                      doc_id: str, 
                      content: Optional[Union[str, bytes]] = None, 
                      content_hash: Optional[str] = None,
                      force_reprocess: bool = False) -> bool:
        """Check if a document should be processed based on status and content hash.
        
        Args:
            doc_id: Unique document identifier
            content: Document content (optional if content_hash is provided)
            content_hash: Pre-computed content hash (optional if content is provided)
            force_reprocess: Force reprocessing regardless of status (except PROCESSING)
            
        Returns:
            True if document should be processed, False otherwise
            
        Raises:
            ValueError: If neither content nor content_hash is provided
        """
        if content is None and content_hash is None:
            raise ValueError("Either content or content_hash must be provided")
        
        if content_hash is None:
            content_hash = self.compute_hash(content)
            
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                '''SELECT content_hash, status FROM documents 
                   WHERE doc_id = ?''',
                (doc_id,)
            ).fetchone()
            
        if not result:
            # Document not in registry
            return True
            
        db_hash, status = result
        
        # Don't process if already in PROCESSING state (prevent concurrent processing)
        if status == ProcessingStatus.PROCESSING.value:
            return False
            
        # Process if hash changed, status is FAILED, or forced reprocess
        return (db_hash != content_hash or 
                status == ProcessingStatus.FAILED.value or
                (force_reprocess and status != ProcessingStatus.PROCESSING.value))
    
    def get_document_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status and metadata of a document.
        
        Args:
            doc_id: Unique document identifier
            
        Returns:
            Dictionary with document status and metadata, or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            result = conn.execute(
                '''SELECT status, document_type, last_processed, error_message, 
                   metadata, content_hash, source
                   FROM documents WHERE doc_id = ?''',
                (doc_id,)
            ).fetchone()
            
        if not result:
            return None
            
        result_dict = dict(result)
        
        # Parse metadata JSON
        if result_dict["metadata"]:
            try:
                result_dict["metadata"] = json.loads(result_dict["metadata"])
            except json.JSONDecodeError:
                result_dict["metadata"] = {}
        else:
            result_dict["metadata"] = {}
            
        result_dict["doc_id"] = doc_id
        return result_dict
            
    def get_documents_by_status(self, status: ProcessingStatus) -> List[Dict[str, Any]]:
        """Get all documents with a specific status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List of document records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            results = conn.execute(
                '''SELECT doc_id, content_hash, document_type, source, 
                   last_processed, metadata FROM documents WHERE status = ?''',
                (status.value,)
            ).fetchall()
            
        documents = []
        for row in results:
            doc = dict(row)
            if doc["metadata"]:
                try:
                    doc["metadata"] = json.loads(doc["metadata"])
                except json.JSONDecodeError:
                    doc["metadata"] = {}
            else:
                doc["metadata"] = {}
            documents.append(doc)
            
        return documents
    
    def get_documents_by_type(self, document_type: str) -> List[Dict[str, Any]]:
        """Get all documents of a specific type.
        
        Args:
            document_type: Document type to filter by
            
        Returns:
            List of document records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            results = conn.execute(
                '''SELECT doc_id, content_hash, status, source, 
                   last_processed, metadata FROM documents WHERE document_type = ?''',
                (document_type,)
            ).fetchall()
            
        documents = []
        for row in results:
            doc = dict(row)
            if doc["metadata"]:
                try:
                    doc["metadata"] = json.loads(doc["metadata"])
                except json.JSONDecodeError:
                    doc["metadata"] = {}
            else:
                doc["metadata"] = {}
            documents.append(doc)
            
        return documents
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics on document processing status.
        
        Returns:
            Dictionary with document processing statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get counts by status
            cursor = conn.execute(
                '''SELECT status, COUNT(*) FROM documents GROUP BY status'''
            )
            status_counts = {status: count for status, count in cursor.fetchall()}
            
            # Get total document count
            cursor = conn.execute('''SELECT COUNT(*) FROM documents''')
            total_documents = cursor.fetchone()[0]
            
            # Get counts by document type
            cursor = conn.execute(
                '''SELECT document_type, COUNT(*) FROM documents GROUP BY document_type'''
            )
            type_counts = {doc_type or 'unknown': count for doc_type, count in cursor.fetchall()}
            
            # Get error counts
            cursor = conn.execute(
                '''SELECT COUNT(*) FROM documents WHERE error_message IS NOT NULL'''
            )
            error_count = cursor.fetchone()[0]
            
            # Get most recent processing timestamp
            cursor = conn.execute(
                '''SELECT MAX(last_processed) FROM documents'''
            )
            last_processed = cursor.fetchone()[0]
            
            # Format the stats
            return {
                "total_documents": total_documents,
                "status_counts": status_counts,
                "document_types": type_counts,
                "error_count": error_count,
                "last_processed": last_processed,
                "database_path": self.db_path
            }
            
    def list_all_documents(self):
        """Return a list of all documents in the registry with their status."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT doc_id, status, last_processed FROM documents"
            )
            return [{
                "doc_id": row[0], 
                "status": row[1], 
                "last_processed": row[2]
            } for row in cursor.fetchall()]
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the registry.
        
        Args:
            doc_id: Unique document identifier
            
        Returns:
            True if document was deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('DELETE FROM documents WHERE doc_id = ?', (doc_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def reset_stalled_processing(self, max_processing_time: int = 3600) -> int:
        """Reset documents that have been in PROCESSING state for too long.
        
        Args:
            max_processing_time: Maximum processing time in seconds
            
        Returns:
            Number of reset documents
        """
        current_time = time.time()
        cutoff_time = current_time - max_processing_time
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                '''UPDATE documents
                   SET status = ?, error_message = ?
                   WHERE status = ? AND last_processed < ?''',
                (ProcessingStatus.FAILED.value, "Processing timeout",
                 ProcessingStatus.PROCESSING.value, cutoff_time)
            )
            conn.commit()
            return cursor.rowcount