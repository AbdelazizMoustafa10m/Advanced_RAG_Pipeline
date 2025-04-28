#!/usr/bin/env python3

from __future__ import absolute_import

"""
Script to fix document types in the document registry database.
This updates existing documents with proper type classification.
"""

import sqlite3
import os
import sys
# Import standard logging directly to avoid conflicts with any local logging module
import logging as python_logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import registry components
from registry.document_registry import DocumentRegistry
from registry.status import ProcessingStatus

# Set up logging
python_logging.basicConfig(
    level=python_logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = python_logging.getLogger(__name__)

def fix_document_types(db_path="./doc_reg.db"):
    """Update document types in the registry database based on file extensions."""
    logger.info(f"Fixing document types in registry database at {db_path}")
    
    # Create registry connection
    registry = DocumentRegistry(db_path=db_path)
    
    # First, get current stats
    before_stats = registry.get_processing_stats()
    logger.info(f"Before fixing - Document stats: {before_stats}")
    
    # Get all documents
    all_docs = registry.list_all_documents()
    logger.info(f"Found {len(all_docs)} documents in registry")
    
    # Update document types based on file extensions
    updated_count = 0
    
    with sqlite3.connect(db_path) as conn:
        for doc in all_docs:
            doc_id = doc['doc_id']
            old_type = None
            new_type = None
            
            # Get current document type
            cursor = conn.execute(
                "SELECT document_type FROM documents WHERE doc_id = ?", 
                (doc_id,)
            )
            result = cursor.fetchone()
            if result:
                old_type = result[0]
            
            # Determine document type by extension or path inspection
            if doc_id.lower().endswith('.py') or '/code/' in doc_id.lower():
                new_type = 'code'
            elif doc_id.lower().endswith(('.pdf', '.docx', '.txt', '.md')):
                new_type = 'document'
            else:
                # Default to document if we can't determine
                new_type = 'document'
            
            # Only update if type changed
            if old_type != new_type:
                # Update document type in the database
                conn.execute(
                    "UPDATE documents SET document_type = ? WHERE doc_id = ?",
                    (new_type, doc_id)
                )
                # Also update metadata to include document_type
                cursor = conn.execute("SELECT metadata FROM documents WHERE doc_id = ?", (doc_id,))
                metadata_row = cursor.fetchone()
                if metadata_row and metadata_row[0]:
                    # For SQLite JSON functions
                    try:
                        conn.execute(
                            "UPDATE documents SET metadata = json_patch(metadata, ?) WHERE doc_id = ?",
                            (f'{{"document_type": "{new_type}", "node_type": "{new_type}"}}', doc_id)
                        )
                    except sqlite3.OperationalError:
                        # Fallback if json_patch not available
                        logger.warning(f"SQLite JSON functions not available, using direct metadata update")
                        conn.execute(
                            "UPDATE documents SET metadata = ? WHERE doc_id = ?",
                            (f'{{"document_type": "{new_type}", "node_type": "{new_type}"}}', doc_id)
                        )
                
                updated_count += 1
                logger.info(f"Updated document '{doc_id}' from '{old_type}' to '{new_type}'")
    
    # Get updated stats
    after_stats = registry.get_processing_stats()
    logger.info(f"After fixing - Document stats: {after_stats}")
    logger.info(f"Updated {updated_count} document type classifications")
    
    return updated_count

if __name__ == "__main__":
    db_path = "./doc_reg.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    fix_document_types(db_path)
