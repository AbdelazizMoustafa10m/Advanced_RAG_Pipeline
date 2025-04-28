# --- indexing/vector_store.py ---

from typing import List, Optional, Dict, Any
import logging
import os

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from core.interfaces import IVectorStore
from core.config import VectorStoreConfig

logger = logging.getLogger(__name__)


class ChromaVectorStoreAdapter(IVectorStore):
    """ChromaDB adapter for vector storage."""
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """Initialize ChromaDB vector store.
        
        Args:
            config: Optional vector store configuration
        """
        self.config = config or VectorStoreConfig()
        self.index = None
        
        # Create directory if it doesn't exist
        os.makedirs(self.config.vector_db_path, exist_ok=True)
        
        # Initialize ChromaDB
        try:
            self.db = chromadb.PersistentClient(path=self.config.vector_db_path)
            self.collection = self.db.get_or_create_collection(self.config.collection_name)
            self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            logger.info(f"Initialized ChromaDB at {self.config.vector_db_path} with collection {self.config.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise
    
    def create_index(self, nodes: List[TextNode]) -> VectorStoreIndex:
        """Create an index from nodes.
        
        Args:
            nodes: The nodes to index
            
        Returns:
            Vector store index
        """
        try:
            self.index = VectorStoreIndex(
                nodes,
                storage_context=self.storage_context
            )
            logger.info(f"Created index with {len(nodes)} nodes")
            return self.index
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise
    
    def persist(self, path: Optional[str] = None) -> None:
        """Persist the index to storage.
        
        Args:
            path: Optional alternative path to store data
        """
        # ChromaDB auto-persists, so this is a no-op for now
        # Could implement additional persistence logic if needed
        logger.info(f"Index persisted to {path or self.config.vector_db_path}")
    
    def load(self, path: Optional[str] = None) -> VectorStoreIndex:
        """Load the index from storage.
        
        Args:
            path: Optional alternative path to load data from
            
        Returns:
            Loaded vector store index
        """
        db_path = path or self.config.vector_db_path
        
        try:
            # Load existing ChromaDB
            db = chromadb.PersistentClient(path=db_path)
            collection = db.get_collection(self.config.collection_name)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Recreate index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=storage_context
            )
            logger.info(f"Loaded index from {db_path}")
            return self.index
        except Exception as e:
            logger.error(f"Error loading index from {db_path}: {str(e)}")
            raise
