# --- core/interfaces.py ---

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Sequence
from pathlib import Path
from enum import Enum

# Import from llama_index namespace with corrected paths
from llama_index.core import Document
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM

from .config import DocumentType


class IDocumentDetector(ABC):
    """Interface for document type detection."""
    
    @abstractmethod
    def detect_type(self, file_path: str) -> DocumentType:
        """Detect document type from file path or content.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Detected document type
        """
        pass


class IDocumentLoader(ABC):
    """Interface for document loading."""
    
    @abstractmethod
    def load_documents(self, source: str) -> List[Document]:
        """Load documents from a source (file or directory).
        
        Args:
            source: Path to file or directory
            
        Returns:
            List of loaded documents
        """
        pass
    
    @abstractmethod
    def supports_source(self, source: str) -> bool:
        """Check if the loader supports a given source.
        
        Args:
            source: Path to file or directory
            
        Returns:
            True if supported, False otherwise
        """
        pass


class IDocumentChunker(ABC):
    """Interface for document chunking."""
    
    @abstractmethod
    def chunk_document(self, document: Document) -> List[TextNode]:
        """Chunk a document into text nodes.
        
        Args:
            document: The document to chunk
            
        Returns:
            List of text nodes
        """
        pass


class IDocumentProcessor(ABC):
    """Interface for document processing."""
    
    @abstractmethod
    def process_document(self, document: Document) -> List[TextNode]:
        """Process a document into text nodes.
        
        Args:
            document: The document to process
            
        Returns:
            List of text nodes
        """
        pass
    
    @abstractmethod
    def process_documents(self, documents: List[Document]) -> List[TextNode]:
        """Process multiple documents into text nodes.
        
        Args:
            documents: The documents to process
            
        Returns:
            List of text nodes
        """
        pass
    
    @abstractmethod
    def supports_document_type(self, document_type: DocumentType) -> bool:
        """Check if the processor supports a given document type.
        
        Args:
            document_type: The document type to check
            
        Returns:
            True if supported, False otherwise
        """
        pass


class IMetadataEnricher(ABC):
    """Interface for metadata enrichment."""
    
    @abstractmethod
    def enrich(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """Enrich nodes with metadata.
        
        Args:
            nodes: The nodes to enrich
            
        Returns:
            List of metadata dictionaries
        """
        pass
    
    @abstractmethod
    async def aenrich(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """Asynchronously enrich nodes with metadata.
        
        Args:
            nodes: The nodes to enrich
            
        Returns:
            List of metadata dictionaries
        """
        pass
    
    @abstractmethod
    def supports_node_type(self, node_type: str) -> bool:
        """Check if the enricher supports a given node type.
        
        Args:
            node_type: The node type to check (e.g., "text", "code")
            
        Returns:
            True if supported, False otherwise
        """
        pass


class IVectorStore(ABC):
    """Interface for vector storage."""
    
    @abstractmethod
    def create_index(self, nodes: List[TextNode]) -> VectorStoreIndex:
        """Create an index from nodes.
        
        Args:
            nodes: The nodes to index
            
        Returns:
            Vector store index
        """
        pass
    
    @abstractmethod
    def persist(self, path: Optional[str] = None) -> None:
        """Persist the index to storage.
        
        Args:
            path: Optional alternative path to store data
        """
        pass
    
    @abstractmethod
    def load(self, path: Optional[str] = None) -> VectorStoreIndex:
        """Load the index from storage.
        
        Args:
            path: Optional alternative path to load data from
            
        Returns:
            Loaded vector store index
        """
        pass


class IQueryProcessor(ABC):
    """Interface for query processing."""
    
    @abstractmethod
    def process_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Process a query and extract metadata filters.
        
        Args:
            query: The query string
            
        Returns:
            Tuple of (processed_query, filters_dict)
        """
        pass
    
    @abstractmethod
    def query(self, query: str, filters: Optional[Dict[str, Any]] = None):
        """Query the knowledge base.
        
        Args:
            query: The query string
            filters: Optional metadata filters
            
        Returns:
            Query response
        """
        pass


class ILLMProvider(ABC):
    """Interface for LLM providers."""
    
    @abstractmethod
    def get_metadata_llm(self) -> LLM:
        """Get LLM for metadata extraction.
        
        Returns:
            LLM instance
        """
        pass
    
    @abstractmethod
    def get_query_llm(self) -> LLM:
        """Get LLM for query processing.
        
        Returns:
            LLM instance
        """
        pass
