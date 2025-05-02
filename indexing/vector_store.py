# --- indexing/vector_store.py ---

from typing import List, Optional, Dict, Any, Union, Type
import logging
import os
import time
from copy import deepcopy

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode, MetadataMode

# Local imports
from core.interfaces import IVectorStore
from core.config import VectorStoreConfig

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SIMILARITY_TOP_K = 5

# Import vector store implementations with graceful fallbacks
# ChromaDB imports
try:
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore as LlamaChromaStore
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB not available. Install with 'pip install chromadb'")

# Qdrant imports
try:
    import qdrant_client
    from qdrant_client.http import models as rest
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant not available. Install with 'pip install qdrant-client'")

# SimpleVectorStore is part of llama_index.core, so it should always be available
try:
    from llama_index.core.vector_stores import SimpleVectorStore
    SIMPLE_AVAILABLE = True
except ImportError:
    SIMPLE_AVAILABLE = False
    logger.warning("SimpleVectorStore not available. This is unexpected as it should be part of llama_index.core")


# Utility functions to avoid code duplication
def prepare_nodes_for_indexing(nodes: List[TextNode]) -> List[TextNode]:
    """
    Prepare nodes for indexing by filtering metadata to include only embedding-relevant fields.
    
    This function creates a copy of each node with only the metadata fields that should be included
    in the embedding process, ensuring consistent metadata handling across different vector stores.
    
    Args:
        nodes: List of nodes to prepare for indexing
        
    Returns:
        List of prepared nodes with filtered metadata
    """
    filtered_nodes = []
    
    for node in nodes:
        # Get the text that was used for embedding
        embed_text = node.get_content(metadata_mode=MetadataMode.EMBED)
        
        # Get only the metadata that should be included in embeddings
        embed_metadata = {}
        if hasattr(node, 'excluded_embed_metadata_keys') and node.excluded_embed_metadata_keys:
            # Copy only the metadata fields that aren't excluded for embedding
            for key, value in node.metadata.items():
                if key not in node.excluded_embed_metadata_keys:
                    embed_metadata[key] = value
        else:
            # If no exclusion list, use formatted metadata fields
            for key in ['formatted_source', 'formatted_location', 'formatted_headings', 
                       'formatted_label', 'formatted_metadata']:
                if key in node.metadata:
                    embed_metadata[key] = node.metadata[key]
        
        # Create a new node with only the embedding-relevant content and metadata
        filtered_node = TextNode(
            text=embed_text,
            metadata=embed_metadata,
            embedding=node.embedding,
            id_=node.node_id,
            relationships=deepcopy(node.relationships) if hasattr(node, 'relationships') else {}
        )
        
        filtered_nodes.append(filtered_node)
    
    return filtered_nodes


def sanitize_node_relationships(nodes: List[TextNode]) -> List[TextNode]:
    """
    Sanitize node relationships to ensure they are properly formatted for vector storage.
    
    This fixes issues where relationships might be stored as strings instead of proper node objects.
    
    Args:
        nodes: The nodes to sanitize
        
    Returns:
        List of nodes with sanitized relationships
    """
    from llama_index.core.schema import RelatedNodeInfo
    
    for node in nodes:
        # Create a copy of relationships to avoid modifying during iteration
        relationships_copy = dict(node.relationships)
        
        for rel_type, rel_value in relationships_copy.items():
            # Handle string relationships
            if isinstance(rel_value, str):
                # Create a proper RelatedNodeInfo object
                node.relationships[rel_type] = RelatedNodeInfo(
                    node_id=rel_value,
                    metadata={}
                )
            # Handle list of strings
            elif isinstance(rel_value, list) and rel_value and isinstance(rel_value[0], str):
                # Convert each string to a RelatedNodeInfo object
                node.relationships[rel_type] = [
                    RelatedNodeInfo(node_id=item, metadata={}) 
                    for item in rel_value
                ]
    
    return nodes


class BaseVectorStore(IVectorStore):
    """Base class for vector store implementations.
    
    This abstract base class provides common functionality for vector store implementations,
    reducing code duplication and ensuring consistent behavior across different adapters.
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """Initialize the base vector store.
        
        Args:
            config: Optional vector store configuration
        """
        self.config = config or VectorStoreConfig()
        self.index = None
        self.vector_store = None
        self.storage_context = None
    
    def get_storage_context(self) -> StorageContext:
        """Get the storage context for this vector store.
        
        Returns:
            StorageContext: The storage context for this vector store
        """
        if not self.storage_context:
            raise ValueError("Storage context not initialized")
        return self.storage_context
    
    def create_index(self, nodes: List[TextNode]) -> VectorStoreIndex:
        """Create an index from nodes.
        
        This base implementation handles common preprocessing steps like sanitizing relationships
        and filtering metadata for consistency across vector stores.
        
        Args:
            nodes: The nodes to index
            
        Returns:
            Vector store index
        """
        try:
            start_time = time.time()
            
            # Sanitize node relationships
            sanitized_nodes = sanitize_node_relationships(nodes)
            
            # Prepare nodes for indexing with consistent metadata handling
            filtered_nodes = prepare_nodes_for_indexing(sanitized_nodes)
            
            # Create the index with the filtered nodes
            # This requires self.storage_context to be initialized by the subclass
            if not self.storage_context:
                raise ValueError("Storage context not initialized by the adapter")
                
            self.index = VectorStoreIndex(
                filtered_nodes,
                storage_context=self.storage_context,
                show_progress=True
            )
            
            end_time = time.time()
            logger.info(f"Created index with {len(nodes)} nodes in {end_time - start_time:.2f} seconds")
            return self.index
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise
    
    def get_query_engine(self, **kwargs):
        """Get a query engine for this index.
        
        Args:
            **kwargs: Additional arguments for the query engine
            
        Returns:
            Query engine for this index
        """
        if not self.index:
            raise ValueError("Index not created or loaded")
        
        similarity_top_k = kwargs.get("similarity_top_k", DEFAULT_SIMILARITY_TOP_K)
        
        return self.index.as_query_engine(
            similarity_top_k=similarity_top_k
        )
    
    def persist(self, path: Optional[str] = None) -> None:
        """Persist the index to storage.
        
        This is a base implementation that should be overridden by subclasses
        if they require specific persistence logic.
        
        Args:
            path: Optional alternative path to store data
        """
        logger.info(f"Base persistence method called with path: {path}")
    
    def load(self, path: Optional[str] = None) -> VectorStoreIndex:
        """Load the index from storage.
        
        This is a base implementation that should be overridden by subclasses.
        
        Args:
            path: Optional alternative path to load data from
            
        Returns:
            Loaded vector store index
        """
        raise NotImplementedError("Load method must be implemented by subclass")


class ChromaDBVectorStore(BaseVectorStore):
    """ChromaDB implementation for vector storage.
    
    This class provides a clean wrapper around LlamaIndex's ChromaVectorStore integration.
    It handles initialization, persistence, and loading of ChromaDB vector stores.
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """Initialize ChromaDB vector store.
        
        Args:
            config: Optional vector store configuration
        """
        super().__init__(config)
        
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is not available. Install with 'pip install chromadb'")
        
        # Create directory if it doesn't exist
        os.makedirs(self.config.vector_db_path, exist_ok=True)
        
        # Initialize ChromaDB
        try:
            start_time = time.time()
            
            self.db = chromadb.PersistentClient(path=self.config.vector_db_path)
            self.collection = self.db.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_metric}
            )
            
            # Initialize LlamaIndex vector store wrapper
            self.vector_store = LlamaChromaStore(chroma_collection=self.collection)
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            end_time = time.time()
            logger.info(f"Initialized ChromaDB at {self.config.vector_db_path} with collection {self.config.collection_name}")
            logger.info(f"ChromaDB initialization completed in {end_time - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise
    
    def get_query_engine(self, **kwargs):
        """Get a query engine for this index.
        
        Extends the base implementation to support ChromaDB-specific features.
        
        Args:
            **kwargs: Additional arguments for the query engine
            
        Returns:
            Query engine for this index
        """
        if not self.index:
            raise ValueError("Index not created or loaded")
        
        similarity_top_k = kwargs.get("similarity_top_k", DEFAULT_SIMILARITY_TOP_K)
        
        # Use standard query engine with ChromaDB-specific parameters
        return self.index.as_query_engine(
            similarity_top_k=similarity_top_k
        )
    
    def persist(self, path: Optional[str] = None) -> None:
        """Persist the index to storage.
        
        Args:
            path: Optional alternative path to store data
        """
        # ChromaDB auto-persists, so this is a no-op
        logger.info(f"ChromaDB index persisted to {path or self.config.vector_db_path}")
    
    def load(self, path: Optional[str] = None) -> VectorStoreIndex:
        """Load the index from storage.
        
        Args:
            path: Optional alternative path to load data from
            
        Returns:
            Loaded vector store index
        """
        db_path = path or self.config.vector_db_path
        
        try:
            start_time = time.time()
            
            # Load existing ChromaDB
            db = chromadb.PersistentClient(path=db_path)
            collection = db.get_collection(self.config.collection_name)
            
            # Initialize LlamaIndex vector store wrapper
            vector_store = LlamaChromaStore(chroma_collection=collection)
            self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Recreate index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store,
                storage_context=self.storage_context
            )
            
            end_time = time.time()
            logger.info(f"Loaded ChromaDB index from {db_path} in {end_time - start_time:.2f} seconds")
            return self.index
        except Exception as e:
            logger.error(f"Error loading ChromaDB index from {db_path}: {str(e)}")
            raise


class QdrantVectorStore(BaseVectorStore):
    """Qdrant implementation for vector storage.
    
    This class provides a clean wrapper around LlamaIndex's QdrantVectorStore integration.
    It handles initialization, persistence, and loading of Qdrant vector stores, with support
    for both local and cloud deployments.
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """Initialize Qdrant vector store.
        
        Args:
            config: Optional vector store configuration
        """
        super().__init__(config or VectorStoreConfig(engine="qdrant"))
        
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant is not available. Install with 'pip install qdrant-client'")
        
        # Initialize Qdrant client based on configuration
        try:
            start_time = time.time()
            
            if self.config.qdrant_location == "local":
                # Create directory if it doesn't exist
                os.makedirs(self.config.qdrant_local_path, exist_ok=True)
                
                # Initialize local Qdrant client
                self.client = qdrant_client.QdrantClient(
                    path=self.config.qdrant_local_path,
                    timeout=self.config.qdrant_timeout,
                    prefer_grpc=self.config.qdrant_prefer_grpc
                )
                logger.info(f"Initialized local Qdrant at {self.config.qdrant_local_path}")
            else:
                # Initialize cloud Qdrant client
                self.client = qdrant_client.QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=self.config.qdrant_api_key,
                    timeout=self.config.qdrant_timeout,
                    prefer_grpc=self.config.qdrant_prefer_grpc
                )
                logger.info(f"Initialized cloud Qdrant at {self.config.qdrant_url}")
            
            # Check if collection exists and create it if not
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.config.collection_name not in collection_names:
                logger.info(f"Creating new Qdrant collection: {self.config.collection_name}")
                # Collection will be created when the first vectors are inserted
                # We don't need to explicitly create it here
            else:
                logger.info(f"Using existing Qdrant collection: {self.config.collection_name}")
            
            # Initialize vector store
            from llama_index.vector_stores.qdrant import QdrantVectorStore as LlamaQdrantStore
            self.vector_store = LlamaQdrantStore(
                client=self.client,
                collection_name=self.config.collection_name,
                on_disk_payload=self.config.qdrant_on_disk_payload
            )
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            end_time = time.time()
            logger.info(f"Qdrant initialization completed in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error initializing Qdrant: {str(e)}")
            raise
    
    def get_query_engine(self, **kwargs):
        """Get a query engine for this index.
        
        Extends the base implementation to support Qdrant-specific features like hybrid search.
        
        Args:
            **kwargs: Additional arguments for the query engine
            
        Returns:
            Query engine for this index
        """
        if not self.index:
            raise ValueError("Index not created or loaded")
        
        similarity_top_k = kwargs.get("similarity_top_k", DEFAULT_SIMILARITY_TOP_K)
        vector_store_query_mode = kwargs.get("vector_store_query_mode", None)
        
        if vector_store_query_mode == "hybrid":
            # Use hybrid search if specified
            return self.index.as_query_engine(
                similarity_top_k=similarity_top_k,
                vector_store_query_mode="hybrid",
                sparse_top_k=kwargs.get("sparse_top_k", 10)
            )
        else:
            # Use standard search
            return super().get_query_engine(**kwargs)
    
    def persist(self, path: Optional[str] = None) -> None:
        """Persist the index to storage.
        
        Args:
            path: Optional alternative path to store data (not used for Qdrant)
        """
        # Qdrant auto-persists, so this is a no-op
        logger.info(f"Qdrant index persisted (auto-persistence)")
    
    def load(self, path: Optional[str] = None) -> VectorStoreIndex:
        """Load the index from storage.
        
        Args:
            path: Optional alternative path to load data from (not used for Qdrant)
            
        Returns:
            Loaded vector store index
        """
        try:
            start_time = time.time()
            
            # For Qdrant, we just need to reconnect to the existing collection
            if self.config.qdrant_location == "local" and path:
                # Use alternative path if provided
                self.client = qdrant_client.QdrantClient(
                    path=path,
                    timeout=self.config.qdrant_timeout,
                    prefer_grpc=self.config.qdrant_prefer_grpc
                )
            
            # Initialize vector store with existing collection
            from llama_index.vector_stores.qdrant import QdrantVectorStore as LlamaQdrantStore
            self.vector_store = LlamaQdrantStore(
                client=self.client,
                collection_name=self.config.collection_name,
                on_disk_payload=self.config.qdrant_on_disk_payload
            )
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Recreate index from vector store
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                storage_context=self.storage_context
            )
            
            end_time = time.time()
            logger.info(f"Loaded Qdrant index from collection {self.config.collection_name} in {end_time - start_time:.2f} seconds")
            return self.index
        except Exception as e:
            logger.error(f"Error loading Qdrant index: {str(e)}")
            raise


class SimpleVectorStoreAdapter(IVectorStore):
    """Simple in-memory vector store adapter that requires no dependencies.
    
    This class provides a lightweight in-memory vector store using LlamaIndex's
    built-in SimpleVectorStore. It serves as a reliable fallback option when
    other vector store dependencies are not available.
    """
    
    def __init__(self, config: VectorStoreConfig):
        """Initialize the simple vector store adapter.
        
        Args:
            config: Vector store configuration
        """
        self.config = config
        self.index = None
        self.vector_store = None
        self.storage_context = None
        
        # Initialize storage context with a simple vector store
        try:
            start_time = time.time()
            
            self.vector_store = SimpleVectorStore()
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            end_time = time.time()
            logger.info(f"Initialized SimpleVectorStore with collection {self.config.collection_name}")
            logger.info(f"SimpleVectorStore initialization completed in {end_time - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error initializing SimpleVectorStore: {str(e)}")
            raise
    
    def get_storage_context(self) -> StorageContext:
        """Get the storage context for this vector store.
        
        Returns:
            StorageContext: The storage context for this vector store
        """
        return self.storage_context
    
    def create_index(self, nodes: List[TextNode]) -> VectorStoreIndex:
        """Create an index from nodes.
        
        Args:
            nodes: The nodes to index
            
        Returns:
            Vector store index
        """
        try:
            start_time = time.time()
            
            # Apply our advanced node preparation functions
            sanitized_nodes = sanitize_node_relationships(nodes)
            filtered_nodes = prepare_nodes_for_indexing(sanitized_nodes)
            
            # Create the index with the filtered nodes
            self.index = VectorStoreIndex(
                filtered_nodes,
                storage_context=self.storage_context,
                show_progress=True
            )
            
            end_time = time.time()
            logger.info(f"Created SimpleVectorStore index with {len(nodes)} nodes in {end_time - start_time:.2f} seconds")
            return self.index
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise
    
    def persist(self, path: Optional[str] = None) -> None:
        """Persist the index to storage.
        
        SimpleVectorStore is in-memory only and does not support persistence.
        
        Args:
            path: Optional alternative path to store data
        """
        logger.info("SimpleVectorStore is in-memory only and does not support persistence")
    
    def load(self, path: Optional[str] = None) -> VectorStoreIndex:
        """Load the index from storage.
        
        SimpleVectorStore is in-memory only and does not support loading from storage.
        
        Args:
            path: Optional alternative path to load data from
            
        Returns:
            Loaded vector store index
        """
        raise ValueError("SimpleVectorStore is in-memory only and does not support loading from storage")
    
    def get_query_engine(self, **kwargs):
        """Get a query engine for this index.
        
        Args:
            **kwargs: Additional arguments for the query engine
            
        Returns:
            Query engine for this index
        """
        if not self.index:
            raise ValueError("Index not created")
        
        similarity_top_k = kwargs.get("similarity_top_k", DEFAULT_SIMILARITY_TOP_K)
        
        return self.index.as_query_engine(
            similarity_top_k=similarity_top_k
        )


class VectorStoreFactory:
    """Factory for creating vector store instances based on configuration.
    
    This factory provides a clean way to instantiate the appropriate vector store
    implementation based on the provided configuration, with graceful fallbacks
    when dependencies are not available.
    """
    
    @staticmethod
    def create_vector_store(config: VectorStoreConfig) -> IVectorStore:
        """Create a vector store instance based on configuration.
        
        Args:
            config: Vector store configuration
            
        Returns:
            Vector store instance with multi-level fallback chain
        """
        # Create the appropriate vector store based on the engine with a robust fallback chain
        try:
            if config.engine == "chroma":
                if not CHROMA_AVAILABLE:
                    logger.warning("ChromaDB not available. Attempting fallback options.")
                    # Try to use Qdrant as first fallback
                    if QDRANT_AVAILABLE:
                        logger.warning("Falling back to Qdrant.")
                        qdrant_config = VectorStoreConfig(
                            engine="qdrant",
                            collection_name=config.collection_name
                        )
                        return QdrantVectorStore(qdrant_config)
                    # Try SimpleVectorStore as last fallback
                    elif SIMPLE_AVAILABLE:
                        logger.warning("Falling back to SimpleVectorStore (in-memory).")
                        return SimpleVectorStoreAdapter(config)
                    else:
                        raise ImportError("No vector stores available.")
                return ChromaDBVectorStore(config)
            
            elif config.engine == "qdrant":
                if not QDRANT_AVAILABLE:
                    logger.warning("Qdrant not available. Attempting fallback options.")
                    # Try to use ChromaDB as first fallback
                    if CHROMA_AVAILABLE:
                        logger.warning("Falling back to ChromaDB.")
                        chroma_config = VectorStoreConfig(
                            engine="chroma",
                            collection_name=config.collection_name
                        )
                        return ChromaDBVectorStore(chroma_config)
                    # Try SimpleVectorStore as last fallback
                    elif SIMPLE_AVAILABLE:
                        logger.warning("Falling back to SimpleVectorStore (in-memory).")
                        return SimpleVectorStoreAdapter(config)
                    else:
                        raise ImportError("No vector stores available.")
                return QdrantVectorStore(config)
            
            elif config.engine == "simple":
                # Explicit request for simple in-memory vector store
                if SIMPLE_AVAILABLE:
                    logger.info("Using SimpleVectorStore as requested.")
                    return SimpleVectorStoreAdapter(config)
                else:
                    raise ImportError("SimpleVectorStore not available.")
            
            else:
                logger.warning(f"Unsupported vector store engine: {config.engine}. Falling back to available options.")
                # Try each vector store option in order of preference
                if CHROMA_AVAILABLE:
                    logger.warning("Falling back to ChromaDB.")
                    return ChromaDBVectorStore(VectorStoreConfig(collection_name=config.collection_name))
                elif QDRANT_AVAILABLE:
                    logger.warning("Falling back to Qdrant.")
                    return QdrantVectorStore(VectorStoreConfig(collection_name=config.collection_name))
                elif SIMPLE_AVAILABLE:
                    logger.warning("Falling back to SimpleVectorStore (in-memory).")
                    return SimpleVectorStoreAdapter(config)
                else:
                    raise ImportError("No vector stores available.")
                    
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            # Final fallback attempt - try SimpleVectorStore if all else failed
            if SIMPLE_AVAILABLE:
                try:
                    logger.warning("Attempting last-resort fallback to SimpleVectorStore due to error.")
                    return SimpleVectorStoreAdapter(config)
                except Exception as inner_e:
                    logger.error(f"Failed to initialize fallback SimpleVectorStore: {str(inner_e)}")
            raise ValueError(f"Failed to create vector store: {str(e)}. Please check dependencies and configuration.")
            
    @staticmethod
    def get_available_engines() -> List[str]:
        """Get a list of available vector store engines.
        
        Returns:
            List of available engine names
        """
        available_engines = []
        
        if CHROMA_AVAILABLE:
            available_engines.append("chroma")
        
        if QDRANT_AVAILABLE:
            available_engines.append("qdrant")
        
        if SIMPLE_AVAILABLE:
            available_engines.append("simple")
        
        return available_engines


# Backward compatibility aliases
ChromaVectorStoreAdapter = ChromaDBVectorStore
QdrantVectorStoreAdapter = QdrantVectorStore
