# --- embedders/llamaindex_embedder_service.py ---

import logging
import os
import time
import json
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Type, Union

from llama_index.core.schema import TextNode, MetadataMode
from llama_index.core.embeddings import BaseEmbedding

from core.config import EmbedderConfig
from core.interfaces import IEmbedder

logger = logging.getLogger(__name__)

class LlamaIndexEmbedderService(IEmbedder):
    """Embedder service using LlamaIndex embedding models."""
    
    def __init__(self, config: Optional[EmbedderConfig] = None):
        """Initialize the embedder service with configuration.
        
        Args:
            config: Optional embedder configuration
        """
        self.config = config or EmbedderConfig()
        self.embed_model = None
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Initialize the embedding model
        self._initialize_embed_model()
        
        # Load cache if enabled
        if self.config.use_cache:
            self._load_cache()
    
    def _initialize_embed_model(self):
        """Initialize the appropriate LlamaIndex embedding model based on configuration."""
        provider = self.config.provider.lower()
        
        try:
            if provider == "huggingface":
                self._initialize_huggingface_embeddings()
            elif provider == "openai":
                self._initialize_openai_embeddings()
            elif provider == "cohere":
                self._initialize_cohere_embeddings()
            elif provider == "vertex":
                self._initialize_vertex_embeddings()
            elif provider == "bedrock":
                self._initialize_bedrock_embeddings()
            elif provider == "ollama":
                self._initialize_ollama_embeddings()
            else:
                logger.warning(f"Unknown provider: {provider}, falling back to huggingface")
                self._initialize_huggingface_embeddings()
                
            logger.info(f"Successfully initialized {provider} embedding model: {self.config.model_name}")
            
        except ImportError as e:
            logger.error(f"Missing dependency for {provider} embeddings: {str(e)}")
            logger.warning(f"Falling back to default huggingface embeddings")
            self._initialize_huggingface_embeddings()
        except Exception as e:
            logger.error(f"Error initializing {provider} embeddings: {str(e)}")
            logger.warning(f"Falling back to default huggingface embeddings")
            self._initialize_huggingface_embeddings()
    
    def _initialize_huggingface_embeddings(self):
        """Initialize HuggingFace embedding model."""
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            
            # Get model name from config
            model_name = self.config.model_name
            
            # Create embedding model with batch size from config
            self.embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                embed_batch_size=self.config.embed_batch_size,
                **self.config.additional_kwargs
            )
            
        except ImportError:
            logger.error("Missing dependency: llama-index-embeddings-huggingface")
            logger.error("Please install with: pip install llama-index-embeddings-huggingface")
            raise
    
    def _initialize_openai_embeddings(self):
        """Initialize OpenAI embedding model."""
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
            
            # Get API key from environment variable if specified
            api_key = os.getenv(self.config.api_key_env_var) if self.config.api_key_env_var else None
            
            # Create embedding model with batch size from config
            self.embed_model = OpenAIEmbedding(
                model=self.config.model_name,
                embed_batch_size=self.config.embed_batch_size,
                api_key=api_key,
                api_base=self.config.api_base,
                **self.config.additional_kwargs
            )
            
        except ImportError:
            logger.error("Missing dependency: llama-index-embeddings-openai")
            logger.error("Please install with: pip install llama-index-embeddings-openai")
            raise
    
    def _initialize_cohere_embeddings(self):
        """Initialize Cohere embedding model."""
        try:
            from llama_index.embeddings.cohere import CohereEmbedding
            
            # Get API key from environment variable if specified
            api_key = os.getenv(self.config.api_key_env_var) if self.config.api_key_env_var else None
            
            # Create embedding model with batch size from config
            self.embed_model = CohereEmbedding(
                model_name=self.config.model_name,
                api_key=api_key,
                embed_batch_size=self.config.embed_batch_size,
                **self.config.additional_kwargs
            )
            
        except ImportError:
            logger.error("Missing dependency: llama-index-embeddings-cohere")
            logger.error("Please install with: pip install llama-index-embeddings-cohere")
            raise
    
    def _initialize_vertex_embeddings(self):
        """Initialize Google Vertex AI embedding model."""
        try:
            from llama_index.embeddings.vertex import VertexEmbedding
            
            # Create embedding model with batch size from config
            self.embed_model = VertexEmbedding(
                model_name=self.config.model_name,
                embed_batch_size=self.config.embed_batch_size,
                **self.config.additional_kwargs
            )
            
        except ImportError:
            logger.error("Missing dependency: llama-index-embeddings-vertex")
            logger.error("Please install with: pip install llama-index-embeddings-vertex")
            raise
    
    def _initialize_bedrock_embeddings(self):
        """Initialize AWS Bedrock embedding model."""
        try:
            from llama_index.embeddings.bedrock import BedrockEmbedding
            
            # Create embedding model with batch size from config
            self.embed_model = BedrockEmbedding(
                model_name=self.config.model_name,
                embed_batch_size=self.config.embed_batch_size,
                **self.config.additional_kwargs
            )
            
        except ImportError:
            logger.error("Missing dependency: llama-index-embeddings-bedrock")
            logger.error("Please install with: pip install llama-index-embeddings-bedrock")
            raise
    
    def _initialize_ollama_embeddings(self):
        """Initialize Ollama embedding model."""
        try:
            from llama_index.embeddings.ollama import OllamaEmbedding
            
            # Get model name from config
            model_name = self.config.model_name
            
            # Get base URL from config or use default
            base_url = self.config.api_base or "http://localhost:11434"
            
            # Create embedding model with batch size from config
            self.embed_model = OllamaEmbedding(
                model_name=model_name,
                base_url=base_url,
                embed_batch_size=self.config.embed_batch_size,
                **self.config.additional_kwargs
            )
            
        except ImportError:
            logger.error("Missing dependency: llama-index-embeddings-ollama")
            logger.error("Please install with: pip install llama-index-embeddings-ollama")
            raise
    
    def embed_nodes(self, nodes: List[TextNode]) -> List[TextNode]:
        """Embed a list of nodes with the configured embedding model.
        
        Args:
            nodes: The nodes to embed
            
        Returns:
            List of nodes with embeddings
        """
        if not nodes:
            logger.warning("No nodes to embed")
            return []
        
        if not self.embed_model:
            logger.error("Embedding model not initialized")
            return nodes
        
        start_time = time.time()
        logger.info(f"Embedding {len(nodes)} nodes with {self.config.provider} provider")
        
        # Process nodes in batches for efficiency
        try:
            # Extract text content from nodes using MetadataMode.EMBED to respect embedding-specific formatting
            texts_to_embed = []
            nodes_to_embed = []
            cached_nodes = []
            
            for node in nodes:
                if node.embedding is None or len(node.embedding) == 0:
                    if self.config.use_cache:
                        cache_key = self._get_cache_key(node)
                        cached_embedding = self._get_from_cache(cache_key)
                        
                        if cached_embedding is not None:
                            node.embedding = cached_embedding
                            cached_nodes.append(node)
                            self._cache_hits += 1
                        else:
                            # Get text content for embedding
                            text = node.get_content(metadata_mode=MetadataMode.EMBED)
                            texts_to_embed.append(text)
                            nodes_to_embed.append(node)
                            self._cache_misses += 1
                    else:
                        # Get text content for embedding
                        text = node.get_content(metadata_mode=MetadataMode.EMBED)
                        texts_to_embed.append(text)
                        nodes_to_embed.append(node)
                else:
                    cached_nodes.append(node)
            
            if not nodes_to_embed:
                logger.info(f"All {len(cached_nodes)} nodes already have embeddings or were cached")
                return nodes
            
            logger.info(f"Embedding {len(nodes_to_embed)} nodes (batch size: {self.config.embed_batch_size})")
            
            # Generate embeddings in batch
            embeddings = self.embed_model.get_text_embedding_batch(texts_to_embed)
            
            # Assign embeddings to nodes
            for i, node in enumerate(nodes_to_embed):
                node.embedding = embeddings[i]
                
                # Update cache if enabled
                if self.config.use_cache:
                    cache_key = self._get_cache_key(node)
                    self._add_to_cache(cache_key, node.embedding)
            
            # Combine processed nodes with cached nodes
            all_nodes = nodes_to_embed + cached_nodes
            
            # Ensure the order is preserved
            node_map = {id(node): node for node in all_nodes}
            result = [node_map.get(id(original_node), original_node) for original_node in nodes]
            
            end_time = time.time()
            embedding_time = end_time - start_time
            nodes_per_second = len(nodes) / embedding_time if embedding_time > 0 else 0
            
            logger.info(f"Embedded {len(nodes_to_embed)} nodes in {embedding_time:.2f} seconds")
            logger.info(f"Embedding efficiency: {nodes_per_second:.2f} nodes/second")
            
            if self.config.use_cache:
                logger.info(f"Cache performance: {self._cache_hits} hits, {self._cache_misses} misses")
                self._save_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Error embedding nodes: {str(e)}")
            return nodes
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query string.
        
        Args:
            query: The query to embed
            
        Returns:
            Embedding vector
        """
        if not self.embed_model:
            logger.error("Embedding model not initialized")
            return []
        
        try:
            embedding = self.embed_model.get_query_embedding(query)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            return []
    
    def get_embedding_model(self) -> BaseEmbedding:
        """Get the underlying LlamaIndex embedding model.
        
        Returns:
            The LlamaIndex embedding model instance
        """
        return self.embed_model
    
    def _get_cache_key(self, node: TextNode) -> str:
        """Generate a cache key for a node.
        
        Args:
            node: The node to generate a cache key for
            
        Returns:
            Cache key
        """
        # Use node content and metadata for the cache key
        content = node.get_content(metadata_mode=MetadataMode.EMBED)
        metadata_str = json.dumps(node.metadata, sort_keys=True) if node.metadata else ""
        
        # Create a hash of the content and metadata
        key_data = f"{content}|{metadata_str}|{self.config.provider}|{self.config.model_name}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[List[float]]:
        """Get an embedding from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached embedding or None if not found
        """
        return self._cache.get(key)
    
    def _add_to_cache(self, key: str, embedding: List[float]) -> None:
        """Add an embedding to the cache.
        
        Args:
            key: Cache key
            embedding: Embedding to cache
        """
        self._cache[key] = embedding
    
    def _get_sanitized_model_name(self) -> str:
        """Get a sanitized version of the model name for use in filenames.
        
        Returns:
            Sanitized model name
        """
        # Replace characters that are not allowed in filenames
        return self.config.model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    def _load_cache(self) -> None:
        """Load the embedding cache from disk."""
        sanitized_model_name = self._get_sanitized_model_name()
        cache_path = os.path.join(self.config.cache_dir, f"{self.config.provider}_{sanitized_model_name}.cache")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    self._cache = pickle.load(f)
                logger.info(f"Loaded embedding cache with {len(self._cache)} entries from {cache_path}")
            except Exception as e:
                logger.error(f"Error loading embedding cache: {str(e)}")
                self._cache = {}
        else:
            logger.info(f"No embedding cache found at {cache_path}")
            self._cache = {}
    
    def _save_cache(self) -> None:
        """Save the embedding cache to disk."""
        if not self._cache:
            logger.info("No embeddings to cache")
            return
        
        os.makedirs(self.config.cache_dir, exist_ok=True)
        sanitized_model_name = self._get_sanitized_model_name()
        cache_path = os.path.join(self.config.cache_dir, f"{self.config.provider}_{sanitized_model_name}.cache")
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(self._cache, f)
            logger.info(f"Saved embedding cache with {len(self._cache)} entries to {cache_path}")
        except Exception as e:
            logger.error(f"Error saving embedding cache: {str(e)}")
            # Log additional information for debugging
            logger.debug(f"Cache directory: {self.config.cache_dir}")
            logger.debug(f"Provider: {self.config.provider}")
            logger.debug(f"Model name: {self.config.model_name}")
            logger.debug(f"Sanitized model name: {sanitized_model_name}")
