#!/usr/bin/env python3

"""
Example script demonstrating the use of the embedder module in the Advanced RAG Pipeline.

This script shows how to:
1. Configure the embedder with different providers
2. Embed a list of nodes
3. Embed a query string
4. Use the embedded nodes with a vector store
"""

import os
import logging
import sys
from pathlib import Path

# Add the project root to the path to ensure imports work correctly
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Import necessary modules
from llama_index.core.schema import TextNode, Document
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from core.config import EmbedderConfig, UnifiedConfig
from embedders.embedder_factory import EmbedderFactory
from embedders.llamaindex_embedder_service import LlamaIndexEmbedderService


def create_sample_nodes():
    """Create a list of sample nodes for embedding."""
    nodes = []
    
    # Create some sample nodes with different content types
    python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
    
# Calculate the 10th Fibonacci number
result = fibonacci(10)
print(f"The 10th Fibonacci number is {result}")
"""
    
    markdown_text = """
# Advanced RAG Pipeline

## Overview
The Advanced RAG Pipeline is a robust framework for processing diverse document types into semantically meaningful chunks suitable for Retrieval-Augmented Generation applications.

## Key Features
- Process multiple document types with type-specific chunking strategies
- Identify document types accurately using multiple detection methods
- Preserve semantic structure during chunking process
"""
    
    technical_doc = """
The embedding module in the Advanced RAG Pipeline leverages LlamaIndex's embedding capabilities to create vector representations of text nodes. It supports various embedding providers including HuggingFace, OpenAI, Cohere, and more.

The module is designed to be configurable and extensible, allowing users to choose different embedding models and providers based on their specific needs.
"""
    
    # Create nodes with appropriate metadata
    nodes.append(TextNode(
        text=python_code,
        metadata={"node_type": "code", "language": "python", "file_name": "fibonacci.py"}
    ))
    
    nodes.append(TextNode(
        text=markdown_text,
        metadata={"node_type": "document", "format": "markdown", "file_name": "README.md"}
    ))
    
    nodes.append(TextNode(
        text=technical_doc,
        metadata={"node_type": "document", "format": "text", "file_name": "embedding_module.txt"}
    ))
    
    return nodes


def main():
    """Main function demonstrating the embedder module."""
    logger.info("Starting embedder module example...")
    
    # Create sample nodes
    nodes = create_sample_nodes()
    logger.info(f"Created {len(nodes)} sample nodes")
    
    # Choose which embedding provider to use
    # Uncomment the configuration you want to use
    
    # Example 1: Using HuggingFace embeddings (default)
    #embedder_config = EmbedderConfig(
    #    provider="huggingface",
    #    model_name="BAAI/bge-small-en-v1.5",
    #    embed_batch_size=10,
    #    use_cache=True,
    #    cache_dir="./.cache/embeddings"
    #)
    
    # Example 2: Using Ollama embeddings (requires Ollama running locally)
    # Make sure to pull the model first with: ollama pull nomic-embed-text
    embedder_config = EmbedderConfig(
        provider="ollama",
        model_name="nomic-embed-text",  # or any other Ollama embedding model
        api_base="http://localhost:11434",  # Default Ollama server URL
        embed_batch_size=10,
        use_cache=True,
        cache_dir="./.cache/embeddings"
    )
    
    # Create the embedder service
    try:
        embedder = EmbedderFactory.create_embedder(embedder_config)
        logger.info(f"Created embedder with provider: {embedder_config.provider}, model: {embedder_config.model_name}")
        
        # Embed the nodes
        logger.info("Embedding nodes...")
        embedded_nodes = embedder.embed_nodes(nodes)
        
        # Check if nodes have embeddings
        nodes_with_embeddings = sum(1 for node in embedded_nodes if node.embedding is not None and len(node.embedding) > 0)
        logger.info(f"Nodes with embeddings: {nodes_with_embeddings}/{len(embedded_nodes)}")
        
        # Print embedding dimension for the first node
        if embedded_nodes[0].embedding is not None:
            logger.info(f"Embedding dimension: {len(embedded_nodes[0].embedding)}")
        
        # Embed a query
        query = "How does the embedding module work?"
        logger.info(f"Embedding query: '{query}'")
        query_embedding = embedder.embed_query(query)
        logger.info(f"Query embedding dimension: {len(query_embedding)}")
        
        # Create a vector store and index the embedded nodes
        logger.info("Creating vector store and indexing nodes...")
        
        # Initialize ChromaDB
        db_path = "./.chroma_example"
        os.makedirs(db_path, exist_ok=True)
        
        # Create a unique collection name based on the embedding model
        # This ensures we don't mix embeddings with different dimensions
        collection_name = f"collection_{embedder_config.provider}_{embedder_config.model_name.replace('/', '_')}"
        
        chroma_client = chromadb.PersistentClient(path=db_path)
        
        # Try to get the collection, and if it exists with wrong dimensions, delete it
        try:
            chroma_collection = chroma_client.get_or_create_collection(collection_name)
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning(f"Collection exists with wrong dimensions. Recreating: {e}")
                # Delete the collection if it exists with wrong dimensions
                try:
                    chroma_client.delete_collection(collection_name)
                except Exception as delete_error:
                    logger.error(f"Error deleting collection: {delete_error}")
                # Create a new collection
                chroma_collection = chroma_client.create_collection(collection_name)
        
        # Create vector store and index
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index from embedded nodes
        index = VectorStoreIndex(
            embedded_nodes,
            storage_context=storage_context,
            embed_model=embedder.get_embedding_model()  # Use the same embedding model
        )
        
        # Perform a query
        logger.info("Performing a query...")
        query_engine = index.as_query_engine()
        response = query_engine.query(query)
        
        logger.info(f"Query response: {response}")
        
        logger.info("Example completed successfully!")
        
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        logger.error("Please install the required dependencies with: pip install -r requirements.txt")
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")


if __name__ == "__main__":
    main()
