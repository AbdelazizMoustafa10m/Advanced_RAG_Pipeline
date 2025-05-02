#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vector Store Example

This example demonstrates how to use the vector store module to index and query documents.
It shows how to:
1. Configure a vector store (ChromaDB or Qdrant)
2. Create an index from nodes
3. Persist and load the index
4. Create a query engine and perform queries

Usage:
    python examples/vector_store_example.py
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import LlamaIndex components
from llama_index.core.schema import TextNode
from llama_index.core import Document

# Import our components
from core.config import VectorStoreConfig, EmbedderConfig
from indexing.vector_store import VectorStoreFactory
from embedders.embedder_factory import EmbedderFactory

def create_sample_nodes(num_nodes: int = 5) -> List[TextNode]:
    """Create sample nodes for testing."""
    nodes = []
    for i in range(num_nodes):
        node = TextNode(
            text=f"This is sample node {i} with some content about artificial intelligence and vector databases.",
            metadata={
                "source": f"sample_document_{i}.txt",
                "page": i,
                "formatted_source": f"Sample Document {i}",
                "formatted_location": f"Page {i}",
                "formatted_headings": "Introduction",
                "formatted_label": f"Sample Node {i}"
            },
            id_=f"node_{i}"
        )
        nodes.append(node)
    return nodes

def run_chroma_example():
    """Run an example with ChromaDB."""
    logger.info("Running ChromaDB example")
    
    # Configure vector store
    vector_store_config = VectorStoreConfig(
        engine="chroma",
        collection_name="sample_collection",
        vector_db_path="./example_vector_db"
    )
    
    # Create vector store
    try:
        vector_store = VectorStoreFactory.create_vector_store(vector_store_config)
        logger.info("Created ChromaDB vector store")
        
        # Create sample nodes
        nodes = create_sample_nodes(10)
        logger.info(f"Created {len(nodes)} sample nodes")
        
        # Configure embedder
        embedder_config = EmbedderConfig(
            provider="huggingface",
            model_name="BAAI/bge-small-en-v1.5",
            use_cache=True
        )
        
        # Create embedder
        embedder = EmbedderFactory.create_embedder(embedder_config)
        logger.info(f"Created embedder with provider: {embedder_config.provider}")
        
        # Embed nodes
        embedded_nodes = embedder.embed_nodes(nodes)
        logger.info(f"Embedded {len(embedded_nodes)} nodes")
        
        # Create index
        index = vector_store.create_index(embedded_nodes)
        logger.info("Created vector store index")
        
        # Persist index
        vector_store.persist()
        logger.info("Persisted index")
        
        # Create query engine
        query_engine = vector_store.get_query_engine(similarity_top_k=2)
        logger.info("Created query engine")
        
        # Perform query
        query = "Tell me about artificial intelligence"
        response = query_engine.query(query)
        logger.info(f"Query: {query}")
        logger.info(f"Response: {response}")
        
        # Load index
        loaded_vector_store = VectorStoreFactory.create_vector_store(vector_store_config)
        loaded_index = loaded_vector_store.load()
        logger.info("Loaded index from disk")
        
        # Create query engine from loaded index
        loaded_query_engine = loaded_vector_store.get_query_engine(similarity_top_k=2)
        
        # Perform query on loaded index
        loaded_response = loaded_query_engine.query(query)
        logger.info(f"Query on loaded index: {query}")
        logger.info(f"Response from loaded index: {loaded_response}")
        
        return True
    except Exception as e:
        logger.error(f"Error in ChromaDB example: {str(e)}")
        return False

def run_qdrant_example():
    """Run an example with Qdrant."""
    logger.info("Running Qdrant example")
    
    # Configure vector store
    vector_store_config = VectorStoreConfig(
        engine="qdrant",
        collection_name="sample_collection",
        qdrant_location="local",
        qdrant_local_path="./example_qdrant_db"
    )
    
    # Create vector store
    try:
        vector_store = VectorStoreFactory.create_vector_store(vector_store_config)
        logger.info("Created Qdrant vector store")
        
        # Create sample nodes
        nodes = create_sample_nodes(10)
        logger.info(f"Created {len(nodes)} sample nodes")
        
        # Configure embedder
        embedder_config = EmbedderConfig(
            provider="huggingface",
            model_name="BAAI/bge-small-en-v1.5",
            use_cache=True
        )
        
        # Create embedder
        embedder = EmbedderFactory.create_embedder(embedder_config)
        logger.info(f"Created embedder with provider: {embedder_config.provider}")
        
        # Embed nodes
        embedded_nodes = embedder.embed_nodes(nodes)
        logger.info(f"Embedded {len(embedded_nodes)} nodes")
        
        # Create index
        index = vector_store.create_index(embedded_nodes)
        logger.info("Created vector store index")
        
        # Persist index
        vector_store.persist()
        logger.info("Persisted index")
        
        # Create query engine
        query_engine = vector_store.get_query_engine(
            similarity_top_k=2,
            vector_store_query_mode="hybrid",  # Use hybrid search for Qdrant
            sparse_top_k=5
        )
        logger.info("Created query engine with hybrid search")
        
        # Perform query
        query = "Tell me about artificial intelligence"
        response = query_engine.query(query)
        logger.info(f"Query: {query}")
        logger.info(f"Response: {response}")
        
        # Load index
        loaded_vector_store = VectorStoreFactory.create_vector_store(vector_store_config)
        loaded_index = loaded_vector_store.load()
        logger.info("Loaded index from disk")
        
        # Create query engine from loaded index
        loaded_query_engine = loaded_vector_store.get_query_engine(
            similarity_top_k=2,
            vector_store_query_mode="hybrid",
            sparse_top_k=5
        )
        
        # Perform query on loaded index
        loaded_response = loaded_query_engine.query(query)
        logger.info(f"Query on loaded index: {query}")
        logger.info(f"Response from loaded index: {loaded_response}")
        
        return True
    except Exception as e:
        logger.error(f"Error in Qdrant example: {str(e)}")
        return False

def main():
    """Run the vector store examples."""
    logger.info("Starting vector store examples")
    
    # Get available engines
    available_engines = VectorStoreFactory.get_available_engines()
    logger.info(f"Available vector store engines: {available_engines}")
    
    # Run examples for available engines
    if "chroma" in available_engines:
        chroma_success = run_chroma_example()
        logger.info(f"ChromaDB example {'succeeded' if chroma_success else 'failed'}")
    else:
        logger.warning("ChromaDB not available, skipping example")
    
    if "qdrant" in available_engines:
        qdrant_success = run_qdrant_example()
        logger.info(f"Qdrant example {'succeeded' if qdrant_success else 'failed'}")
    else:
        logger.warning("Qdrant not available, skipping example")
    
    logger.info("Vector store examples completed")

if __name__ == "__main__":
    main()
