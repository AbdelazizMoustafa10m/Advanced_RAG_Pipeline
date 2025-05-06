#!/usr/bin/env python
# --- examples/embedder_query_example.py ---

import sys
import os
import logging
from pathlib import Path
import time

# Add the project root to the path to make imports work
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from core.config import QueryPipelineConfig, EmbedderConfig
from core.config import UnifiedConfig, VectorStoreConfig
from pipeline.orchestrator import PipelineOrchestrator
from llm.providers import DefaultLLMProvider
from indexing.vector_store import VectorStoreFactory
from query.query_pipeline import QueryPipeline
from embedders.embedder_factory import EmbedderFactory
from core.config import LLMConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("embedder-query-example")


def print_section_header(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def print_source_nodes(nodes, max_nodes=3):
    """Print source nodes in a readable format."""
    print(f"\nTop {min(len(nodes), max_nodes)} Source Nodes:")
    print("-" * 80)
    
    for i, node in enumerate(nodes[:max_nodes]):
        print(f"Source {i+1} (score: {node.score:.4f}):")
        print(f"  Source: {node.node.metadata.get('file_path', 'Unknown')}")
        
        # Format content (first 100 chars)
        content = node.node.get_content()
        if len(content) > 100:
            content = content[:97] + "..."
        
        print(f"  Content: {content}")
        print()


def run_query_with_explicit_embedder():
    """Run a query example with explicit embedder configuration."""
    print_section_header("Query with Explicit Embedder")
    
    # Create vector store configuration
    vector_store_config = VectorStoreConfig(
        engine=os.getenv("VECTOR_STORE_ENGINE", "chroma"),
        collection_name=os.getenv("VECTOR_STORE_COLLECTION", "unified_knowledge"),
        distance_metric=os.getenv("VECTOR_STORE_DISTANCE_METRIC", "cosine"),
        
        # ChromaDB specific settings
        vector_db_path=os.getenv("CHROMA_DB_PATH", "./vector_db"),
        
        # Qdrant specific settings
        qdrant_location=os.getenv("QDRANT_LOCATION", "local"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        qdrant_local_path=os.getenv("QDRANT_LOCAL_PATH", "./qdrant_db"),
        qdrant_prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "False").lower() == "true"
    )
    
    # Create LLM provider for queries
    llm_config = LLMConfig()
    llm_config.query_llm.enabled = True
    llm_config.query_llm.model_name = os.getenv("QUERY_LLM_MODEL")
    llm_config.query_llm.provider = os.getenv("QUERY_LLM_PROVIDER")
    llm_provider = DefaultLLMProvider(llm_config)
    query_llm = llm_provider.get_query_llm()
    
    if not query_llm:
        logger.error("Could not initialize query LLM. Using default settings.")
        from llama_index.llms import OpenAI
        query_llm = OpenAI()
    
    # Create embedder with explicit configuration
    embedder_config = EmbedderConfig(
        provider=os.getenv("EMBEDDER_PROVIDER", "ollama"),
        model_name=os.getenv("EMBEDDER_MODEL", "nomic-embed-text"),
        api_key_env_var=os.getenv("EMBEDDER_API_KEY_ENV_VAR"),
        api_base=os.getenv("EMBEDDER_API_BASE"),
        embed_batch_size=int(os.getenv("EMBEDDER_BATCH_SIZE", "10")),
        use_cache=os.getenv("EMBEDDER_USE_CACHE", "True").lower() == "true",
        cache_dir=os.getenv("EMBEDDER_CACHE_DIR", "./.cache/embeddings")
    )
    
    # Log embedder configuration
    logger.info(f"Using embedder provider: {embedder_config.provider}")
    logger.info(f"Using embedder model: {embedder_config.model_name}")
    
    # Create embedder instance
    embedder = EmbedderFactory.create_embedder(embedder_config)
    
    # Load vector store and index
    vector_store = VectorStoreFactory.create_vector_store(vector_store_config)
    index = vector_store.load()
    
    # Create query pipeline with properly configured components
    query_pipeline = QueryPipeline(
        config=QueryPipelineConfig(),
        index=index,
        llm=query_llm,
        embedder=embedder,  # Pass the embedder explicitly
    )
    
    # Run a simple query
    query = "how can I SET IO PIN?"
    
    logger.info(f"Running query: '{query}'")
    start_time = time.time()
    response = query_pipeline.query(query)
    end_time = time.time()
    
    # Print results
    print(f"\nQuery: {query}")
    print(f"\nAnswer (generated in {end_time - start_time:.2f} seconds):")
    print("-" * 80)
    print(response.response)
    
    # Print source nodes
    print_source_nodes(response.source_nodes)
    
    # Print metrics
    if response.metadata and "metrics" in response.metadata:
        print("\nPerformance Metrics:")
        metrics = response.metadata["metrics"]
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}s")
            else:
                print(f"  {key}: {value}")


def main():
    """Main function executing the examples."""
    logger.info("Starting Embedder Query Example")
    
    try:
        # Run the example
        run_query_with_explicit_embedder()
        
    except Exception as e:
        logger.error(f"Error in examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
