# --- examples/query_example.py ---

import sys
import os
import logging
from pathlib import Path
import time

# Add the project root to the path to make imports work
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from core.config import QueryPipelineConfig
from core.config import UnifiedConfig, ParallelConfig, DocumentType, RegistryConfig, EmbedderConfig
from core.config import VectorStoreConfig
from embedders.embedder_factory import EmbedderFactory
from pipeline.orchestrator import PipelineOrchestrator
from llm.providers import DefaultLLMProvider
from indexing.vector_store import VectorStoreFactory
from query.query_pipeline import QueryPipeline
from query.transformers import HyDEQueryExpander, LLMQueryRewriter
from query.rerankers import SemanticReranker, LLMReranker
from query.synthesis import RefineResponseSynthesizer, TreeSynthesizer, CompactResponseSynthesizer
from core.config import LLMConfig, LLMSettings
from registry.document_registry import DocumentRegistry
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
logger = logging.getLogger("query-example")


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

def initialize_Pipeline():
    """Initialize the pipeline.
    
    Returns:
        tuple: (processed_nodes, index, vector_store)
            - processed_nodes: List of TextNode objects processed by the pipeline
            - index: VectorStoreIndex created from the processed nodes
            - vector_store: Vector store instance used for indexing
    """

    try:
        # Create configuration object
        # Adjust input_directory to point to the ACTUAL root
        # containing your 'code_repository' and 'technical_docs' subdirs
        # Import LLMConfig and LLMSettings for custom configuration
        from core.config import LLMConfig, LLMSettings
        
        # Create a custom LLMConfig with selective metadata enrichment
        llm_config = LLMConfig()
        
        # Set up the metadata LLM
        llm_config.metadata_llm.enabled = False  # Enable the metadata LLM
        llm_config.metadata_llm.model_name = os.getenv("METADATA_LLM_MODEL")
        llm_config.metadata_llm.provider = os.getenv("METADATA_LLM_PROVIDER")
        llm_config.metadata_llm.api_key_env_var = os.getenv("METADATA_LLM_API_KEY_ENV_VAR")
        
        # Debug LLM configuration
        logger.debug(f"Metadata LLM Model: {llm_config.metadata_llm.model_name}")
        logger.debug(f"Metadata LLM Provider: {llm_config.metadata_llm.provider}")
        logger.debug(f"Metadata LLM API Key Env Var: {llm_config.metadata_llm.api_key_env_var}")
        logger.debug(f"Actual API Key Value: {'[SET]' if os.getenv(llm_config.metadata_llm.api_key_env_var) else '[NOT SET]'}")
        
        # Disable the other LLMs
        llm_config.query_llm.enabled = False
        llm_config.coding_llm.enabled = False
        
        # Configure selective enrichment
        llm_config.enrich_documents = True  # Enable metadata enrichment for Docling documents (PDF, DOCX, etc.)
        llm_config.enrich_code = True       # Enable metadata enrichment for code documents
        
        # Create embedder configuration
        embedder_config = EmbedderConfig(
            provider=os.getenv("EMBEDDER_PROVIDER", "huggingface"),
            model_name=os.getenv("EMBEDDER_MODEL", "BAAI/bge-small-en-v1.5"),
            api_key_env_var=os.getenv("EMBEDDER_API_KEY_ENV_VAR"),
            api_base=os.getenv("EMBEDDER_API_BASE"),
            embed_batch_size=int(os.getenv("EMBEDDER_BATCH_SIZE", "10")),
            use_cache=True,
            cache_dir="./.cache/embeddings"
        )
        
        # Create vector store configuration
        from core.config import VectorStoreConfig
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

        config = UnifiedConfig(
            input_directory="./data", # Or "/path/to/your/data"
            llm=llm_config, # Use the empty LLM config to avoid API key errors
            parallel=ParallelConfig(max_workers=4), # Use max_workers from config
            registry=RegistryConfig(
                db_path="./doc_reg_db/document_registry.db"
            ),
            embedder=embedder_config, # Add the embedder configuration
            vector_store=vector_store_config # Add the vector store configuration
        )
        logger.info("Configuration loaded successfully.")
        # You can print the config for verification: logger.debug(config)
    except ValueError as e:
        logger.error(f"Configuration validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}", exc_info=True)
        sys.exit(1)

    # The embedder is now configured through the EmbedderConfig and initialized in the orchestrator
    # No need to set global embedding model as it's handled by the embedder service
    logger.info(f"Configured embedder with provider: {config.embedder.provider}, model: {config.embedder.model_name}")

    # 2. --- Pipeline Execution ---
    # Initialize and pass to orchestrator
    db_path = config.registry.db_path or os.path.join(config.output_dir, "document_registry.db")
    document_registry = DocumentRegistry(db_path=db_path)
    
    # Reset any stalled processing
    reset_count = document_registry.reset_stalled_processing()
    if reset_count > 0:
        logger.info(f"Reset {reset_count} stalled documents")
    
    # Get processing statistics
    stats = document_registry.get_processing_stats()
    logger.info(f"Document processing stats: {stats}")
    
    processed_nodes = [] # Initialize
    try:
        logger.info("Initializing Pipeline Orchestrator...")
        orchestrator = PipelineOrchestrator(config, document_registry)
        
        # Debug orchestrator components
        logger.debug(f"Embedder initialized: {orchestrator.embedder is not None}")
        logger.debug(f"LLM Provider initialized: {orchestrator.llm_provider is not None}")
        if orchestrator.llm_provider:
            logger.debug(f"LLM Provider type: {type(orchestrator.llm_provider).__name__}")
            logger.debug(f"LLM Provider model: {getattr(orchestrator.llm_provider, 'model_name', 'Unknown')}")
        
        logger.info("Running the processing pipeline...")
        # The orchestrator now handles loading, detecting, routing, processing, enriching
        try:
            run_result = orchestrator.run()
            processed_nodes, index, vector_store = run_result
        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}", exc_info=True)
            # Continue with empty nodes list to see what we can salvage
            processed_nodes = []
            index = None
            vector_store = None
        
        # Count nodes by type for better statistics
        code_nodes = [n for n in processed_nodes if n.metadata.get('node_type') == 'code']
        document_nodes = [n for n in processed_nodes if n.metadata.get('node_type') == 'document']
        unknown_nodes = [n for n in processed_nodes if n.metadata.get('node_type') not in ['code', 'document']]

        # Reset any stalled processing
        reset_count = document_registry.reset_stalled_processing()
        if reset_count > 0:
            logger.info(f"Reset {reset_count} stalled documents")
        
        # Get processing statistics
        stats = document_registry.get_processing_stats()
        logger.info(f"Document processing stats: {stats}")
        
        # Log detailed statistics
        logger.info(f"Pipeline finished. Processed {len(processed_nodes)} total nodes:")
        logger.info(f"  - {len(code_nodes)} code nodes from CodeProcessor")
        logger.info(f"  - {len(document_nodes)} document nodes from DoclingChunker")
        if unknown_nodes:
            logger.info(f"  - {len(unknown_nodes)} nodes with unknown type")
        
        # Print all tracked documents
        print("\n=== Document Registry Contents ===")
        all_docs = document_registry.list_all_documents()
        for doc in all_docs:
            print(f"Document: {doc['doc_id']}")
            print(f"  Status: {doc['status']}")
            print(f"  Last processed: {time.ctime(doc['last_processed'])}")
            print("---")
        print("===============================\n")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)

    if not processed_nodes:
        logger.warning("No nodes were generated by the pipeline.")
        return [], None, None
        
    return processed_nodes, index, vector_store


def run_basic_query_example():
    """Run a basic query example with default settings."""
    print_section_header("Basic Query Example")
    
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


def run_advanced_query_example():
    """Run an advanced query example with custom components."""
    print_section_header("Advanced Query Example")
    
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
    nodes = vector_store.get_all_nodes() if hasattr(vector_store, 'get_all_nodes') else None
    
    # Create custom query pipeline configuration
    from core.config import (
        QueryPipelineConfig, 
        QueryTransformationConfig, 
        RetrievalConfig, 
        RerankerConfig, 
        SynthesisConfig
    )
    
    query_config = QueryPipelineConfig(
        transformation=QueryTransformationConfig(
            enable_query_expansion=True,
            use_hyde=True,
            hyde_prompt_template="Generate a comprehensive passage that would answer this question: {query}",
            enable_query_rewriting=True,
            rewriting_technique="instruct"
        ),
        retrieval=RetrievalConfig(
            retriever_strategy="hybrid" if nodes else "vector",
            similarity_top_k=7,
            hybrid_alpha=0.7  # Weight more towards vector search
        ),
        reranker=RerankerConfig(
            enable_reranking=True,
            reranker_type="semantic",
            rerank_top_n=5
        ),
        synthesis=SynthesisConfig(
            synthesis_strategy="refine",
            include_citations=True
        )
    )
    
    # Create query pipeline with advanced configuration
    query_pipeline = QueryPipeline(
        config=query_config,
        index=index,
        llm=query_llm,
        embedder=embedder,  # Pass the embedder explicitly
    )
    
    # Run a complex query
    query = "Explain how the different techniques for FBL Validation work and how each of them affect the FBL behviour when it comes to Memory and which is the most effective FBL Validation strategy"
    
    logger.info(f"Running query with advanced configuration: '{query}'")
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
    
    # Print transformation info if available
    if response.metadata and "transformations" in response.metadata:
        print("\nQuery Transformations:")
        transformations = response.metadata["transformations"]
        
        for transformer, result in transformations.items():
            print(f"  {transformer}:")
            if "rewritten_query" in result:
                print(f"    Rewritten query: {result['rewritten_query']}")
            if "hypothetical_document" in result and result["hypothetical_document"]:
                hypodoc = result["hypothetical_document"]
                print(f"    Hypothetical document: {hypodoc[:100]}...")
    
    # Print metrics
    if response.metadata and "metrics" in response.metadata:
        print("\nPerformance Metrics:")
        metrics = response.metadata["metrics"]
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}s")
            else:
                print(f"  {key}: {value}")


def run_query_component_comparison():
    """Run a comparison of different query components."""
    print_section_header("Query Component Comparison")
    
    # Create vector store configuration
    from core.config import VectorStoreConfig
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
    from core.config import LLMConfig
    from llm.providers import DefaultLLMProvider
    
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
    

    vector_store = VectorStoreFactory.create_vector_store(vector_store_config)
    index = vector_store.load()
    
    # Define the test query
    query = "What techniques are used for query transformation in the advanced RAG pipeline?"
    
    # 1. Test query rewriting
    print("\n1. Testing Query Rewriting")
    rewriter = LLMQueryRewriter(llm=query_llm)
    start_time = time.time()
    rewrite_result = rewriter.transform(query)
    end_time = time.time()
    
    print(f"Original Query: {query}")
    print(f"Rewritten Query: {rewrite_result.get('rewritten_query', 'N/A')}")
    print(f"Time: {end_time - start_time:.2f}s")
    
    # 2. Test different response synthesizers
    synthesizers = {
        "Refine Synthesizer": RefineResponseSynthesizer(llm=query_llm),
        "Tree Synthesizer": TreeSynthesizer(llm=query_llm),
        "Compact Synthesizer": CompactResponseSynthesizer(llm=query_llm)
    }
    
    # Get retrieval results once
    from query.retrieval import EnhancedRetriever
    retriever = EnhancedRetriever(index=index)
    retrieved_nodes = retriever.retrieve(query, top_k=5)
    
    # Test each synthesizer
    for name, synthesizer in synthesizers.items():
        print(f"\n2. Testing {name}")
        start_time = time.time()
        response = synthesizer.synthesize(query, retrieved_nodes)
        end_time = time.time()
        
        print(f"Answer (first 150 chars):")
        print(f"{response.response[:150]}...")
        print(f"Time: {end_time - start_time:.2f}s")


def main():
    """Main function executing the examples."""
    logger.info("Starting Query Module Examples")
    
    try:
        # Run the examples
        run_basic_query_example()
        run_advanced_query_example()
        run_query_component_comparison()
        
    except Exception as e:
        logger.error(f"Error in examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()