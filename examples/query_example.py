# --- examples/query_example.py ---

import sys
import os
import logging
from pathlib import Path
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple

# Add the project root to the path to make imports work
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import colored logging utility
from utils.colored_logging import setup_colored_logging

# Import core modules for configuration management
from core.config_manager import ConfigManager
from core.config import UnifiedConfig

# Import components needed for initialization
from registry.document_registry import DocumentRegistry
from llm.providers import DefaultLLMProvider
from embedders.embedder_factory import EmbedderFactory
from indexing.vector_store import VectorStoreFactory
from pipeline.orchestrator import PipelineOrchestrator


# Import project modules
from core.config import QueryPipelineConfig, ApplicationEnvironment
from query.query_pipeline import QueryPipeline
from query.transformers import HyDEQueryExpander, LLMQueryRewriter
from query.rerankers import SemanticReranker, LLMReranker
from query.synthesis import RefineResponseSynthesizer, TreeSynthesizer, CompactResponseSynthesizer
from llama_index.core.schema import TextNode, MetadataMode

# Configure colored logging
setup_colored_logging(
    level=logging.INFO
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
        
        # Show more context - first 300 chars
        content = node.node.get_content()
        if len(content) > 300:
            displayed_content = content[:297] + "..."
        else:
            displayed_content = content
            
        print(f"  Content: {displayed_content}")
        
        # Print key metadata
        print("  Metadata:")
        for key in ['doc_type', 'section', 'title']:
            if key in node.node.metadata:
                print(f"    {key}: {node.node.metadata[key]}")
        print()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Query Example")
    parser.add_argument(
        "--config", 
        type=str, 
        default=os.environ.get("RAG_CONFIG_PATH", "./config.yaml"),
        help="Path to configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--env", 
        type=str, 
        default=os.environ.get("RAG_ENV_FILE", ".env"),
        help="Path to environment file"
    )
    parser.add_argument(
        "--environment", 
        choices=[e.value for e in ApplicationEnvironment],
        default=os.environ.get("RAG_ENVIRONMENT", ApplicationEnvironment.DEVELOPMENT.value),
        help="Application environment"
    )
    return parser.parse_args()

def initialize_components(config: UnifiedConfig) -> Dict[str, Any]:
    """Initialize components based on configuration."""
    components = {}
    
    # Initialize document registry
    if config.registry.enabled:
        db_path = config.registry.db_path or os.path.join(config.output_dir, "document_registry.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        components["registry"] = DocumentRegistry(db_path=db_path)
        logger.info(f"Initialized document registry at {db_path}")
    else:
        components["registry"] = None
        logger.info("Document registry is disabled")
    
    # Initialize LLM provider
    components["llm_provider"] = DefaultLLMProvider(config.llm)
    logger.info(f"Initialized LLM provider with metadata model: {config.llm.metadata_llm.model_name}")
    
    # Initialize embedder
    components["embedder"] = EmbedderFactory.create_embedder(config.embedder)
    logger.info(f"Initialized embedder with provider: {config.embedder.provider}, model: {config.embedder.model_name}")
    
    # Initialize vector store
    components["vector_store"] = VectorStoreFactory.create_vector_store(config.vector_store)
    logger.info(f"Initialized vector store with engine: {config.vector_store.engine}")
    
    # Initialize pipeline orchestrator
    components["orchestrator"] = PipelineOrchestrator(
        config,
        document_registry=components["registry"],
        llm_provider=components["llm_provider"]
    )
    logger.info("Initialized pipeline orchestrator")
    
    return components


def load_configuration(args=None):
    """Load configuration using ConfigManager directly.
    
    Args:
        args: Command line arguments (if None, will parse them)
        
    Returns:
        tuple: (config, components)
            - config: UnifiedConfig object with all configuration settings
            - components: Dictionary of initialized components
    """
    # Parse arguments if not provided
    if args is None:
        args = parse_args()
    
    # Build configuration overrides for the example
    config_overrides = {
        "input_directory": os.path.join(project_root, "data"),
        "output_dir": os.path.join(project_root, "output", "example_query"),
        "project_name": "query-example"
    }
    
    try:
        # Initialize configuration manager
        config_manager = ConfigManager(
            config_path=args.config,
            env_file=args.env,
            environment=args.environment
        )
        
        # Get unified configuration
        config = config_manager.get_unified_config(overrides=config_overrides)
        logger.info(f"Configuration loaded for project: {config.project_name}")
        logger.info(f"Environment: {config.environment}")
        logger.info(f"Input directory: {config.input_directory}")
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Using embedder provider: {config.embedder.provider}")
        logger.info(f"Using embedder model: {config.embedder.model_name}")
        logger.info(f"Using vector store engine: {config.vector_store.engine}")
        logger.info(f"Using collection name: {config.vector_store.collection_name}")
        
        # Initialize components
        components = initialize_components(config)
        
        return config, components
        
    except ValueError as e:
        logger.error(f"Configuration validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}", exc_info=True)
        sys.exit(1)


def initialize_Pipeline():
    """Initialize the pipeline.
    
    Returns:
        tuple: (processed_nodes, index, vector_store)
            - processed_nodes: List of TextNode objects processed by the pipeline
            - index: VectorStoreIndex created from the processed nodes
            - vector_store: Vector store instance used for indexing
    """

    try:
        # Load configuration and components
        config, components = load_configuration()
        
        # Extract components
        registry = components["registry"]
        vector_store = components["vector_store"]
        orchestrator = components["orchestrator"]
        llm_provider = components["llm_provider"]
    except ValueError as e:
        logger.error(f"Configuration validation error: {e}")
        return [], None, None
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}", exc_info=True)
        return [], None, None

    # 2. --- Pipeline Execution ---
    # Reset any stalled processing
    if registry:
        reset_count = registry.reset_stalled_processing()
        if reset_count > 0:
            logger.info(f"Reset {reset_count} stalled documents")
            
        # Get processing stats
        stats = registry.get_processing_stats()
        logger.info(f"Document processing stats: {stats}")
        
    # Initialize processed nodes list
    processed_nodes = []
    try:
        logger.info("Using pre-initialized Pipeline Orchestrator")
        
        # Debug orchestrator components
        logger.debug(f"Embedder initialized: {orchestrator.embedder is not None}")
        logger.debug(f"Detector initialized: {orchestrator.detector is not None}")
        logger.debug(f"Vector store initialized: {orchestrator.vector_store is not None}")
        
        # Load vector store and index
        if not vector_store:
            logger.warning("No vector store available. Skipping index loading.")
            return [], None, None
        
        # Load the index
        index = vector_store.load()
        
        # Get all nodes if available
        if hasattr(vector_store, 'get_all_nodes'):
            processed_nodes = vector_store.get_all_nodes()
            logger.info(f"Loaded {len(processed_nodes)} nodes from vector store")
        else:
            # If get_all_nodes is not available, use an empty list
            processed_nodes = []
            logger.warning("Vector store does not support get_all_nodes. Using empty list.")
            
        # If no nodes were loaded but we have a valid index, create some dummy nodes for testing
        if not processed_nodes and index:
            logger.info("Creating dummy nodes for testing since no nodes were loaded")
            # Create a few dummy nodes for testing
            from llama_index.core.schema import TextNode
            processed_nodes = [
                TextNode(text="This is a test node about the Advanced RAG Pipeline.", 
                        metadata={"doc_type": "document", "title": "Test Document"}),
                TextNode(text="The query pipeline supports multiple retrieval strategies.", 
                        metadata={"doc_type": "code", "file_path": "query/retrieval.py"})
            ]       
        # Count nodes by type for better statistics
        code_nodes = [n for n in processed_nodes if n.metadata.get('node_type') == 'code']
        document_nodes = [n for n in processed_nodes if n.metadata.get('node_type') == 'document']
        unknown_nodes = [n for n in processed_nodes if n.metadata.get('node_type') not in ['code', 'document']]

        # Log detailed statistics
        logger.info(f"Pipeline finished. Processed {len(processed_nodes)} total nodes:")
        
        # Print all tracked documents if registry exists
        if registry:
            print("\n=== Document Registry Contents ===")
            all_docs = registry.list_all_documents()
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
    
    # Load configuration and components
    config, components = load_configuration()
    
    # Extract components
    llm_provider = components["llm_provider"]
    embedder = components["embedder"]
    vector_store = components["vector_store"]
    
    # Get the query LLM from the provider
    query_llm = llm_provider.get_query_llm()
    
    if not query_llm:
        logger.error("Could not initialize query LLM. Using default settings.")
        from llama_index.llms import OpenAI
        query_llm = OpenAI()
    
    
    # Load vector store index
    index = vector_store.load()
    
    # Check if we have a valid index
    if not index:
        logger.warning("No valid index available. Cannot run basic query example.")
        return
    
    # Create query pipeline with properly configured components
    # Disable caching to ensure we see the full pipeline execution
    pipeline_config = QueryPipelineConfig()
    pipeline_config.cache_results = False
    
    query_pipeline = QueryPipeline(
        config=pipeline_config,
        index=index,
        llm=query_llm,
        embedder=embedder,  # Pass the embedder explicitly
    )
    
    # Run a simple query with a slight variation to avoid cache hits
    query = "how can I set an IO pin in my code?"
    
    # Debug: Extract and print the HyDE hypothetical document
    logger.info(f"Running query: '{query}'")
    
    # Add debug code to verify HyDE is working correctly
    print("\n===== HyDE Debug Information =====")
    
    # Get the HyDE transformer
    hyde_transformer = None
    for transformer in query_pipeline.query_transformers:  # Use query_transformers instead of transformers
        if isinstance(transformer, HyDEQueryExpander):
            hyde_transformer = transformer
            break
    
    if hyde_transformer:
        # Process the query with HyDE transformer only to see the result
        hyde_result = hyde_transformer.transform(query)
        hypothetical_doc = hyde_result.get("hypothetical_document")
        
        print("Original Query:", query)
        print("\nHyDE Generated Document:")
        print("-" * 80)
        print(hypothetical_doc)
        print("-" * 80)
        
        # Create a query bundle to verify custom_embedding_strs
        from llama_index.core.schema import QueryBundle
        query_bundle = QueryBundle(
            query_str=query,
            custom_embedding_strs=[hypothetical_doc]
        )
        
        # Verify that custom_embedding_strs is being used
        print("\nVerifying QueryBundle:")
        print(f"- Has custom_embedding_strs: {hasattr(query_bundle, 'custom_embedding_strs')}")
        if hasattr(query_bundle, 'custom_embedding_strs'):
            print(f"- Length of custom_embedding_strs: {len(query_bundle.custom_embedding_strs)}")
            print(f"- First 100 chars: {query_bundle.custom_embedding_strs[0][:100]}...")
    else:
        print("HyDE transformer not found in the pipeline!")
    
    print("===== End of HyDE Debug Information =====\n")
    
    # Run the actual query
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
    
    # Load configuration and components
    config, components = load_configuration()
    
    # Extract components
    llm_provider = components["llm_provider"]
    embedder = components["embedder"]
    vector_store = components["vector_store"]
    
    # Create a proper QueryPipelineConfig object with custom settings
    from core.config import QueryPipelineConfig, QueryTransformationConfig, RetrievalConfig, RerankerConfig, SynthesisConfig
    
    # Create component configs
    transformation_config = QueryTransformationConfig(
        enable_query_expansion=True,
        enable_query_rewriting=True,
        use_hyde=True
    )
    
    retrieval_config = RetrievalConfig(
        retriever_strategy="hybrid",
        similarity_top_k=5,
        use_hybrid_search=True,
        hybrid_alpha=0.5
    )
    
    reranker_config = RerankerConfig(
        enable_reranking=True,
        reranker_type="semantic",
        rerank_top_n=5
    )
    
    synthesis_config = SynthesisConfig(
        synthesis_strategy="tree",
        include_citations=True,
        tree_width=3
    )
    
    # Create the main query pipeline config
    query_pipeline_config = QueryPipelineConfig(
        transformation=transformation_config,
        retrieval=retrieval_config,
        reranker=reranker_config,
        synthesis=synthesis_config,
        timeout_seconds=30,
        cache_results=True,
        cache_dir=os.path.join(project_root, ".cache", "query_results")
    )
    
    # Get the query LLM from the provider
    query_llm = llm_provider.get_query_llm()
    
    # Ensure we're using Ollama for embeddings
    if config.embedder.provider.lower() != "ollama":
        try:
            # Create a direct Ollama embedding model
            from llama_index.embeddings.ollama import OllamaEmbedding
            from core.config import EmbedderConfig
            from embedders.embedder_factory import EmbedderFactory
            
            # Create a new embedder config with Ollama
            embedder_config = EmbedderConfig(
                provider="ollama",
                model_name="nomic-embed-text",
                api_base="http://localhost:11434",
                use_cache=True
            )
            
            # Create a new embedder with Ollama
            embedder = EmbedderFactory.create_embedder(embedder_config)
            logger.info("Successfully created Ollama embedder")
        except Exception as e:
            logger.error(f"Failed to create Ollama embedder: {e}. Using configured embedder.")
    
    # Load the vector store index
    index = vector_store.load()
    
    # Check if we have a valid index
    if not index:
        logger.warning("No valid index available. Cannot run advanced query example.")
        return
    
    # Create query pipeline with advanced configuration
    query_pipeline = QueryPipeline(
        config=query_pipeline_config,
        index=index,
        llm=query_llm,
        embedder=embedder
    )
    
    # Run a complex query
    query = "Explain how the reranking components work and how they improve retrieval quality"
    
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
    
    # Load configuration and components
    config, components = load_configuration()
    
    # Extract components
    llm_provider = components["llm_provider"]
    embedder = components["embedder"]
    vector_store = components["vector_store"]
    
    # Load the vector store index
    index = vector_store.load()
    
    # Check if we have a valid index
    if not index:
        logger.warning("No valid index available. Cannot run query component comparison.")
        return
    
    # Get the query LLM from the provider
    query_llm = llm_provider.get_query_llm()
    
    # Log configuration
    logger.info(f"Using embedder provider: {config.embedder.provider}")
    logger.info(f"Using embedder model: {config.embedder.model_name}")
    logger.info(f"Using vector store engine: {config.vector_store.engine}")
    logger.info(f"Using vector store collection: {config.vector_store.collection_name}")
    
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
    
    # Force using Ollama for embeddings to avoid API key issues
    try:
        # Create a direct Ollama embedding model
        from llama_index.embeddings.ollama import OllamaEmbedding
        
        # Get model name from config or use a default
        model_name = "nomic-embed-text"  # Default to a reliable model
        if config.embedder.provider.lower() == "ollama":
            model_name = config.embedder.model_name or model_name
        
        # Configure API base URL
        api_base = "http://localhost:11434"  # Default Ollama URL
        if hasattr(config.embedder, 'api_base') and config.embedder.api_base:
            api_base = config.embedder.api_base
        
        # Create the embedding model
        direct_embed_model = OllamaEmbedding(
            model_name=model_name,
            base_url=api_base
        )
        
        logger.info(f"Using Ollama embeddings with model: {model_name}")
        
        # If our embedder doesn't match this configuration, create a new one
        if not hasattr(embedder, 'embed_model') or \
           not isinstance(embedder.embed_model, OllamaEmbedding) or \
           embedder.embed_model.model_name != model_name:
            # Update the embedder for future use
            from core.config import EmbedderConfig
            from embedders.embedder_factory import EmbedderFactory
            embedder_config = EmbedderConfig(
                provider="ollama",
                model_name=model_name,
                api_base=api_base,
                use_cache=True
            )
            embedder = EmbedderFactory.create_embedder(embedder_config)
    except Exception as e:
        logger.warning(f"Error creating embedding model: {e}. Using default embedding.")
        from llama_index.embeddings import DefaultEmbedding
        direct_embed_model = DefaultEmbedding()
    
    # Use the direct embedding model with the retriever
    retriever = EnhancedRetriever(index=index, embed_model=direct_embed_model)
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
        # Load configuration once at the beginning
        config, components = load_configuration()
        
        # Run the examples
        #run_basic_query_example()
        run_advanced_query_example()
        #run_query_component_comparison()
        
    except Exception as e:
        logger.error(f"Error in examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()