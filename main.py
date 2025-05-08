"""
Advanced RAG Pipeline - Main Application

This is the main entry point for the Advanced RAG Pipeline application.
It initializes the configuration, sets up components, runs the pipeline,
and handles the output.
"""

import logging
import time
import os
import sys
import argparse
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Add the project root to the path to make imports work
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import colored logging utility
from utils.colored_logging import setup_colored_logging

# Import configuration modules
from core.config_manager import ConfigManager
from core.config import UnifiedConfig, ApplicationEnvironment

# Import components
from pipeline.orchestrator import PipelineOrchestrator
from registry.document_registry import DocumentRegistry
from registry.status import ProcessingStatus
from indexing.vector_store import VectorStoreFactory
from embedders.embedder_factory import EmbedderFactory
from llm.providers import DefaultLLMProvider

# Import LlamaIndex components
from llama_index.core.schema import TextNode, MetadataMode

# Configure colored logging
setup_colored_logging(
    level=logging.INFO
)

# Reduce verbosity from libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Advanced RAG Pipeline")
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
    parser.add_argument(
        "--input-dir", 
        type=str, 
        help="Input directory path (overrides configuration)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Output directory path (overrides configuration)"
    )
    return parser.parse_args()


def build_config_overrides(args) -> Dict[str, Any]:
    """Build configuration overrides from command line arguments."""
    overrides = {}
    
    if args.input_dir:
        overrides["input_directory"] = args.input_dir
    
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    
    return overrides


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


def save_nodes(
    nodes: List[TextNode],
    output_path: str = "node_contents.txt",
    include_all_metadata: bool = True,
    include_llm_view: bool = True,
    include_embed_view: bool = True
) -> None:
    """Save processed nodes to a file."""
    logger.info(f"Saving {len(nodes)} processed nodes to {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        for i, node in enumerate(nodes):
            f.write(f"--- Node {i+1}/{len(nodes)} ---\n")
            
            if include_all_metadata:
                f.write(node.get_content(metadata_mode=MetadataMode.ALL))
                f.write("\n" + "-" * 30 + "\n" + "-" * 30 + "\n\n")
            
            if include_llm_view:
                f.write("The LLM sees this: \n")
                f.write(node.get_content(metadata_mode=MetadataMode.LLM))
                f.write("\n" + "-" * 30 + "\n" + "-" * 30 + "\n\n")
            
            if include_embed_view:
                f.write("The Embedding model sees this: \n")
                f.write(node.get_content(metadata_mode=MetadataMode.EMBED))
                f.write("\n" + "-" * 30 + "\n" + "-" * 30 + "\n\n")
    
    logger.info(f"All node contents have been saved to {output_path}")


def print_summary(
    nodes: List[TextNode],
    registry: Optional[DocumentRegistry] = None
) -> None:
    """Print a summary of processed nodes."""
    # Count nodes by type
    code_nodes = [n for n in nodes if n.metadata.get('node_type') == 'code']
    document_nodes = [n for n in nodes if n.metadata.get('node_type') == 'document']
    unknown_nodes = [n for n in nodes if n.metadata.get('node_type') not in ['code', 'document']]
    
    print("\n=== Node Processing Summary ===")
    print(f"Total nodes processed: {len(nodes)}")
    print(f"Code nodes: {len(code_nodes)}")
    print(f"Document nodes: {len(document_nodes)}")
    print(f"Unknown type nodes: {len(unknown_nodes)}")
    print("============================")
    
    # Print registry stats if available
    if registry:
        # Get processing statistics
        stats = registry.get_processing_stats()
        
        print("\n=== Document Registry Contents ===")
        all_docs = registry.list_all_documents()
        for doc in all_docs:
            print(f"Document: {doc['doc_id']}")
            print(f"  Status: {doc['status']}")
            print(f"  Last processed: {time.ctime(doc['last_processed'])}")
            print("---")
        print("===============================\n")
        
        print("\n=== Document Processing Stats ===")
        for status, count in stats.items():
            print(f"{status}: {count}")
        print("===============================\n")


def main():
    """Main entry point."""
    # Apply multiprocessing guard
    freeze_support()
    
    # Parse command line arguments
    args = parse_args()
    
    # Build configuration overrides
    config_overrides = build_config_overrides(args)
    
    try:
        logger.info("Starting Advanced RAG Pipeline...")
        
        # Initialize configuration manager
        config_manager = ConfigManager(
            config_path=args.config,
            env_file=args.env,
            environment=args.environment
        )
        
        # Get unified configuration
        config = config_manager.get_unified_config(overrides=config_overrides)
        logger.info(f"Configuration loaded for project: {config.project_name}")
        logger.info(f"Input directory: {config.input_directory}")
        
        # Validate input directory
        input_path = Path(config.input_directory)
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {config.input_directory}")
            sys.exit(1)
        
        # Initialize components
        components = initialize_components(config)
        registry = components["registry"]
        orchestrator = components["orchestrator"]
        
        # Reset any stalled documents
        if registry and config.registry.enabled:
            reset_count = registry.reset_stalled_processing(
                max_processing_time=config.registry.reset_stalled_after_seconds
            )
            if reset_count > 0:
                logger.info(f"Reset {reset_count} stalled documents")
        
        # Run the pipeline
        logger.info("Running the pipeline...")
        start_time = time.time()
        
        try:
            # The orchestrator returns a tuple of (nodes, index, vector_store) or just () when no docs processed
            result = orchestrator.run()
            
            # Handle the case where no new documents are processed
            if not result:
                logger.info("No new documents to process. Pipeline completed with 0 nodes.")
                # Set default values for variables
                processed_nodes = []
                index = None
                vector_store = None
            else:
                # Unpack the returned tuple when documents were processed
                processed_nodes, index, vector_store = result
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            nodes_count = len(processed_nodes) if processed_nodes else 0
            
            logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
            logger.info(f"Processed {nodes_count} nodes")
            
            if nodes_count > 0:
                # Save nodes to file
                output_file_path = os.path.join(config.output_dir, "node_contents.txt")
                save_nodes(processed_nodes, output_file_path)
                
                # Print summary
                print_summary(processed_nodes, registry)
            else:
                logger.warning("No nodes were generated by the pipeline")
        
        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}", exc_info=True)
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()