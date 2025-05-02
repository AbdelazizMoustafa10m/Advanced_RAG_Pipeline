#!/usr/bin/env python3

# Basic imports
from re import T
import sys
import os
import time
import logging
from multiprocessing import freeze_support

# Use direct imports from the project root
from core.config import UnifiedConfig, ParallelConfig, DocumentType, RegistryConfig, EmbedderConfig
from pipeline.orchestrator import PipelineOrchestrator
from indexing.vector_store import ChromaVectorStoreAdapter
from core.interfaces import IVectorStore
    
# LlamaIndex imports
from llama_index.core.schema import MetadataMode
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from registry.document_registry import DocumentRegistry

# --- Standard Library Imports ---
from dotenv import load_dotenv

# --- Global Setup ---
load_dotenv() # Load API keys from .env file

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout) # Ensure logs go to console
        # logging.FileHandler("rag_pipeline.log") # Optional: Log to file
    ]
)
# Reduce verbosity from libraries if needed
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
# ------------------


# --- Main Execution ---
if __name__ == "__main__":
    # Apply multiprocessing guard
    freeze_support()
    logger.info("Starting Unified RAG Parser Example...")

    # 1. --- Configuration ---
    try:
        # Create configuration object
        # Adjust input_directory to point to the ACTUAL root
        # containing your 'code_repository' and 'technical_docs' subdirs
        # Import LLMConfig and LLMSettings for custom configuration
        from core.config import LLMConfig, LLMSettings
        
        # Create a custom LLMConfig with selective metadata enrichment
        llm_config = LLMConfig()
        
        # Set up the metadata LLM
        llm_config.metadata_llm.enabled = True  # Enable the metadata LLM
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
            processed_nodes = orchestrator.run()
        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}", exc_info=True)
            # Continue with empty nodes list to see what we can salvage
            processed_nodes = []
        
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
        logger.warning("No nodes were generated by the pipeline. Exiting.")
        sys.exit(0)


       # --- Output ---
    # Print a summary of processed nodes
    print("\n=== Node Processing Summary ===")
    print(f"Total nodes processed: {len(processed_nodes)}")
    print(f"Code nodes: {len([n for n in processed_nodes if n.metadata.get('node_type') == 'code'])}")
    print(f"Document nodes: {len([n for n in processed_nodes if n.metadata.get('node_type') == 'document'])}")
    print(f"Unknown type nodes: {len([n for n in processed_nodes if n.metadata.get('node_type') not in ['code', 'document']])}")
    print("============================")
    
    print("\n--- Saving All Enriched Nodes to File ---")
    
    # Save all nodes to a file with the requested format
    output_file_path = "node_contents.txt"
    with open(output_file_path, "w") as f:
        for i, node in enumerate(processed_nodes):
            f.write(f"--- Node {i+1}/{len(processed_nodes)} ---\n")
            f.write(node.get_content(metadata_mode=MetadataMode.ALL))
            f.write("\n" + "-" * 30 + "\n" + "-" * 30 + "\n\n")
            
            f.write("The LLM sees this: \n")
            f.write(node.get_content(metadata_mode=MetadataMode.LLM))
            f.write("\n" + "-" * 30 + "\n" + "-" * 30 + "\n\n")
            
            f.write("The Embedding model sees this: \n")
            f.write(node.get_content(metadata_mode=MetadataMode.EMBED))
            f.write("\n" + "-" * 30 + "\n" + "-" * 30 + "\n\n")
    
    print(f"All node contents have been saved to {output_file_path}")