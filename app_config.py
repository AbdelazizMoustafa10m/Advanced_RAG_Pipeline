#!/usr/bin/env python3
"""
Application Configuration Loader

This script initializes the configuration for the Advanced RAG Pipeline
application. It loads configuration from multiple sources with clear precedence,
validates it, and ensures consistency between components.

Usage:
    python app_config.py [--config CONFIG_PATH] [--env ENV_FILE] [--environment ENVIRONMENT]

Example:
    python app_config.py --config ./config.json --env ./.env.production --environment production
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Add the project root to the path to make imports work
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import colored logging utility
from utils.colored_logging import setup_colored_logging

from core.config_manager import ConfigManager
from core.config import UnifiedConfig, ApplicationEnvironment

# Configure colored logging
setup_colored_logging(
    level=logging.INFO
)
logger = logging.getLogger("app_config")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Initialize application configuration")
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
        type=str, 
        choices=[e.value for e in ApplicationEnvironment],
        default=os.environ.get("RAG_ENVIRONMENT", ApplicationEnvironment.DEVELOPMENT.value),
        help="Application environment"
    )
    return parser.parse_args()


def initialize_config(
    config_path: Optional[str] = None,
    env_file: Optional[str] = None,
    environment: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> UnifiedConfig:
    """
    Initialize the application configuration.
    
    Args:
        config_path: Path to configuration file
        env_file: Path to environment file
        environment: Application environment
        overrides: Explicit configuration overrides
        
    Returns:
        Unified configuration
    """
    logger.info(f"Initializing configuration (environment: {environment})")
    
    # Create configuration manager
    config_manager = ConfigManager(
        config_path=config_path,
        env_file=env_file,
        environment=environment
    )
    
    # Get unified configuration with overrides
    config = config_manager.get_unified_config(overrides=overrides)
    
    logger.info(f"Configuration initialized for project: {config.project_name}")
    logger.info(f"Input directory: {config.input_directory}")
    logger.info(f"Embedder provider: {config.embedder.provider}, model: {config.embedder.model_name}")
    logger.info(f"Vector store engine: {config.vector_store.engine}")
    
    return config


def initialize_application(config: UnifiedConfig):
    """
    Initialize the application with the unified configuration.
    
    Args:
        config: Unified configuration
    """
    # Set up logging
    logging.getLogger().setLevel(getattr(logging, config.logging.level))
    
    # Initialize document registry
    if config.registry.enabled:
        from registry.document_registry import DocumentRegistry
        registry = DocumentRegistry(db_path=config.registry.db_path)
        logger.info(f"Document registry initialized with database at {config.registry.db_path}")
    else:
        registry = None
        logger.info("Document registry disabled")
    
    # Initialize LLM provider
    from llm.providers import DefaultLLMProvider
    llm_provider = DefaultLLMProvider(config.llm)
    logger.info(f"LLM provider initialized with metadata model: {config.llm.metadata_llm.model_name}")
    
    # Initialize embedder
    from embedders.embedder_factory import EmbedderFactory
    embedder = EmbedderFactory.create_embedder(config.embedder)
    logger.info(f"Embedder initialized with provider: {config.embedder.provider}")
    
    # Initialize vector store
    from indexing.vector_store import VectorStoreFactory
    vector_store = VectorStoreFactory.create_vector_store(config.vector_store)
    logger.info(f"Vector store initialized with engine: {config.vector_store.engine}")
    
    # Initialize pipeline orchestrator
    from pipeline.orchestrator import PipelineOrchestrator
    orchestrator = PipelineOrchestrator(config, document_registry=registry, llm_provider=llm_provider)
    logger.info("Pipeline orchestrator initialized")
    
    # Return initialized components
    return {
        "config": config,
        "registry": registry,
        "llm_provider": llm_provider,
        "embedder": embedder,
        "vector_store": vector_store,
        "orchestrator": orchestrator
    }


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Initialize configuration
        config = initialize_config(
            config_path=args.config,
            env_file=args.env,
            environment=args.environment
        )
        
        # Initialize application
        app = initialize_application(config)
        
        # Print success message
        logger.info("Application initialized successfully!")
        logger.info(f"Configuration summary:")
        logger.info(f"  Project: {config.project_name}")
        logger.info(f"  Environment: {config.environment}")
        logger.info(f"  Input directory: {config.input_directory}")
        logger.info(f"  Output directory: {config.output_dir}")
        logger.info(f"  LLM providers: {config.llm.metadata_llm.provider}, {config.llm.query_llm.provider}")
        logger.info(f"  Embedder: {config.embedder.provider}/{config.embedder.model_name}")
        logger.info(f"  Vector store: {config.vector_store.engine}")
        logger.info(f"Ready to process documents!")
        
    except Exception as e:
        logger.error(f"Error initializing application: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()