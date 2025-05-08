"""
Configuration Manager for Advanced RAG Pipeline.

This module provides a central configuration management system that loads
configuration from multiple sources with clear precedence:
1. Default values (lowest precedence)
2. Environment variables
3. YAML configuration files
4. Explicit overrides (highest precedence)

It handles validation, environment-specific configuration, and ensures
consistency between related components.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, Type, TypeVar
from pathlib import Path
from dotenv import load_dotenv

# Import YAML library with proper error handling
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logging.warning("PyYAML is not installed. Install with 'pip install pyyaml'")

from pydantic import BaseModel

# Import from the new config_models module instead of config
from core.config import (
    UnifiedConfig, ApplicationEnvironment, 
    EmbedderConfig, VectorStoreConfig, 
    LLMConfig
)

# Type variable for generic config models
T = TypeVar('T', bound=BaseModel)

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Central configuration manager for the Advanced RAG Pipeline.
    
    Handles loading configuration from multiple sources with clear precedence,
    validation, and environment-specific configuration.
    """
    
    def __init__(
        self, 
        env_prefix: str = "RAG_",
        config_path: Optional[str] = None,
        env_file: Optional[str] = None,
        environment: Optional[str] = None
    ):
        """
        Initialize the configuration manager.
        
        Args:
            env_prefix: Prefix for environment variables
            config_path: Path to YAML configuration file
            env_file: Path to .env file
            environment: Application environment (development, testing, staging, production)
        """
        # Check if PyYAML is available
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for configuration management. "
                "Please install with 'pip install pyyaml'"
            )
            
        self.env_prefix = env_prefix
        self.config_path = config_path
        self.env_file = env_file or ".env"
        
        # Load environment variables
        self._load_env_variables()
        
        # Determine environment
        self.environment = environment or os.getenv(
            f"{env_prefix}ENVIRONMENT", 
            ApplicationEnvironment.DEVELOPMENT.value
        )
        
        # Store loaded configurations
        self._config_cache: Dict[str, BaseModel] = {}
    
    def _load_env_variables(self) -> None:
        """Load environment variables from .env file."""
        if os.path.exists(self.env_file):
            load_dotenv(self.env_file)
            logger.info(f"Loaded environment variables from {self.env_file}")
    
    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            file_path: Path to YAML configuration file
            
        Returns:
            Configuration as a dictionary
        """
        if not os.path.exists(file_path):
            logger.warning(f"Configuration file not found: {file_path}")
            return {}
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in ('.yaml', '.yml'):
            logger.warning(f"Expected YAML file but got {file_ext} extension: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
                logger.info(f"Loaded YAML configuration from {file_path}")
                return config_data if config_data else {}
        except Exception as e:
            logger.error(f"Error loading configuration from {file_path}: {e}")
            return {}
    
    def _get_env_config(self, model_class: Type[T]) -> Dict[str, Any]:
        """
        Get configuration from environment variables for a specific model.
        
        Args:
            model_class: Pydantic model class
            
        Returns:
            Configuration from environment variables as a dictionary
        """
        env_config = {}
        prefix = f"{self.env_prefix}{model_class.__name__.upper()}_"
        
        # Get all environment variables with the prefix
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert environment variable name to config field name
                field_name = key[len(prefix):].lower()
                env_config[field_name] = value
        
        return env_config
    
    def _merge_config_sources(
        self,
        model_class: Type[T],
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Merge configuration from multiple sources with clear precedence.
        
        Precedence order (lowest to highest):
        1. Default values from model
        2. Environment-specific configuration file
        3. Main configuration file
        4. Environment variables
        5. Explicit overrides
        
        Args:
            model_class: Pydantic model class
            overrides: Explicit configuration overrides
            
        Returns:
            Merged configuration as a dictionary
        """
        # Start with an empty configuration
        config_dict = {}
        
        # Load from configuration file if specified
        if self.config_path:
            file_config = self._load_from_file(self.config_path)
            
            # Extract relevant section for this model if it exists
            model_name = model_class.__name__
            if model_name in file_config:
                config_dict.update(file_config[model_name])
            elif model_name.lower() in file_config:
                config_dict.update(file_config[model_name.lower()])
            # For UnifiedConfig, use the top-level configuration
            elif model_class.__name__ == 'UnifiedConfig':
                config_dict.update(file_config)
        
        # Load environment-specific configuration
        env_config_path = os.path.join(
            os.path.dirname(self.config_path) if self.config_path else ".",
            f"config.{self.environment.lower()}.yaml"
        )
        if os.path.exists(env_config_path):
            env_file_config = self._load_from_file(env_config_path)
            
            # Extract relevant section for this model
            model_name = model_class.__name__
            if model_name in env_file_config:
                config_dict.update(env_file_config[model_name])
            elif model_name.lower() in env_file_config:
                config_dict.update(env_file_config[model_name.lower()])
            # For UnifiedConfig, use the top-level configuration
            elif model_class.__name__ == 'UnifiedConfig':
                config_dict.update(env_file_config)
        
        # Get configuration from environment variables
        env_config = self._get_env_config(model_class)
        config_dict.update(env_config)
        
        # Apply explicit overrides (highest precedence)
        if overrides:
            config_dict.update(overrides)
        
        return config_dict
    
    def get_config(
        self,
        model_class: Type[T],
        overrides: Optional[Dict[str, Any]] = None,
        cache: bool = True
    ) -> T:
        """
        Get configuration for a specific model, merged from all sources.
        
        Args:
            model_class: Pydantic model class
            overrides: Explicit configuration overrides
            cache: Whether to cache the configuration
            
        Returns:
            Configuration as a Pydantic model instance
        """
        # Check cache first
        cache_key = model_class.__name__
        if cache and cache_key in self._config_cache:
            # Apply overrides to cached config if needed
            if overrides:
                return model_class.model_validate({
                    **self._config_cache[cache_key].model_dump(),
                    **overrides
                })
            return self._config_cache[cache_key]
        
        # Merge configuration from all sources
        config_dict = self._merge_config_sources(model_class, overrides)
        
        # Create model instance
        try:
            config = model_class.model_validate(config_dict)
            
            # Cache configuration if requested
            if cache:
                self._config_cache[cache_key] = config
            
            return config
        except Exception as e:
            logger.error(f"Error validating configuration for {model_class.__name__}: {e}")
            
            # For UnifiedConfig, ensure we provide input_directory if missing
            if model_class.__name__ == 'UnifiedConfig' and 'input_directory' not in config_dict:
                # Try to add input_directory from environment or a fallback for testing
                input_dir = os.getenv('RAG_INPUT_DIRECTORY', './data')
                logger.warning(f"Adding missing required input_directory: {input_dir}")
                
                # Create a temporary directory for testing if needed
                if input_dir == './data' and not os.path.exists(input_dir):
                    os.makedirs(input_dir, exist_ok=True)
                
                # Create model with input_directory
                return model_class(input_directory=input_dir)
            
            # Fall back to default configuration
            return model_class()
    
    def get_unified_config(
        self,
        overrides: Optional[Dict[str, Any]] = None,
        ensure_embedder_consistency: bool = True
    ) -> UnifiedConfig:
        """
        Get the unified configuration, ensuring consistency between components.
        
        Args:
            overrides: Explicit configuration overrides
            ensure_embedder_consistency: Whether to ensure consistency between
                indexing and querying embedders
            
        Returns:
            Unified configuration
        """
        # Get unified configuration with overrides
        config = self.get_config(UnifiedConfig, overrides)
        
        # Ensure consistency between embedders if requested
        if ensure_embedder_consistency:
            if config.vector_store.engine in ["chroma", "qdrant"]:
                # Extract embedder settings to ensure consistency
                embedder = config.embedder
                retrieval = config.query_pipeline.retrieval
                
                # If retrieval is using embedding-based retrieval, ensure consistency
                if retrieval.retriever_strategy in ["vector", "hybrid", "ensemble"]:
                    # Ensure retrieval uses the same embedder settings
                    logger.info(
                        f"Ensuring consistent embedder settings between indexing and querying: "
                        f"{embedder.provider}/{embedder.model_name}"
                    )
                    
                    # Set retrieval embedder settings to match indexing embedder
                    # Note: This would typically be handled via proper model inheritance in a real system
                    if hasattr(retrieval, "embedder_provider"):
                        setattr(retrieval, "embedder_provider", embedder.provider)
                    if hasattr(retrieval, "embedder_model"):
                        setattr(retrieval, "embedder_model", embedder.model_name)
        
        return config
    
    def save_config(self, config: BaseModel, file_path: Optional[str] = None) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config: Configuration to save
            file_path: Path to save the configuration to
        """
        file_path = file_path or self.config_path
        if not file_path:
            logger.warning("No file path specified for saving configuration")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            # Convert model to dictionary
            config_dict = config.model_dump()
            
            # Ensure file has YAML extension
            file_path = self._ensure_yaml_extension(file_path)
            
            # Save to file in YAML format
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Configuration saved to {file_path} in YAML format")
        except Exception as e:
            logger.error(f"Error saving configuration to {file_path}: {e}")
    
    def _ensure_yaml_extension(self, file_path: str) -> str:
        """
        Ensure file path has a YAML extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            Path with .yaml extension
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in ('.yaml', '.yml'):
            return os.path.splitext(file_path)[0] + '.yaml'
        return file_path