# --- embedders/embedder_factory.py ---

import logging
from typing import Optional, Dict, Type

from core.config import EmbedderConfig
from core.interfaces import IEmbedder
from .llamaindex_embedder_service import LlamaIndexEmbedderService

logger = logging.getLogger(__name__)

class EmbedderFactory:
    """Factory for creating embedder instances based on configuration."""
    
    @classmethod
    def create_embedder(cls, config: Optional[EmbedderConfig] = None) -> IEmbedder:
        """Create an embedder instance based on configuration.
        
        Args:
            config: Optional embedder configuration
            
        Returns:
            Embedder instance
        """
        config = config or EmbedderConfig()
        
        try:
            # Create and return the LlamaIndex embedder service
            return LlamaIndexEmbedderService(config)
        except Exception as e:
            logger.error(f"Error creating embedder: {str(e)}")
            
            # Try fallback if specified
            if config.fallback_provider and config.fallback_provider != config.provider:
                logger.info(f"Trying fallback provider: {config.fallback_provider}")
                fallback_config = EmbedderConfig(
                    provider=config.fallback_provider,
                    model_name=config.fallback_model or config.model_name,
                    embed_batch_size=config.embed_batch_size,
                    use_cache=config.use_cache,
                    cache_dir=config.cache_dir,
                    api_key_env_var=config.api_key_env_var,
                    api_base=config.api_base,
                    additional_kwargs=config.additional_kwargs
                )
                try:
                    return LlamaIndexEmbedderService(fallback_config)
                except Exception as fallback_error:
                    logger.error(f"Error creating fallback embedder: {str(fallback_error)}")
            
            # If all else fails, create a basic huggingface embedder
            logger.warning("Creating default huggingface embedder as last resort")
            default_config = EmbedderConfig()
            return LlamaIndexEmbedderService(default_config)
