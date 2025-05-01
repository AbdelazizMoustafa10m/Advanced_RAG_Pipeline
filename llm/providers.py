# llm/providers.py
import os
import logging
from typing import Optional

from llama_index.core.llms import LLM
# Import specific LLM integrations as needed
from llama_index.llms.groq import Groq
from llama_index.llms.openai import OpenAI # Example for another provider

# Import local interfaces and config
from core.interfaces import ILLMProvider
from core.config import LLMConfig, LLMSettings
from typing import Dict

logger = logging.getLogger(__name__)

class DefaultLLMProvider(ILLMProvider):
    """
    Default provider for LLM instances based on configuration.
    Handles different LLM roles (metadata, query, etc.).
    """
    def __init__(self, config: LLMConfig):
        """Initialize LLM provider with configuration."""
        self.config = config
        # Cache initialized LLM instances
        self._llm_instances: Dict[str, LLM] = {}

    def _get_llm_instance(self, role: str) -> Optional[LLM]:
        """Helper to get or create LLM instance for a specific role."""
        if role in self._llm_instances:
            return self._llm_instances[role]

        # Get the settings for the requested role
        settings: Optional[LLMSettings] = getattr(self.config, role, None)

        # Check if settings exists
        if settings is None:
            logger.error(f"LLM settings for role '{role}' are not configured.")
            return None
            
        # Check if LLM is enabled for this role
        if not settings.enabled:
            logger.info(f"LLM for role '{role}' is disabled in configuration. Returning None.")
            return None
            
        # Check if model_name is configured
        if settings.model_name is None:
            logger.warning(f"Model name for LLM role '{role}' is missing. Returning None.")
            return None

        api_key = settings.api_key # Get API key via property
        if not api_key:
            raise ValueError(f"API key for LLM role '{role}' (env var: {settings.api_key_env_var}) is missing.")

        provider_name = settings.provider.lower()
        logger.info(f"Initializing LLM for role '{role}' using provider '{provider_name}' and model '{settings.model_name}'...")

        llm_instance: LLM
        try:
            # --- Add logic for different providers ---
            if provider_name == "groq":
                llm_instance = Groq(
                    model=settings.model_name,
                    api_key=api_key,
                    request_timeout=settings.request_timeout,
                    # Pass additional kwargs if needed and supported by Groq class
                    **settings.additional_kwargs
                )
            elif provider_name == "openai":
                 llm_instance = OpenAI(
                     model=settings.model_name,
                     api_key=api_key,
                     temperature=settings.temperature,
                     max_tokens=settings.max_tokens,
                     request_timeout=settings.request_timeout,
                     max_retries=settings.max_retries,
                     # Pass additional kwargs if needed
                     **settings.additional_kwargs
                 )
            # --- Add other providers here (Anthropic, HuggingFace, etc.) ---
            # elif provider_name == "huggingface":
            #     from llama_index.llms.huggingface import HuggingFaceLLM
            #     llm_instance = HuggingFaceLLM(...) # Configure as needed
            else:
                raise ValueError(f"Unsupported LLM provider configured: {provider_name}")

            self._llm_instances[role] = llm_instance
            logger.info(f"Successfully initialized LLM for role '{role}'.")
            return llm_instance

        except Exception as e:
            logger.error(f"Failed to initialize LLM for role '{role}': {e}", exc_info=True)
            raise

    def get_metadata_llm(self) -> Optional[LLM]:
        """Get LLM configured for metadata extraction."""
        return self._get_llm_instance("metadata_llm")

    def get_query_llm(self) -> Optional[LLM]:
        """Get LLM configured for query processing/response synthesis."""
        return self._get_llm_instance("query_llm")

    def get_coding_llm(self) -> Optional[LLM]:
        """Get LLM configured for code generation/analysis (if configured)."""
        try:
            return self._get_llm_instance("coding_llm")
        except ValueError:
            logger.warning("Coding LLM not configured, returning None.")
            return None