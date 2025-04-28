# --- llm/cache.py ---

import os
import json
import hashlib
from typing import Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMCache:
    """Cache for LLM responses to avoid redundant API calls."""
    
    def __init__(self, cache_dir: str = "./.cache/llm"):
        """Initialize LLM cache.
        
        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, prompt: str, model: str) -> str:
        """Generate a unique cache key for a prompt and model.
        
        Args:
            prompt: The prompt text
            model: The model name
            
        Returns:
            Cache key string
        """
        # Create a hash of the prompt and model
        combined = f"{prompt}_{model}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        """Get the file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            File path for the cache entry
        """
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def get(self, prompt: str, model: str) -> Optional[str]:
        """Get cached response for a prompt and model.
        
        Args:
            prompt: The prompt text
            model: The model name
            
        Returns:
            Cached response or None if not found
        """
        key = self._get_cache_key(prompt, model)
        cache_path = self._get_cache_path(key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                return cache_data.get("response")
            except Exception as e:
                logger.error(f"Error reading cache: {str(e)}")
                return None
        
        return None
    
    def put(self, prompt: str, model: str, response: str) -> None:
        """Save response to cache.
        
        Args:
            prompt: The prompt text
            model: The model name
            response: The response to cache
        """
        key = self._get_cache_key(prompt, model)
        cache_path = self._get_cache_path(key)
        
        try:
            cache_data = {
                "prompt": prompt,
                "model": model,
                "response": response
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.error(f"Error writing to cache: {str(e)}")
