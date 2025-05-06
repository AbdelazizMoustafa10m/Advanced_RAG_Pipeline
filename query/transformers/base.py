# query/transformers/base.py
from core.interfaces import IQueryTransformer
from typing import List, Optional, Union, Dict, Any
from llama_index.core.llms import LLM
from llama_index.core.schema import QueryBundle
from core.config import QueryTransformationConfig

class QueryTransformer(IQueryTransformer):
    """Base class for query transformation components."""
    
    def __init__(self, config: Optional[QueryTransformationConfig] = None):
        """Initialize query transformer with configuration.
        
        Args:
            config: Optional query transformation configuration
        """
        self.config = config or QueryTransformationConfig()
    
    def transform(self, query: str, **kwargs) -> Union[str, Dict[str, Any]]:
        """Transform a query string.
        
        Args:
            query: The original query string
            **kwargs: Additional arguments for the transformer
            
        Returns:
            Transformed query or transformation results
        """
        # Base implementation just returns the original query
        return query
    
    async def atransform(self, query: str, **kwargs) -> Union[str, Dict[str, Any]]:
        """Asynchronously transform a query string.
        
        Args:
            query: The original query string
            **kwargs: Additional arguments for the transformer
            
        Returns:
            Transformed query or transformation results
        """
        # Base implementation calls the synchronous method
        return self.transform(query, **kwargs)