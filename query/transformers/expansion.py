# --- query/transformers/expansion.py ---
import logging
from typing import List, Optional, Union, Dict, Any

from llama_index.core.schema import NodeWithScore
from llama_index.core.llms import LLM
from llama_index.core.schema import QueryBundle
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.core.prompts import PromptTemplate

from core.interfaces import IQueryTransformer
from core.config import QueryTransformationConfig
from query.transformers.base import QueryTransformer

logger = logging.getLogger(__name__)

class HyDEQueryExpander(QueryTransformer):
    """Query expander using Hypothetical Document Embeddings (HyDE).
    
    HyDE works by:
    1. Using an LLM to generate a hypothetical document that would answer the query
    2. Embedding that document instead of/in addition to the query
    3. Using the document embedding for retrieval
    
    This can improve retrieval for complex or ambiguous queries.
    """
    
    def __init__(
        self,
        llm: LLM,
        config: Optional[QueryTransformationConfig] = None,
        **kwargs
    ):
        """Initialize HyDE query expander.
        
        Args:
            llm: LLM for generating hypothetical documents
            config: Optional query transformation configuration
        """
        super().__init__(config)
        self.llm = llm
        
        # Initialize LlamaIndex HyDE transformer
        self.hyde_transformer = HyDEQueryTransform(
            llm=self.llm,
            include_original=True,
            **kwargs
        )
    
    def transform(self, query: str, **kwargs) -> Dict[str, Any]:
        """Transform a query using HyDE.
        
        Args:
            query: The original query string
            **kwargs: Additional arguments for the transformer
            
        Returns:
            Dictionary with original query and hypothetical document
        """
        try:
            # Create query bundle
            query_bundle = QueryBundle(query_str=query)
            
            # Use LlamaIndex's HyDE implementation
            transformed_query_bundle = self.hyde_transformer.run(query_bundle)
            
            # Extract hypothetical document if present
            hypothetical_document = None
            if transformed_query_bundle.custom_embedding_strs:
                hypothetical_document = transformed_query_bundle.custom_embedding_strs[0]
            
            # Return transformation results in a standard format
            return {
                "original_query": query,
                "transformed_query": transformed_query_bundle.query_str,
                "hypothetical_document": hypothetical_document
            }
        except Exception as e:
            logger.error(f"Error in HyDE query expansion: {str(e)}")
            # Fall back to original query
            return {
                "original_query": query,
                "transformed_query": query,
                "hypothetical_document": None
            }
    
    async def atransform(self, query: str, **kwargs) -> Dict[str, Any]:
        """Asynchronously transform a query using HyDE.
        
        Args:
            query: The original query string
            **kwargs: Additional arguments for the transformer
            
        Returns:
            Dictionary with original query and hypothetical document
        """
        # Currently, LlamaIndex's HyDE doesn't have an async interface
        # so we call the synchronous method
        return self.transform(query, **kwargs)

    def _ensure_template_object(template):
        """Ensure template is a proper LlamaIndex template object."""
        if isinstance(template, str):
            return LlamaPromptTemplate(template)
        return template    