# --- query/transformers/expansion.py ---
import logging
from typing import List, Optional, Union, Dict, Any

from llama_index.core.schema import NodeWithScore
from llama_index.core.llms import LLM
from llama_index.core.schema import QueryBundle
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

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
    ):
        """Initialize HyDE query expander.
        
        Args:
            llm: LLM for generating hypothetical documents
            config: Optional query transformation configuration
        """
        super().__init__(config)
        self.llm = llm
        self.prompt_template = self.config.hyde_prompt_template
        
        # Create the HyDE transformer
        from llama_index.core.prompts import PromptTemplate as LlamaPromptTemplate
        hyde_prompt = LlamaPromptTemplate(template=self.prompt_template)
        
        self.hyde_transformer = HyDEQueryTransform(
            llm=self.llm,
            include_original=True,  # Include original query as fallback
            hyde_prompt=hyde_prompt
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
            logger.info(f"HyDE transformer processing query: '{query}'")
            
            # Generate a hypothetical document directly using the LLM
            from llama_index.core.llms import ChatMessage, MessageRole
            
            # Format the prompt template with the actual query
            formatted_prompt = self.prompt_template.format(query=query)
            logger.info(f"Using formatted HyDE prompt: '{formatted_prompt}'")
            
            # Create a system message to better guide the LLM
            messages = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content="You are a helpful assistant that generates detailed, factual passages to help with information retrieval. Your response should be a comprehensive passage about setting IO pins in code."
                ),
                ChatMessage(
                    role=MessageRole.USER,
                    content=formatted_prompt
                )
            ]
            
            # Generate the hypothetical document directly
            hyde_response = self.llm.chat(messages)
            hypothetical_document = hyde_response.message.content
            
            logger.info(f"HyDE generated document: '{hypothetical_document[:100]}...'")
            
            # Create a query bundle with the hypothetical document
            # In LlamaIndex, we need to use custom_embedding_strs instead of directly setting embedding_strs
            query_bundle = QueryBundle(
                query_str=query,
                custom_embedding_strs=[hypothetical_document]
            )
            
            # Return the transformation results
            return {
                "original_query": query,
                "transformed_query": query,  # Keep original query for synthesis
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