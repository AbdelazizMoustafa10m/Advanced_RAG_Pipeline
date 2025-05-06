# --- query/transformers/rewriting.py ---
import logging
from typing import List, Optional, Union, Dict, Any

from llama_index.core.llms import LLM
from query.transformers.base import QueryTransformer
from core.config import QueryTransformationConfig

logger = logging.getLogger(__name__)


class LLMQueryRewriter(QueryTransformer):
    """Query rewriter using LLMs to improve query effectiveness.
    
    This rewriter uses an LLM to reformulate queries to improve 
    retrieval by making them more specific, clear, or comprehensive.
    """
    
    # Prompt templates for different rewriting techniques
    PROMPT_TEMPLATES = {
        "instruct": """
        Your task is to rewrite the following query to make it more effective for retrieval from a knowledge base.
        Make it more specific, clear, and comprehensive without changing the user's intent.
        
        Original query: {query}
        
        Rewritten query:
        """,
        
        "reflexion": """
        Consider this query: {query}
        
        Think about what the user is really looking for. What are the key concepts? What specific information would satisfy this query?
        
        Now, rewrite the query to make it more precise and effective for retrieval from a knowledge base.
        
        Rewritten query:
        """,
        
        "step_back": """
        Original query: {query}
        
        Before answering this specific question, what broader context or concepts should be understood?
        Think about what high-level knowledge would help answer this query more effectively.
        
        Now, rewrite the query to include both the specific question and the relevant broader context.
        
        Rewritten query:
        """
    }
    
    def __init__(
        self,
        llm: LLM,
        config: Optional[QueryTransformationConfig] = None,
    ):
        """Initialize LLM query rewriter.
        
        Args:
            llm: LLM for rewriting queries
            config: Optional query transformation configuration
        """
        super().__init__(config or QueryTransformationConfig())
        self.llm = llm
        self.technique = self.config.rewriting_technique
        self.iterations = self.config.query_iterations
        
        # Select the appropriate prompt template
        if self.technique in self.PROMPT_TEMPLATES:
            self.prompt_template = self.PROMPT_TEMPLATES[self.technique]
        else:
            logger.warning(f"Unknown rewriting technique: {self.technique}, using 'instruct' instead")
            self.prompt_template = self.PROMPT_TEMPLATES["instruct"]
    
    def transform(self, query: str, **kwargs) -> Dict[str, Any]:
        """Transform a query using LLM rewriting.
        
        Args:
            query: The original query string
            **kwargs: Additional arguments for the transformer
            
        Returns:
            Dictionary with original query and rewritten query
        """
        try:
            current_query = query
            iterations = []
            
            # Perform multiple iterations of rewriting if configured
            for i in range(self.iterations):
                # Format the prompt for query rewriting
                prompt = self.prompt_template.format(query=current_query)
                
                # Generate rewritten query
                response = self.llm.complete(prompt)
                rewritten_query = response.text.strip()
                
                # Update the current query for the next iteration
                current_query = rewritten_query
                iterations.append(rewritten_query)
                
                logger.info(f"Iteration {i+1}: Rewrote query from '{query}' to '{rewritten_query}'")
            
            # Return both the original query and rewritten queries
            return {
                "original_query": query,
                "rewritten_query": current_query,
                "iterations": iterations
            }
        except Exception as e:
            logger.error(f"Error in LLM query rewriting: {str(e)}")
            # Fall back to original query
            return {
                "original_query": query,
                "rewritten_query": query,
                "iterations": []
            }
    
    async def atransform(self, query: str, **kwargs) -> Dict[str, Any]:
        """Asynchronously transform a query using LLM rewriting.
        
        Args:
            query: The original query string
            **kwargs: Additional arguments for the transformer
            
        Returns:
            Dictionary with original query and rewritten query
        """
        try:
            current_query = query
            iterations = []
            
            # Perform multiple iterations of rewriting if configured
            for i in range(self.iterations):
                # Format the prompt for query rewriting
                prompt = self.prompt_template.format(query=current_query)
                
                # Generate rewritten query
                response = await self.llm.acomplete(prompt)
                rewritten_query = response.text.strip()
                
                # Update the current query for the next iteration
                current_query = rewritten_query
                iterations.append(rewritten_query)
                
                logger.info(f"Iteration {i+1}: Rewrote query from '{query}' to '{rewritten_query}'")
            
            # Return both the original query and rewritten queries
            return {
                "original_query": query,
                "rewritten_query": current_query,
                "iterations": iterations
            }
        except Exception as e:
            logger.error(f"Error in async LLM query rewriting: {str(e)}")
            # Fall back to original query
            return {
                "original_query": query,
                "rewritten_query": query,
                "iterations": []
            }
