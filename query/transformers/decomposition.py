# --- query/transformers/decomposition.py ---
import logging
from typing import List, Optional, Union, Dict, Any

from llama_index.core.llms import LLM
from llama_index.core.indices.query.query_transform.base import StepDecomposeQueryTransform
from llama_index.core.schema import QueryBundle

from query.transformers.base import QueryTransformer
from core.config import QueryTransformationConfig

logger = logging.getLogger(__name__)


class QueryDecomposer(QueryTransformer):
    """Decomposes complex queries into sub-questions.
    
    This transformer breaks down complex queries into simpler 
    sub-questions that can be processed more effectively.
    """
    
    def __init__(
        self,
        llm: LLM,
        config: Optional[QueryTransformationConfig] = None,
    ):
        """Initialize query decomposer.
        
        Args:
            llm: LLM for decomposing queries
            config: Optional query transformation configuration
        """
        super().__init__(config or QueryTransformationConfig())
        self.llm = llm
        
        # Initialize LlamaIndex decomposer based on mode
        if self.config.decomposition_mode == "step":
            self.decomposer = StepDecomposeQueryTransform(
                llm=self.llm,
                verbose=self.config.verbose if hasattr(self.config, 'verbose') else False
            )
        else:
            # Use custom implementation for sub_question mode
            self.decomposer = None
            self.max_sub_questions = self.config.max_sub_questions
            
            # Sub-question decomposition prompt
            self.prompt_template = """
            Given a complex question, break it down into {max_sub_questions} or fewer simpler sub-questions 
            that together would help answer the complex question.
            These questions should be answerable independently of each other.
            
            Complex question: {query}
            
            Sub-questions:
            """
    
    def transform(self, query: str, **kwargs) -> Dict[str, Any]:
        """Transform a query by decomposing it.
        
        Args:
            query: The original query string
            **kwargs: Additional arguments for the transformer
            
        Returns:
            Dictionary with original query and sub-questions
        """
        try:
            if self.decomposer:  # Step decomposition using LlamaIndex
                query_bundle = QueryBundle(query_str=query)
                transformed_query = self.decomposer.run(query_bundle)
                
                return {
                    "original_query": query,
                    "transformed_query": transformed_query.query_str,
                    "sub_questions": [transformed_query.query_str]
                }
            else:  # Sub-question decomposition using custom implementation
                # Format the prompt
                prompt = self.prompt_template.format(
                    max_sub_questions=self.max_sub_questions,
                    query=query
                )
                
                # Generate sub-questions
                response = self.llm.complete(prompt)
                
                # Extract sub-questions
                sub_questions = [
                    line.strip().replace("- ", "")
                    for line in response.text.split("\n")
                    if line.strip() and not line.strip().isdigit() and "-" in line
                ]
                
                if not sub_questions:
                    sub_questions = [query]
                    
                return {
                    "original_query": query,
                    "sub_questions": sub_questions
                }
                
        except Exception as e:
            logger.error(f"Error in query decomposition: {str(e)}")
            # Fall back to original query
            return {
                "original_query": query,
                "sub_questions": [query]
            }
    
    async def atransform(self, query: str, **kwargs) -> Dict[str, Any]:
        """Asynchronously transform a query by decomposing it.
        
        Args:
            query: The original query string
            **kwargs: Additional arguments for the transformer
            
        Returns:
            Dictionary with original query and sub-questions
        """
        # Most LlamaIndex transformers don't have async interfaces yet,
        # so we call the synchronous method
        return self.transform(query, **kwargs)