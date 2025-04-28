# --- indexing/query_engine.py ---

from typing import Dict, Any, Optional, Tuple, List
import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from core.interfaces import IQueryProcessor
from core.config import QueryConfig
from llm.providers import DefaultLLMProvider

logger = logging.getLogger(__name__)


class QueryEngine(IQueryProcessor):
    """Query engine for the unified RAG system."""
    
    def __init__(
        self, 
        index: VectorStoreIndex,
        config: Optional[QueryConfig] = None,
        llm_provider: Optional[DefaultLLMProvider] = None
    ):
        """Initialize query engine.
        
        Args:
            index: Vector store index
            config: Optional query configuration
            llm_provider: Optional LLM provider
        """
        self.index = index
        self.config = config or QueryConfig()
        self.llm_provider = llm_provider or DefaultLLMProvider(config=None)
        
        # Initialize query engine with LLM
        self.llm = self.llm_provider.get_query_llm()
        self.query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=self.config.similarity_top_k,
            response_mode=self.config.response_mode
        )
    
    def process_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Process a query and extract metadata filters.
        
        Args:
            query: The query string
            
        Returns:
            Tuple of (processed_query, filters_dict)
        """
        filters = {}
        
        # Extract language filter if enabled
        if self.config.detect_language_in_query:
            try:
                language_prompt = f"If the query mentions a specific programming language, extract just the language name. If no language is mentioned, reply with 'None':\n{query}"
                language = self.llm.complete(language_prompt).text.strip().lower()
                
                if language not in ["none", "n/a", ""]:
                    filters["language"] = language
                    logger.info(f"Detected language filter: {language}")
            except Exception as e:
                logger.error(f"Error detecting language in query: {str(e)}")
        
        # Detect document type preference (code vs document)
        try:
            doc_type_prompt = f"""Based on the query, determine if the user is looking for:
            1. Code examples or implementation (respond with "code")
            2. Documentation, explanations, or concepts (respond with "document")
            3. Mixed or unclear (respond with "mixed")
            
            Query: {query}
            
            Response (just one word - "code", "document", or "mixed"):"""
            
            doc_type = self.llm.complete(doc_type_prompt).text.strip().lower()
            
            if doc_type in ["code", "document"]:
                filters["node_type"] = doc_type
                logger.info(f"Detected document type filter: {doc_type}")
        except Exception as e:
            logger.error(f"Error detecting document type in query: {str(e)}")
        
        return query, filters
    
    def query(self, query: str, filters: Optional[Dict[str, Any]] = None):
        """Query the knowledge base.
        
        Args:
            query: The query string
            filters: Optional metadata filters
            
        Returns:
            Query response
        """
        try:
            # Apply filters if provided
            filtered_query_engine = self.query_engine
            if filters:
                # Create a filtered retriever
                retriever = self.index.as_retriever(
                    similarity_top_k=self.config.similarity_top_k,
                    filters=filters
                )
                
                # Create a new query engine with the filtered retriever
                filtered_query_engine = self.index.as_query_engine(
                    llm=self.llm,
                    retriever=retriever,
                    response_mode=self.config.response_mode
                )
            
            # Execute query
            response = filtered_query_engine.query(query)
            logger.info(f"Query executed with {len(response.source_nodes)} source nodes")
            
            return response
        
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
