# --- query/rerankers/reranker.py ---
import logging
from typing import List, Optional, Dict, Any, Union
import heapq
import time

from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.llm_rerank import LLMRerank as LlamaIndexLLMRerank

from core.interfaces import IReranker
from core.config import RerankerConfig

logger = logging.getLogger(__name__)


class Reranker(IReranker):
    """Base class for reranking components."""
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        """Initialize reranker with configuration.
        
        Args:
            config: Optional reranker configuration
        """
        self.config = config or RerankerConfig()
    
    def rerank(self, query_bundle_or_str: Union[QueryBundle, str], nodes: List[NodeWithScore], **kwargs) -> List[NodeWithScore]:
        """Rerank a list of retrieved nodes.
        
        Base implementation just returns the original nodes.
        
        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for reranking
            
        Returns:
            Reranked list of nodes
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str
            
        return nodes
    
    async def arerank(self, query_bundle_or_str: Union[QueryBundle, str], nodes: List[NodeWithScore], **kwargs) -> List[NodeWithScore]:
        """Asynchronously rerank a list of retrieved nodes.
        
        Base implementation calls the synchronous method.
        
        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for reranking
            
        Returns:
            Reranked list of nodes
        """
        return self.rerank(query_bundle_or_str, nodes, **kwargs)


class SemanticReranker(Reranker):
    """Reranker using semantic similarity models.
    
    This reranker uses cross-encoder models to directly score
    query-document pairs for more accurate relevance assessment.
    """
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        """Initialize semantic reranker.
        
        Args:
            config: Optional reranker configuration
        """
        super().__init__(config)
        self.model_name = self.config.semantic_model
        self.rerank_top_n = self.config.rerank_top_n
        
        # Initialize cross-encoder model
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Initializing semantic reranker with model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            self.model_initialized = True
            
            logger.info("Semantic reranker initialized successfully")
            
        except ImportError:
            logger.warning("SentenceTransformers package not available. Install with 'pip install sentence-transformers'")
            self.model_initialized = False
    
    def rerank(self, query_bundle_or_str: Union[QueryBundle, str], nodes: List[NodeWithScore], **kwargs) -> List[NodeWithScore]:
        """Rerank nodes using semantic cross-encoder.
        
        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for reranking
            
        Returns:
            Reranked list of nodes
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str
            
        if not self.model_initialized:
            logger.warning("Semantic reranker not properly initialized")
            return nodes
        
        try:
            start_time = time.time()
            
            # Limit to top N nodes for efficiency
            nodes_to_rerank = nodes[:self.rerank_top_n] if len(nodes) > self.rerank_top_n else nodes
            
            # Create query-document pairs for scoring
            query_doc_pairs = [(query_bundle.query_str, node.node.get_content()) for node in nodes_to_rerank]
            
            # Score the pairs
            scores = self.model.predict(query_doc_pairs)
            
            # Create reranked nodes
            reranked_nodes = []
            for i, node in enumerate(nodes_to_rerank):
                reranked_nodes.append(
                    NodeWithScore(
                        node=node.node,
                        score=float(scores[i])
                    )
                )
            
            # Sort by new scores
            reranked_nodes.sort(key=lambda x: x.score if x.score is not None else 0.0, reverse=True)
            
            # Combine with any remaining nodes (retain original order of non-reranked nodes)
            if len(nodes) > self.rerank_top_n:
                remaining_nodes = nodes[self.rerank_top_n:]
                reranked_nodes.extend(remaining_nodes)
            
            end_time = time.time()
            logger.info(f"Reranked {len(nodes_to_rerank)} nodes in {end_time - start_time:.2f} seconds")
            
            return reranked_nodes
            
        except Exception as e:
            logger.error(f"Error in semantic reranking: {str(e)}")
            return nodes
    
    async def arerank(self, query_bundle_or_str: Union[QueryBundle, str], nodes: List[NodeWithScore], **kwargs) -> List[NodeWithScore]:
        """Asynchronously rerank nodes using semantic cross-encoder.
        
        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for reranking
            
        Returns:
            Reranked list of nodes
        """
        # Just call the synchronous method as this doesn't support async
        return self.rerank(query_bundle_or_str, nodes, **kwargs)


class LLMReranker(Reranker):
    """Reranker using LLMs to score query-document relevance.
    
    This reranker uses an LLM to evaluate the relevance of each
    retrieved document to the query.
    """
    
    def __init__(
        self,
        llm: LLM,
        config: Optional[RerankerConfig] = None,
    ):
        """Initialize LLM reranker.
        
        Args:
            llm: LLM for relevance scoring
            config: Optional reranker configuration
        """
        super().__init__(config)
        self.llm = llm
        self.rerank_top_n = self.config.rerank_top_n
        self.batch_size = self.config.batch_size
        
        # Initialize LlamaIndex LLMRerank
        self.llamaindex_reranker = LlamaIndexLLMRerank(
            llm=self.llm,
            choice_batch_size=self.batch_size,
            top_n=self.rerank_top_n,
        )
    
    def rerank(self, query_bundle_or_str: Union[QueryBundle, str], nodes: List[NodeWithScore], **kwargs) -> List[NodeWithScore]:
        """Rerank nodes using LLM relevance scoring.
        
        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for reranking
            
        Returns:
            Reranked list of nodes
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str
            
        try:
            start_time = time.time()
            
            # Use LlamaIndex's implementation
            reranked_nodes = self.llamaindex_reranker.postprocess_nodes(
                nodes,
                query_bundle
            )
            
            end_time = time.time()
            logger.info(f"LLM reranked {len(nodes)} nodes in {end_time - start_time:.2f} seconds")
            
            return reranked_nodes
            
        except Exception as e:
            logger.error(f"Error in LLM reranking: {str(e)}")
            return nodes
    
    async def arerank(self, query_bundle_or_str: Union[QueryBundle, str], nodes: List[NodeWithScore], **kwargs) -> List[NodeWithScore]:
        """Asynchronously rerank nodes using LLM relevance scoring.
        
        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for reranking
            
        Returns:
            Reranked list of nodes
        """
        # Just call the synchronous method as LlamaIndex's implementation doesn't yet support async
        return self.rerank(query_bundle_or_str, nodes, **kwargs)


class CohereReranker(Reranker):
    """Reranker using Cohere's Rerank API.
    
    This reranker uses Cohere's specialized reranking models for
    improved relevance assessment.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[RerankerConfig] = None,
        model_name: str = "rerank-english-v2.0",
    ):
        """Initialize Cohere reranker.
        
        Args:
            api_key: Cohere API key
            config: Optional reranker configuration
            model_name: Cohere rerank model name
        """
        super().__init__(config)
        self.api_key = api_key or self.config.cohere_api_key
        self.model_name = model_name
        self.rerank_top_n = self.config.rerank_top_n
        
        # Initialize Cohere client
        try:
            from llama_index.postprocessor.cohere_rerank import CohereRerank
            
            logger.info(f"Initializing Cohere reranker with model: {self.model_name}")
            self.llamaindex_reranker = CohereRerank(
                api_key=self.api_key,
                top_n=self.rerank_top_n,
                model_name=self.model_name
            )
            self.cohere_initialized = True
            
            logger.info("Cohere reranker initialized successfully")
            
        except ImportError:
            logger.warning("Cohere package not available. Install with 'pip install llama-index-postprocessor-cohere-rerank'")
            self.cohere_initialized = False
    
    def rerank(self, query_bundle_or_str: Union[QueryBundle, str], nodes: List[NodeWithScore], **kwargs) -> List[NodeWithScore]:
        """Rerank nodes using Cohere's reranking API.
        
        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for reranking
            
        Returns:
            Reranked list of nodes
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str
            
        if not self.cohere_initialized:
            logger.warning("Cohere reranker not properly initialized")
            return nodes
        
        try:
            start_time = time.time()
            
            # Use LlamaIndex's implementation
            reranked_nodes = self.llamaindex_reranker.postprocess_nodes(
                nodes,
                query_bundle
            )
            
            end_time = time.time()
            logger.info(f"Cohere reranked {len(nodes)} nodes in {end_time - start_time:.2f} seconds")
            
            return reranked_nodes
            
        except Exception as e:
            logger.error(f"Error in Cohere reranking: {str(e)}")
            return nodes
    
    async def arerank(self, query_bundle_or_str: Union[QueryBundle, str], nodes: List[NodeWithScore], **kwargs) -> List[NodeWithScore]:
        """Asynchronously rerank nodes using Cohere's reranking API.
        
        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for reranking
            
        Returns:
            Reranked list of nodes
        """
        # Just call the synchronous method as this doesn't support async yet
        return self.rerank(query_bundle_or_str, nodes, **kwargs)