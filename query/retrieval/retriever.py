# --- query/retrieval/retriever.py ---
import logging
from typing import List, Optional, Dict, Any, Tuple, Union
import heapq
from dataclasses import dataclass
import time

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.embeddings import BaseEmbedding

from core.interfaces import IRetriever
from core.config import RetrievalConfig

logger = logging.getLogger(__name__)


class EnhancedRetriever(IRetriever):
    """Base class for enhanced retrieval components."""
    
    def __init__(
        self,
        index: VectorStoreIndex,
        config: Optional[RetrievalConfig] = None,
        callback_manager: Optional[CallbackManager] = None,
        embed_model: Optional[BaseEmbedding] = None,
    ):
        """Initialize enhanced retriever with configuration.
        
        Args:
            index: Vector store index for retrieval
            config: Optional retrieval configuration
            callback_manager: Optional callback manager
            embed_model: Optional embedding model to use for query embedding
        """
        self.index = index
        self.config = config or RetrievalConfig()
        self.callback_manager = callback_manager
        self.embed_model = embed_model
        
        # Initialize default vector retriever
        self.vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.config.similarity_top_k,
            callback_manager=self.callback_manager,
            embed_model=self.embed_model  # Pass embedding model explicitly
        )
    
    def retrieve(self, query_bundle_or_str: Union[QueryBundle, str], **kwargs) -> List[NodeWithScore]:
        """Retrieve relevant nodes for a query.
        
        Args:
            query_bundle_or_str: The query string or QueryBundle
            **kwargs: Additional arguments for retrieval
            
        Returns:
            List of nodes with relevance scores
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str
            
        # Override top_k if provided in kwargs
        similarity_top_k = kwargs.get("top_k", self.config.similarity_top_k)
        
        # Update top_k for retriever
        self.vector_retriever.similarity_top_k = similarity_top_k
            
        # Default implementation uses basic vector retrieval
        start_time = time.time()
        nodes = self.vector_retriever.retrieve(query_bundle)
        end_time = time.time()
        
        logger.info(f"Retrieved {len(nodes)} nodes in {end_time - start_time:.2f} seconds")
        return nodes
    
    async def aretrieve(self, query_bundle_or_str: Union[QueryBundle, str], **kwargs) -> List[NodeWithScore]:
        """Asynchronously retrieve relevant nodes for a query.
        
        Args:
            query_bundle_or_str: The query string or QueryBundle
            **kwargs: Additional arguments for retrieval
            
        Returns:
            List of nodes with relevance scores
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str
            
        # Override top_k if provided in kwargs
        similarity_top_k = kwargs.get("top_k", self.config.similarity_top_k)
        
        # Update top_k for retriever
        self.vector_retriever.similarity_top_k = similarity_top_k
            
        # Default implementation uses basic vector retrieval
        start_time = time.time()
        nodes = await self.vector_retriever.aretrieve(query_bundle)
        end_time = time.time()
        
        logger.info(f"Retrieved {len(nodes)} nodes asynchronously in {end_time - start_time:.2f} seconds")
        return nodes
    
    @staticmethod
    def normalize_scores(nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Normalize scores of nodes to the range [0, 1].
        
        Args:
            nodes: List of nodes with scores
            
        Returns:
            List of nodes with normalized scores
        """
        if not nodes:
            return []
        
        # Find min and max scores
        scores = [node.score for node in nodes if node.score is not None]
        if not scores:
            return nodes
        
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            return nodes
        
        # Create new nodes with normalized scores
        normalized_nodes = []
        for node in nodes:
            if node.score is not None:
                normalized_score = (node.score - min_score) / (max_score - min_score)
                normalized_nodes.append(
                    NodeWithScore(
                        node=node.node,
                        score=normalized_score
                    )
                )
            else:
                normalized_nodes.append(node)
        
        return normalized_nodes


class HybridRetriever(EnhancedRetriever):
    """Hybrid retriever combining vector and keyword search.
    
    This retriever combines the strengths of semantic (vector) search
    and keyword-based search for improved retrieval performance.
    """
    
    def __init__(
        self,
        index: VectorStoreIndex,
        nodes: List[TextNode],
        config: Optional[RetrievalConfig] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        """Initialize hybrid retriever.
        
        Args:
            index: Vector store index for semantic search
            nodes: List of text nodes for keyword search
            config: Optional retrieval configuration
            callback_manager: Optional callback manager
        """
        super().__init__(index, config, callback_manager)
        self.nodes = nodes
        
        # Initialize keyword retriever using LlamaIndex components
        try:
            from llama_index.retrievers.bm25 import BM25Retriever
            self.keyword_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=self.config.keyword_top_k
            )
            logger.info("Initialized BM25 keyword retriever")
        except ImportError:
            # Fallback to basic keyword retriever
            from llama_index.core.retrievers import KeywordTableSimpleRetriever
            try:
                # Try to create KeywordTableSimpleRetriever from index
                self.keyword_retriever = KeywordTableSimpleRetriever(
                    index=self.index,
                    similarity_top_k=self.config.keyword_top_k,
                )
                logger.info("Initialized KeywordTableSimpleRetriever")
            except Exception as e:
                logger.warning(f"Could not initialize keyword retriever: {str(e)}")
                self.keyword_retriever = None
        
        # Set alpha parameter for hybrid search
        self.alpha = self.config.hybrid_alpha
    
    def retrieve(self, query_bundle_or_str: Union[QueryBundle, str], **kwargs) -> List[NodeWithScore]:
        """Retrieve relevant nodes using hybrid search.
        
        Args:
            query_bundle_or_str: The query string or QueryBundle
            **kwargs: Additional arguments for retrieval
            
        Returns:
            List of nodes with relevance scores
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str
            
        try:
            # Override top_k if provided in kwargs
            top_k = kwargs.get("top_k", self.config.similarity_top_k)
            
            start_time = time.time()
            
            # Get vector search results
            self.vector_retriever.similarity_top_k = top_k
            vector_results = self.vector_retriever.retrieve(query_bundle)
            
            # Skip hybrid search if no keyword retriever
            if not self.keyword_retriever:
                logger.warning("Keyword retriever not available, using vector retrieval only")
                return vector_results
            
            # Get keyword search results
            self.keyword_retriever.similarity_top_k = top_k
            keyword_results = self.keyword_retriever.retrieve(query_bundle)
            
            # Normalize scores
            vector_results = self.normalize_scores(vector_results)
            keyword_results = self.normalize_scores(keyword_results)
            
            # Create a mapping from node ID to node and scores
            node_map = {}
            
            # Add vector results to the map
            for node in vector_results:
                node_id = node.node.node_id
                vector_score = node.score if node.score is not None else 0.0
                
                node_map[node_id] = {
                    "node": node.node,
                    "vector_score": vector_score,
                    "keyword_score": 0.0
                }
            
            # Add keyword results to the map or update existing entries
            for node in keyword_results:
                node_id = node.node.node_id
                keyword_score = node.score if node.score is not None else 0.0
                
                if node_id in node_map:
                    node_map[node_id]["keyword_score"] = keyword_score
                else:
                    node_map[node_id] = {
                        "node": node.node,
                        "vector_score": 0.0,
                        "keyword_score": keyword_score
                    }
            
            # Combine results based on hybrid_mode
            if self.config.hybrid_mode == "AND":
                # Only keep nodes that appear in both result sets
                filtered_node_map = {
                    node_id: data for node_id, data in node_map.items()
                    if data["vector_score"] > 0 and data["keyword_score"] > 0
                }
                node_map = filtered_node_map
            
            # Combine scores using the alpha parameter
            # score = alpha * vector_score + (1 - alpha) * keyword_score
            combined_results = []
            for node_id, data in node_map.items():
                combined_score = (
                    self.alpha * data["vector_score"] +
                    (1.0 - self.alpha) * data["keyword_score"]
                )
                
                combined_results.append(
                    NodeWithScore(
                        node=data["node"],
                        score=combined_score
                    )
                )
            
            # Sort by combined score in descending order
            combined_results.sort(key=lambda x: x.score if x.score is not None else 0.0, reverse=True)
            
            # Limit to top-k results
            results = combined_results[:top_k]
            
            end_time = time.time()
            logger.info(f"Retrieved {len(results)} nodes using hybrid search in {end_time - start_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            # Fall back to vector retrieval
            return super().retrieve(query_bundle, **kwargs)
    
    async def aretrieve(self, query_bundle_or_str: Union[QueryBundle, str], **kwargs) -> List[NodeWithScore]:
        """Asynchronously retrieve relevant nodes using hybrid search.
        
        Args:
            query_bundle_or_str: The query string or QueryBundle
            **kwargs: Additional arguments for retrieval
            
        Returns:
            List of nodes with relevance scores
        """
        # For now, we'll use the synchronous implementation
        # as most LlamaIndex retrievers don't have async interfaces yet
        return self.retrieve(query_bundle_or_str, **kwargs)


class EnsembleRetriever(EnhancedRetriever):
    """Ensemble retriever that combines results from multiple retrievers.
    
    This retriever allows for the combination of different retrieval
    strategies to improve overall performance.
    """
    
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: Optional[List[float]] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        """Initialize ensemble retriever.
        
        Args:
            retrievers: List of retrievers to ensemble
            weights: Optional weights for each retriever
            config: Optional retrieval configuration
        """
        self.retrievers = retrievers
        self.config = config or RetrievalConfig()
        
        # Initialize weights (equal by default)
        if weights is None:
            self.weights = [1.0 / len(retrievers)] * len(retrievers)
        else:
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
    
    def retrieve(self, query_bundle_or_str: Union[QueryBundle, str], **kwargs) -> List[NodeWithScore]:
        """Retrieve relevant nodes using ensemble of retrievers.
        
        Args:
            query_bundle_or_str: The query string or QueryBundle
            **kwargs: Additional arguments for retrieval
            
        Returns:
            List of nodes with relevance scores
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str
            
        try:
            start_time = time.time()
            
            # Get results from each retriever
            all_results = []
            for i, retriever in enumerate(self.retrievers):
                top_k = kwargs.get("top_k", self.config.similarity_top_k)
                
                # Set top_k if retriever supports it
                if hasattr(retriever, "similarity_top_k"):
                    retriever.similarity_top_k = top_k
                
                results = retriever.retrieve(query_bundle)
                results = EnhancedRetriever.normalize_scores(results)
                all_results.append(results)
            
            # Combine results using weights
            node_map = {}
            
            for i, results in enumerate(all_results):
                weight = self.weights[i]
                
                for node in results:
                    node_id = node.node.node_id
                    weighted_score = (node.score if node.score is not None else 0.0) * weight
                    
                    if node_id in node_map:
                        node_map[node_id]["score"] += weighted_score
                    else:
                        node_map[node_id] = {
                            "node": node.node,
                            "score": weighted_score
                        }
            
            # Create combined results
            combined_results = [
                NodeWithScore(
                    node=data["node"],
                    score=data["score"]
                )
                for node_id, data in node_map.items()
            ]
            
            # Sort by combined score in descending order
            combined_results.sort(key=lambda x: x.score if x.score is not None else 0.0, reverse=True)
            
            # Limit to top-k results
            top_k = kwargs.get("top_k", self.config.similarity_top_k)
            results = combined_results[:top_k]
            
            end_time = time.time()
            logger.info(f"Retrieved {len(results)} nodes using ensemble retrieval in {end_time - start_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in ensemble retrieval: {str(e)}")
            # Fall back to first retriever
            if self.retrievers:
                return self.retrievers[0].retrieve(query_bundle)
            else:
                return []
    
    async def aretrieve(self, query_bundle_or_str: Union[QueryBundle, str], **kwargs) -> List[NodeWithScore]:
        """Asynchronously retrieve relevant nodes using ensemble of retrievers.
        
        Args:
            query_bundle_or_str: The query string or QueryBundle
            **kwargs: Additional arguments for retrieval
            
        Returns:
            List of nodes with relevance scores
        """
        # For now, we'll use the synchronous implementation
        # as most LlamaIndex retrievers don't have async interfaces yet
        return self.retrieve(query_bundle_or_str, **kwargs)