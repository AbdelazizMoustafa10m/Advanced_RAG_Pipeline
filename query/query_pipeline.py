# --- query/query_pipeline.py ---
import logging
import time
import os
import hashlib
import json
from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio

from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.base.response.schema import Response
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding

from core.interfaces import IQueryPipeline, IEmbedder
from core.config import QueryPipelineConfig
from query.transformers import QueryTransformer, HyDEQueryExpander, LLMQueryRewriter, QueryDecomposer
from query.retrieval import EnhancedRetriever, HybridRetriever, EnsembleRetriever
from query.rerankers import Reranker, SemanticReranker, LLMReranker, CohereReranker
from query.synthesis import (
    ResponseSynthesizer, 
    SimpleResponseSynthesizer, 
    RefineResponseSynthesizer, 
    TreeSynthesizer, 
    CompactResponseSynthesizer
)

logger = logging.getLogger(__name__)


class QueryPipeline(IQueryPipeline):
    """Advanced RAG query pipeline with query transformation, retrieval, reranking, and synthesis."""
    
    def __init__(
        self,
        config: QueryPipelineConfig,
        index: VectorStoreIndex,
        llm: LLM,
        nodes: Optional[List] = None,
        embedder: Optional[IEmbedder] = None,
    ):
        """Initialize the query pipeline.
        
        Args:
            config: Query pipeline configuration
            index: Vector store index for retrieval
            llm: LLM for query transformation and response synthesis
            nodes: Optional list of text nodes for additional retrieval methods
            embedder: Optional embedder for query embedding
        """
        self.config = config
        self.index = index
        self.llm = llm
        self.nodes = nodes
        
        # Get or create embedder
        if embedder is None:
            from embedders.embedder_factory import EmbedderFactory
            self.embedder = EmbedderFactory.create_embedder()
        else:
            self.embedder = embedder
        
        # Initialize pipeline components based on configuration
        self.query_transformers = self._init_query_transformers()
        self.retriever = self._init_retriever()
        self.reranker = self._init_reranker()
        self.synthesizer = self._init_synthesizer()
        
        # Create cache directory if caching is enabled
        if self.config.cache_results and self.config.cache_dir:
            os.makedirs(self.config.cache_dir, exist_ok=True)
    
    def _init_query_transformers(self) -> List[QueryTransformer]:
        """Initialize query transformation components based on configuration.
        
        Returns:
            List of query transformers
        """
        transformers = []
        
        # Add transformers based on config
        if self.config.transformation.enable_query_expansion:
            if self.config.transformation.use_hyde:
                # Add HyDE transformer
                transformers.append(
                    HyDEQueryExpander(
                        llm=self.llm,
                        config=self.config.transformation
                    )
                )
        
        if self.config.transformation.enable_query_rewriting:
            # Add query rewriter
            transformers.append(
                LLMQueryRewriter(
                    llm=self.llm,
                    config=self.config.transformation
                )
            )
            
        if self.config.transformation.enable_decomposition:
            # Add query decomposer
            transformers.append(
                QueryDecomposer(
                    llm=self.llm,
                    config=self.config.transformation
                )
            )
        
        logger.info(f"Initialized {len(transformers)} query transformers")
        return transformers
    
    def _init_retriever(self) -> EnhancedRetriever:
        """Initialize retrieval component based on configuration.
        
        Returns:
            Retriever instance
        """
        # Get embedding model from embedder
        embed_model = self.embedder.get_embedding_model()
        
        # Use default basic retriever if no special settings
        if self.config.retrieval.retriever_strategy == "vector":
            retriever = EnhancedRetriever(
                index=self.index,
                config=self.config.retrieval,
                embed_model=embed_model
            )
        elif self.config.retrieval.retriever_strategy == "hybrid" and self.nodes:
            # Use hybrid retriever if nodes are provided
            retriever = HybridRetriever(
                index=self.index,
                nodes=self.nodes,
                config=self.config.retrieval,
                embed_model=embed_model
            )
        elif self.config.retrieval.retriever_strategy == "ensemble":
            # Try to create ensemble retriever
            try:
                # Create multiple retrievers
                retrievers = []
                
                # Add vector retriever
                vector_retriever = EnhancedRetriever(
                    index=self.index,
                    config=self.config.retrieval,
                    embed_model=embed_model
                ).vector_retriever
                retrievers.append(vector_retriever)
                
                # Add keyword retriever if available
                if self.nodes:
                    try:
                        from llama_index.retrievers.bm25 import BM25Retriever
                        keyword_retriever = BM25Retriever.from_defaults(
                            nodes=self.nodes,
                            similarity_top_k=self.config.retrieval.keyword_top_k
                        )
                        retrievers.append(keyword_retriever)
                    except ImportError:
                        pass
                
                # Create ensemble retriever if we have multiple retrievers
                if len(retrievers) > 1:
                    retriever = EnsembleRetriever(
                        retrievers=retrievers,
                        config=self.config.retrieval
                    )
                else:
                    # Fall back to basic retriever
                    retriever = EnhancedRetriever(
                        index=self.index,
                        config=self.config.retrieval
                    )
            except Exception as e:
                logger.error(f"Error creating ensemble retriever: {str(e)}")
                # Fall back to basic retriever
                retriever = EnhancedRetriever(
                    index=self.index,
                    config=self.config.retrieval
                )
        else:
            # Fallback to basic retriever
            retriever = EnhancedRetriever(
                index=self.index,
                config=self.config.retrieval
            )
        
        logger.info(f"Initialized retriever with strategy: {self.config.retrieval.retriever_strategy}")
        return retriever
    
    def _init_reranker(self) -> Optional[Reranker]:
        """Initialize reranking component based on configuration.
        
        Returns:
            Reranker instance or None if reranking is disabled
        """
        if not self.config.reranker.enable_reranking:
            logger.info("Reranking is disabled")
            return None
        
        # Initialize reranker based on type
        if self.config.reranker.reranker_type == "semantic":
            reranker = SemanticReranker(config=self.config.reranker)
        elif self.config.reranker.reranker_type == "llm":
            reranker = LLMReranker(
                llm=self.llm,
                config=self.config.reranker
            )
        elif self.config.reranker.reranker_type == "cohere":
            reranker = CohereReranker(
                config=self.config.reranker
            )
        else:
            # Default to basic reranker
            reranker = Reranker(config=self.config.reranker)
        
        logger.info(f"Initialized reranker with type: {self.config.reranker.reranker_type}")
        return reranker
    
    def _init_synthesizer(self) -> ResponseSynthesizer:
        """Initialize response synthesis component based on configuration.
        
        Returns:
            Response synthesizer instance
        """
        # Initialize synthesizer based on strategy
        if self.config.synthesis.synthesis_strategy == "refine":
            synthesizer = RefineResponseSynthesizer(
                llm=self.llm,
                config=self.config.synthesis
            )
        elif self.config.synthesis.synthesis_strategy == "tree":
            synthesizer = TreeSynthesizer(
                llm=self.llm,
                config=self.config.synthesis
            )
        elif self.config.synthesis.synthesis_strategy == "compact":
            synthesizer = CompactResponseSynthesizer(
                llm=self.llm,
                config=self.config.synthesis
            )
        elif self.config.synthesis.synthesis_strategy == "simple":
            synthesizer = SimpleResponseSynthesizer(
                llm=self.llm,
                config=self.config.synthesis
            )
        else:
            # Default to simple synthesizer
            synthesizer = ResponseSynthesizer(
                llm=self.llm,
                config=self.config.synthesis
            )
        
        logger.info(f"Initialized synthesizer with strategy: {self.config.synthesis.synthesis_strategy}")
        return synthesizer
    
    def _get_cache_key(self, query: str, **kwargs) -> str:
        """Generate a cache key for a query.
        
        Args:
            query: The query string
            **kwargs: Additional arguments that affect results
            
        Returns:
            Cache key string
        """
        # Create a dictionary of all factors that affect results
        cache_factors = {
            "query": query,
            "retriever_strategy": self.config.retrieval.retriever_strategy,
            "similarity_top_k": self.config.retrieval.similarity_top_k,
            "reranker_type": self.config.reranker.reranker_type if self.config.reranker.enable_reranking else None,
            "synthesis_strategy": self.config.synthesis.synthesis_strategy,
            # Add additional kwargs
            **{k: v for k, v in kwargs.items() if k not in ["stream", "verbose"]}
        }
        
        # Convert to a stable JSON string
        cache_string = json.dumps(cache_factors, sort_keys=True)
        
        # Create a hash of the cache string
        return hashlib.md5(cache_string.encode("utf-8")).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached results for a query.
        
        Args:
            cache_key: Cache key for the query
            
        Returns:
            Cached results or None if not found
        """
        if not self.config.cache_results:
            return None
        
        cache_file = os.path.join(self.config.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
                
                logger.info(f"Retrieved results from cache: {cache_key}")
                return cache_data
            except Exception as e:
                logger.error(f"Error reading from cache: {str(e)}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save results to cache.
        
        Args:
            cache_key: Cache key for the query
            data: Results to cache
        """
        if not self.config.cache_results:
            return
        
        cache_file = os.path.join(self.config.cache_dir, f"{cache_key}.json")
        
        try:
            # Create a JSON-serializable version of the data
            serializable_data = {
                "response": data["response"],
                "source_nodes": [
                    {
                        "node_id": node["node"].node_id,
                        "text": node["node"].get_content(),
                        "metadata": node["node"].metadata,
                        "score": node["score"]
                    }
                    for node in data["source_nodes"]
                ],
                "metadata": data.get("metadata", {})
            }
            
            with open(cache_file, "w") as f:
                json.dump(serializable_data, f)
            
            logger.info(f"Saved results to cache: {cache_key}")
        except Exception as e:
            logger.error(f"Error writing to cache: {str(e)}")
    
    def query(
        self, 
        query_str: str, 
        **kwargs
    ) -> Response:
        """Process a query through the pipeline.
        
        Args:
            query_str: The query string
            **kwargs: Additional arguments for processing
                - stream: Whether to stream the response
                - verbose: Whether to include verbose debug info
                - top_k: Override for similarity_top_k
                - filter: Metadata filter for retrieval
                
        Returns:
            Query response
        """
        # Check cache first if enabled
        if self.config.cache_results:
            cache_key = self._get_cache_key(query_str, **kwargs)
            cached_results = self._get_from_cache(cache_key)
            
            if cached_results:
                try:
                    # Try to reconstruct Response object from cache
                    from llama_index.core.schema import TextNode
                    
                    # Create source nodes
                    source_nodes = []
                    for node_data in cached_results["source_nodes"]:
                        # Create text node
                        text_node = TextNode(
                            text=node_data["text"],
                            metadata=node_data.get("metadata", {}),
                            id_=node_data.get("node_id")
                        )
                        
                        # Create NodeWithScore
                        source_nodes.append(
                            NodeWithScore(
                                node=text_node,
                                score=node_data.get("score")
                            )
                        )
                    
                    # Create response
                    response = Response(
                        response=cached_results["response"],
                        source_nodes=source_nodes,
                        metadata=cached_results.get("metadata", {})
                    )
                    
                    return response
                except Exception as e:
                    logger.error(f"Error reconstructing response from cache: {str(e)}")
        
        start_time = time.time()
        
        try:
            # 1. Query Transformation
            transformed_query = query_str
            transformation_results = {}
            
            for transformer in self.query_transformers:
                try:
                    result = transformer.transform(query_str)
                    
                    # Different transformers return different formats
                    if isinstance(result, dict):
                        # Store the transformation result
                        transformation_results[transformer.__class__.__name__] = result
                        
                        # Update transformed query if available
                        if "rewritten_query" in result:
                            transformed_query = result["rewritten_query"]
                        elif "transformed_query" in result:
                            transformed_query = result["transformed_query"]
                        elif "expanded_query" in result:
                            transformed_query = result["expanded_query"]
                except Exception as e:
                    logger.error(f"Error in query transformer {transformer.__class__.__name__}: {str(e)}")
            
            # Use the final transformed query for retrieval
            logger.info(f"Original query: {query_str}")
            if transformed_query != query_str:
                logger.info(f"Transformed query: {transformed_query}")
            
            # Handle HyDE hypothetical document for retrieval
            query_bundle = QueryBundle(query_str=transformed_query)
            hyde_doc = None
            for transformer_name, result in transformation_results.items():
                if "hypothetical_document" in result and result["hypothetical_document"]:
                    hyde_doc = result["hypothetical_document"]
                    query_bundle.custom_embedding_strs = [hyde_doc]
                    break
            
            # 2. Retrieval
            retrieval_start_time = time.time()
            
            # Get top_k from kwargs or config
            top_k = kwargs.get("top_k", self.config.retrieval.similarity_top_k)
            
            # Get filter from kwargs
            filter_dict = kwargs.get("filter", None)
            
            # Retrieve relevant nodes
            retrieve_kwargs = {"top_k": top_k}
            if filter_dict:
                retrieve_kwargs["filter"] = filter_dict
                
            retrieved_nodes = self.retriever.retrieve(
                query_bundle,
                **retrieve_kwargs
            )
            
            retrieval_end_time = time.time()
            logger.info(f"Retrieved {len(retrieved_nodes)} nodes in {retrieval_end_time - retrieval_start_time:.2f} seconds")
            
            # 3. Reranking (if enabled)
            reranking_end_time = retrieval_end_time  # Default if no reranking
            
            if self.reranker and self.config.reranker.enable_reranking:
                reranking_start_time = time.time()
                
                # Rerank retrieved nodes
                reranked_nodes = self.reranker.rerank(
                    query_str,  # Use original query for reranking
                    retrieved_nodes
                )
                
                reranking_end_time = time.time()
                logger.info(f"Reranked nodes in {reranking_end_time - reranking_start_time:.2f} seconds")
                
                # Use reranked nodes for synthesis
                nodes_for_synthesis = reranked_nodes
            else:
                # Use retrieved nodes directly
                nodes_for_synthesis = retrieved_nodes
            
            # 4. Response Synthesis
            synthesis_start_time = time.time()
            
            # Synthesize response
            response = self.synthesizer.synthesize(
                query_str,  # Use original query for synthesis
                nodes_for_synthesis
            )
            
            synthesis_end_time = time.time()
            logger.info(f"Synthesized response in {synthesis_end_time - synthesis_start_time:.2f} seconds")
            
            # Calculate metrics
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"Total query processing time: {total_time:.2f} seconds")
            
            # Add metrics to response metadata
            response.metadata = response.metadata or {}
            response.metadata["metrics"] = {
                "total_time": total_time,
                "retrieval_time": retrieval_end_time - retrieval_start_time,
                "reranking_time": reranking_end_time - retrieval_end_time if self.reranker and self.config.reranker.enable_reranking else 0,
                "synthesis_time": synthesis_end_time - reranking_end_time,
                "num_source_nodes": len(nodes_for_synthesis),
            }
            
            # Add transformation results to metadata
            if transformation_results:
                response.metadata["transformations"] = transformation_results
            
            # Cache the results if enabled
            if self.config.cache_results:
                # Convert to a cacheable format
                cache_data = {
                    "response": response.response,
                    "source_nodes": [
                        {
                            "node": node.node,
                            "score": node.score
                        }
                        for node in response.source_nodes
                    ],
                    "metadata": response.metadata
                }
                
                # Save to cache
                cache_key = self._get_cache_key(query_str, **kwargs)
                self._save_to_cache(cache_key, cache_data)
            
            return response
        
        except Exception as e:
            logger.error(f"Error in query pipeline: {str(e)}")
            
            # Return error response
            return Response(
                response=f"Error processing query: {str(e)}",
                source_nodes=[],
            )
    async def aquery(
        self, 
        query_str: str, 
        **kwargs
    ) -> Response:
        """Asynchronously process a query through the pipeline.
        
        Args:
            query_str: The query string
            **kwargs: Additional arguments for processing
                - stream: Whether to stream the response
                - verbose: Whether to include verbose debug info
                - top_k: Override for similarity_top_k
                - filter: Metadata filter for retrieval
                
        Returns:
            Query response
        """
        # Check cache first if enabled
        if self.config.cache_results:
            cache_key = self._get_cache_key(query_str, **kwargs)
            cached_results = self._get_from_cache(cache_key)
            
            if cached_results:
                try:
                    # Try to reconstruct Response object from cache
                    from llama_index.core.schema import TextNode
                    
                    # Create source nodes
                    source_nodes = []
                    for node_data in cached_results["source_nodes"]:
                        # Create text node
                        text_node = TextNode(
                            text=node_data["text"],
                            metadata=node_data.get("metadata", {}),
                            id_=node_data.get("node_id")
                        )
                        
                        # Create NodeWithScore
                        source_nodes.append(
                            NodeWithScore(
                                node=text_node,
                                score=node_data.get("score")
                            )
                        )
                    
                    # Create response
                    response = Response(
                        response=cached_results["response"],
                        source_nodes=source_nodes,
                        metadata=cached_results.get("metadata", {})
                    )
                    
                    return response
                except Exception as e:
                    logger.error(f"Error reconstructing response from cache: {str(e)}")
        
        start_time = time.time()
        
        try:
            # 1. Query Transformation
            transformed_query = query_str
            transformation_results = {}
            
            # Process query transformers sequentially for now
            # Many transformers don't have async implementations yet
            for transformer in self.query_transformers:
                try:
                    if hasattr(transformer, 'atransform'):
                        result = await transformer.atransform(query_str)
                    else:
                        result = transformer.transform(query_str)
                    
                    # Different transformers return different formats
                    if isinstance(result, dict):
                        # Store the transformation result
                        transformation_results[transformer.__class__.__name__] = result
                        
                        # Update transformed query if available
                        if "rewritten_query" in result:
                            transformed_query = result["rewritten_query"]
                        elif "transformed_query" in result:
                            transformed_query = result["transformed_query"]
                        elif "expanded_query" in result:
                            transformed_query = result["expanded_query"]
                except Exception as e:
                    logger.error(f"Error in async query transformer {transformer.__class__.__name__}: {str(e)}")
            
            # Use the final transformed query for retrieval
            logger.info(f"Original query: {query_str}")
            if transformed_query != query_str:
                logger.info(f"Transformed query: {transformed_query}")
            
            # Handle HyDE hypothetical document for retrieval
            query_bundle = QueryBundle(query_str=transformed_query)
            hyde_doc = None
            for transformer_name, result in transformation_results.items():
                if "hypothetical_document" in result and result["hypothetical_document"]:
                    hyde_doc = result["hypothetical_document"]
                    query_bundle.custom_embedding_strs = [hyde_doc]
                    break
            
            # 2. Retrieval
            retrieval_start_time = time.time()
            
            # Get top_k from kwargs or config
            top_k = kwargs.get("top_k", self.config.retrieval.similarity_top_k)
            
            # Get filter from kwargs
            filter_dict = kwargs.get("filter", None)
            
            # Retrieve relevant nodes
            retrieve_kwargs = {"top_k": top_k}
            if filter_dict:
                retrieve_kwargs["filter"] = filter_dict
                
            retrieved_nodes = await self.retriever.aretrieve(
                query_bundle,
                **retrieve_kwargs
            )
            
            retrieval_end_time = time.time()
            logger.info(f"Retrieved {len(retrieved_nodes)} nodes in {retrieval_end_time - retrieval_start_time:.2f} seconds")
            
            # 3. Reranking (if enabled)
            reranking_end_time = retrieval_end_time  # Default if no reranking
            
            if self.reranker and self.config.reranker.enable_reranking:
                reranking_start_time = time.time()
                
                # Rerank retrieved nodes
                if hasattr(self.reranker, 'arerank'):
                    reranked_nodes = await self.reranker.arerank(
                        query_str,  # Use original query for reranking
                        retrieved_nodes
                    )
                else:
                    reranked_nodes = self.reranker.rerank(
                        query_str,  # Use original query for reranking
                        retrieved_nodes
                    )
                
                reranking_end_time = time.time()
                logger.info(f"Reranked nodes in {reranking_end_time - reranking_start_time:.2f} seconds")
                
                # Use reranked nodes for synthesis
                nodes_for_synthesis = reranked_nodes
            else:
                # Use retrieved nodes directly
                nodes_for_synthesis = retrieved_nodes
            
            # 4. Response Synthesis
            synthesis_start_time = time.time()
            
            # Synthesize response
            if self.config.synthesis.use_async:
                response = await self.synthesizer.asynthesize(
                    query_str,  # Use original query for synthesis
                    nodes_for_synthesis
                )
            else:
                response = self.synthesizer.synthesize(
                    query_str,  # Use original query for synthesis
                    nodes_for_synthesis
                )
            
            synthesis_end_time = time.time()
            logger.info(f"Synthesized response in {synthesis_end_time - synthesis_start_time:.2f} seconds")
            
            # Calculate metrics
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"Total query processing time: {total_time:.2f} seconds")
            
            # Add metrics to response metadata
            response.metadata = response.metadata or {}
            response.metadata["metrics"] = {
                "total_time": total_time,
                "retrieval_time": retrieval_end_time - retrieval_start_time,
                "reranking_time": reranking_end_time - retrieval_end_time if self.reranker and self.config.reranker.enable_reranking else 0,
                "synthesis_time": synthesis_end_time - reranking_end_time,
                "num_source_nodes": len(nodes_for_synthesis),
            }
            
            # Add transformation results to metadata
            if transformation_results:
                response.metadata["transformations"] = transformation_results
            
            # Cache the results if enabled
            if self.config.cache_results:
                # Convert to a cacheable format
                cache_data = {
                    "response": response.response,
                    "source_nodes": [
                        {
                            "node": node.node,
                            "score": node.score
                        }
                        for node in response.source_nodes
                    ],
                    "metadata": response.metadata
                }
                
                # Save to cache
                cache_key = self._get_cache_key(query_str, **kwargs)
                self._save_to_cache(cache_key, cache_data)
            
            return response
        
        except Exception as e:
            logger.error(f"Error in async query pipeline: {str(e)}")
            
            # Return error response
            return Response(
                response=f"Error processing async query: {str(e)}",
                source_nodes=[],
            )            