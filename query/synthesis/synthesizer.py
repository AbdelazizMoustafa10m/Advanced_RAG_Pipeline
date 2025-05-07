# --- query/synthesis/synthesizer.py ---
import logging
from typing import List, Optional, Dict, Any, Union, Type
from dataclasses import dataclass
import time

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
from llama_index.core.base.response.schema import Response
from llama_index.core.llms import LLM
from llama_index.core.response_synthesizers import (
    SimpleSummarize,
    Refine,
    CompactAndRefine,
    TreeSummarize
)
from pydantic import BaseModel

from core.interfaces import IResponseSynthesizer
from core.config import SynthesisConfig

logger = logging.getLogger(__name__)


class ResponseSynthesizer(IResponseSynthesizer):
    """Base class for response synthesis components."""

    def __init__(
        self,
        llm: LLM,
        config: Optional[SynthesisConfig] = None,
    ):
        """Initialize response synthesizer.

        Args:
            llm: LLM for response generation
            config: Optional synthesis configuration
        """
        self.llm = llm
        self.config = config or SynthesisConfig()

    def synthesize(
        self,
        query_bundle_or_str: Union[QueryBundle, str],
        nodes: List[NodeWithScore],
        **kwargs
    ) -> Response:
        """Synthesize a response from retrieved nodes.

        Base implementation generates a simple response.

        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for synthesis

        Returns:
            Synthesized response
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str

        if not nodes:
            return Response(
                response="No relevant information found.",
                source_nodes=[],
            )

        # Default behavior: Return the text of the top node
        return Response(
            response=nodes[0].node.get_content(),
            source_nodes=nodes,
        )

    async def asynthesize(
        self,
        query_bundle_or_str: Union[QueryBundle, str],
        nodes: List[NodeWithScore],
        **kwargs
    ) -> Response:
        """Asynchronously synthesize a response from retrieved nodes.

        Base implementation calls the synchronous method.

        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for synthesis

        Returns:
            Synthesized response
        """
        return self.synthesize(query_bundle_or_str, nodes, **kwargs)


class SimpleResponseSynthesizer(ResponseSynthesizer):
    """Simple response synthesizer that creates a response from retrieved nodes.

    This synthesizer uses a single prompt to generate a response based on
    the retrieved contexts.
    """

    def __init__(
        self,
        llm: LLM,
        config: Optional[SynthesisConfig] = None,
    ):
        """Initialize simple response synthesizer.

        Args:
            llm: LLM for response generation
            config: Optional synthesis configuration
        """
        super().__init__(llm, config)

        # Initialize LlamaIndex synthesizer
        self.llamaindex_synthesizer = SimpleSummarize(
            llm=self.llm,
            streaming=self.config.streaming
        )

    def synthesize(
        self,
        query_bundle_or_str: Union[QueryBundle, str],
        nodes: List[NodeWithScore],
        **kwargs
    ) -> Response:
        """Synthesize a response using simple concatenation and summarization.

        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for synthesis

        Returns:
            Synthesized response
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str

        if not nodes:
            return Response(
                response="No relevant information found.",
                source_nodes=[],
            )

        try:
            # Use LlamaIndex synthesizer
            start_time = time.time()

            response = self.llamaindex_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )

            end_time = time.time()
            logger.info(f"Synthesized response in {end_time - start_time:.2f} seconds")

            return response
        except Exception as e:
            logger.error(f"Error in simple response synthesis: {str(e)}")

            # Fallback to base implementation
            return super().synthesize(query_bundle, nodes, **kwargs)

    async def asynthesize(
        self,
        query_bundle_or_str: Union[QueryBundle, str],
        nodes: List[NodeWithScore],
        **kwargs
    ) -> Response:
        """Asynchronously synthesize a response.

        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for synthesis

        Returns:
            Synthesized response
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str

        if not nodes:
            return Response(
                response="No relevant information found.",
                source_nodes=[],
            )

        try:
            # Use LlamaIndex synthesizer
            start_time = time.time()

            response = await self.llamaindex_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            end_time = time.time()
            logger.info(f"Asynchronously synthesized response in {end_time - start_time:.2f} seconds")

            return response
        except Exception as e:
            logger.error(f"Error in async simple response synthesis: {str(e)}")

            # Fallback to synchronous implementation
            return self.synthesize(query_bundle, nodes, **kwargs)


class RefineResponseSynthesizer(ResponseSynthesizer):
    """Response synthesizer that refines responses iteratively.

    This synthesizer processes retrieved nodes one by one, refining
    the response with each additional context.
    """

    def __init__(
        self,
        llm: LLM,
        config: Optional[SynthesisConfig] = None,
    ):
        """Initialize refine response synthesizer.

        Args:
            llm: LLM for response generation
            config: Optional synthesis configuration
        """
        super().__init__(llm, config)

        # Initialize LlamaIndex synthesizer
        self.llamaindex_synthesizer = Refine(
            llm=self.llm,
            streaming=self.config.streaming,
            structured_answer_filtering=self.config.structured_answer_filtering,
        )

    def synthesize(
        self,
        query_bundle_or_str: Union[QueryBundle, str],
        nodes: List[NodeWithScore],
        **kwargs
    ) -> Response:
        """Synthesize a response using iterative refinement.

        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for synthesis

        Returns:
            Synthesized response
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str

        if not nodes:
            return Response(
                response="No relevant information found.",
                source_nodes=[],
            )

        try:
            # Use LlamaIndex synthesizer
            start_time = time.time()

            response = self.llamaindex_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )

            end_time = time.time()
            logger.info(f"Synthesized response with refinement in {end_time - start_time:.2f} seconds")

            return response
        except Exception as e:
            logger.error(f"Error in refine response synthesis: {str(e)}")

            # Fallback to base implementation
            return super().synthesize(query_bundle, nodes, **kwargs)

    async def asynthesize(
        self,
        query_bundle_or_str: Union[QueryBundle, str],
        nodes: List[NodeWithScore],
        **kwargs
    ) -> Response:
        """Asynchronously synthesize a response with refinement.

        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for synthesis

        Returns:
            Synthesized response
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str

        if not nodes:
            return Response(
                response="No relevant information found.",
                source_nodes=[],
            )

        try:
            # Use LlamaIndex synthesizer
            start_time = time.time()

            response = await self.llamaindex_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            end_time = time.time()
            logger.info(f"Asynchronously synthesized response with refinement in {end_time - start_time:.2f} seconds")

            return response
        except Exception as e:
            logger.error(f"Error in async refine response synthesis: {str(e)}")

            # Fallback to synchronous implementation
            return self.synthesize(query_bundle, nodes, **kwargs)


class TreeSynthesizer(ResponseSynthesizer):
    """Response synthesizer using hierarchical tree-based summarization.

    This synthesizer processes retrieved nodes in a tree structure,
    summarizing groups of nodes and then combining those summaries.
    """

    def __init__(
        self,
        llm: LLM,
        config: Optional[SynthesisConfig] = None,
    ):
        """Initialize tree synthesizer.

        Args:
            llm: LLM for response generation
            config: Optional synthesis configuration
        """
        super().__init__(llm, config)

        # Initialize LlamaIndex synthesizer
        self.llamaindex_synthesizer = TreeSummarize(
            llm=self.llm,
            streaming=self.config.streaming,
            use_async=self.config.use_async,
            verbose=False,
        )

    def synthesize(
        self,
        query_bundle_or_str: Union[QueryBundle, str],
        nodes: List[NodeWithScore],
        **kwargs
    ) -> Response:
        """Synthesize a response using hierarchical tree summarization.

        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for synthesis

        Returns:
            Synthesized response
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str

        if not nodes:
            return Response(
                response="No relevant information found.",
                source_nodes=[],
            )

        try:
            # Use LlamaIndex synthesizer
            start_time = time.time()

            response = self.llamaindex_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )

            end_time = time.time()
            logger.info(f"Synthesized response with tree summarization in {end_time - start_time:.2f} seconds")

            return response
        except Exception as e:
            logger.error(f"Error in tree response synthesis: {str(e)}")

            # Fallback to base implementation
            return super().synthesize(query_bundle, nodes, **kwargs)

    async def asynthesize(
        self,
        query_bundle_or_str: Union[QueryBundle, str],
        nodes: List[NodeWithScore],
        **kwargs
    ) -> Response:
        """Asynchronously synthesize a response with tree summarization.

        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for synthesis

        Returns:
            Synthesized response
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str

        if not nodes:
            return Response(
                response="No relevant information found.",
                source_nodes=[],
            )

        try:
            # Use LlamaIndex synthesizer
            start_time = time.time()

            response = await self.llamaindex_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            end_time = time.time()
            logger.info(f"Asynchronously synthesized response with tree summarization in {end_time - start_time:.2f} seconds")

            return response
        except Exception as e:
            logger.error(f"Error in async tree response synthesis: {str(e)}")

            # Fallback to synchronous implementation
            return self.synthesize(query_bundle, nodes, **kwargs)


class CompactResponseSynthesizer(ResponseSynthesizer):
    """Response synthesizer that preprocesses nodes to reduce token usage.

    This synthesizer compacts context nodes before generating a response,
    making it more efficient for handling many documents.
    """

    def __init__(
        self,
        llm: LLM,
        config: Optional[SynthesisConfig] = None,
    ):
        """Initialize compact response synthesizer.

        Args:
            llm: LLM for response generation
            config: Optional synthesis configuration
        """
        super().__init__(llm, config)

        # Initialize LlamaIndex synthesizer
        self.llamaindex_synthesizer = CompactAndRefine(
            llm=self.llm,
            streaming=self.config.streaming,
            structured_answer_filtering=self.config.structured_answer_filtering,
        )

    def synthesize(
        self,
        query_bundle_or_str: Union[QueryBundle, str],
        nodes: List[NodeWithScore],
        **kwargs
    ) -> Response:
        """Synthesize a response using compaction and refinement.

        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for synthesis

        Returns:
            Synthesized response
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str

        if not nodes:
            return Response(
                response="No relevant information found.",
                source_nodes=[],
            )

        try:
            # Use LlamaIndex synthesizer
            start_time = time.time()

            response = self.llamaindex_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )

            end_time = time.time()
            logger.info(f"Synthesized response with compaction in {end_time - start_time:.2f} seconds")

            return response
        except Exception as e:
            logger.error(f"Error in compact response synthesis: {str(e)}")

            # Fallback to base implementation
            return super().synthesize(query_bundle, nodes, **kwargs)

    async def asynthesize(
        self,
        query_bundle_or_str: Union[QueryBundle, str],
        nodes: List[NodeWithScore],
        **kwargs
    ) -> Response:
        """Asynchronously synthesize a response with compaction.

        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for synthesis

        Returns:
            Synthesized response
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str

        if not nodes:
            return Response(
                response="No relevant information found.",
                source_nodes=[],
            )

        try:
            # Use LlamaIndex synthesizer
            start_time = time.time()

            response = await self.llamaindex_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            end_time = time.time()
            logger.info(f"Asynchronously synthesized response with compaction in {end_time - start_time:.2f} seconds")

            return response
        except Exception as e:
            logger.error(f"Error in async compact response synthesis: {str(e)}")

            # Fallback to synchronous implementation
            return self.synthesize(query_bundle, nodes, **kwargs)


class StructuredResponseSynthesizer(ResponseSynthesizer):
    """Response synthesizer that returns structured data.

    This synthesizer uses LLMs to generate responses in a structured format
    defined by a Pydantic model.
    """

    def __init__(
        self,
        llm: LLM,
        output_cls: Type[BaseModel],
        config: Optional[SynthesisConfig] = None,
    ):
        """Initialize structured response synthesizer.

        Args:
            llm: LLM for response generation
            output_cls: Pydantic model class for output structure
            config: Optional synthesis configuration
        """
        super().__init__(llm, config)
        self.output_cls = output_cls

        # Initialize LlamaIndex synthesizer if available
        try:
            from llama_index.core.response_synthesizers.structured import StructuredRefineResponseSynthesizer

            self.llamaindex_synthesizer = StructuredRefineResponseSynthesizer(
                llm=self.llm,
                output_cls=self.output_cls,
                streaming=self.config.streaming,
            )
            self.initialized = True
        except ImportError:
            logger.warning(f"Could not import StructuredRefineResponseSynthesizer, falling back to base synthesizer")
            self.initialized = False

    def synthesize(
        self,
        query_bundle_or_str: Union[QueryBundle, str],
        nodes: List[NodeWithScore],
        **kwargs
    ) -> Response:
        """Synthesize a structured response from retrieved nodes.

        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for synthesis

        Returns:
            Synthesized response
        """
        # Convert string to QueryBundle if needed
        if isinstance(query_bundle_or_str, str):
            query_bundle = QueryBundle(query_str=query_bundle_or_str)
        else:
            query_bundle = query_bundle_or_str

        if not nodes:
            return Response(
                response="No relevant information found.",
                source_nodes=[],
            )

        if not self.initialized:
            logger.warning("Structured response synthesizer not properly initialized")
            return super().synthesize(query_bundle, nodes, **kwargs)

        try:
            # Use LlamaIndex synthesizer
            start_time = time.time()

            response = self.llamaindex_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )

            end_time = time.time()
            logger.info(f"Synthesized structured response in {end_time - start_time:.2f} seconds")

            return response
        except Exception as e:
            logger.error(f"Error in structured response synthesis: {str(e)}")

            # Fallback to base implementation
            return super().synthesize(query_bundle, nodes, **kwargs)

    async def asynthesize(
        self,
        query_bundle_or_str: Union[QueryBundle, str],
        nodes: List[NodeWithScore],
        **kwargs
    ) -> Response:
        """Asynchronously synthesize a structured response.

        Args:
            query_bundle_or_str: The query string or QueryBundle
            nodes: List of nodes with relevance scores
            **kwargs: Additional arguments for synthesis

        Returns:
            Synthesized response
        """
        # Structured synthesizer doesn't yet support async, use sync version
        return self.synthesize(query_bundle_or_str, nodes, **kwargs)