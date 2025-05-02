"""Embedder module for the Advanced RAG Pipeline.

This module provides embedding capabilities for the Advanced RAG Pipeline using
LlamaIndex embedding models. It supports various embedding providers including
HuggingFace, OpenAI, Cohere, Vertex AI, and AWS Bedrock.
"""

from .embedder_factory import EmbedderFactory
from .llamaindex_embedder_service import LlamaIndexEmbedderService

__all__ = ["EmbedderFactory", "LlamaIndexEmbedderService"]
