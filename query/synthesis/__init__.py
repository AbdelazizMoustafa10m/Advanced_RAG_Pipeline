# --- query/synthesis/__init__.py ---
"""Answer synthesis components for advanced RAG."""

from .synthesizer import (
    ResponseSynthesizer,
    SimpleResponseSynthesizer,
    RefineResponseSynthesizer,
    TreeSynthesizer,
    CompactResponseSynthesizer,
    StructuredResponseSynthesizer,
)

__all__ = [
    "ResponseSynthesizer",
    "SimpleResponseSynthesizer",
    "RefineResponseSynthesizer",
    "TreeSynthesizer",
    "CompactResponseSynthesizer",
    "StructuredResponseSynthesizer",
]