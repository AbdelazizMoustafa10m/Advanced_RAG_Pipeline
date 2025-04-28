# --- sidekick/__init__.py ---
"""
Unified document parsing system for code and technical documents.
"""

__version__ = "0.1.0"

# Make key modules available at the top level
from sidekick import core
from sidekick import pipeline
from sidekick import processors
from sidekick import indexing
from sidekick import llm

__all__ = [
    "core",
    "pipeline",
    "processors",
    "indexing",
    "llm",
]