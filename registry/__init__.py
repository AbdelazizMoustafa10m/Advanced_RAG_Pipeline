# --- registry/__init__.py ---

"""
Core components and interfaces for the Document Registry.
"""

from .document_registry import DocumentRegistry
from .status import ProcessingStatus
from .exceptions import DocumentRegistryError, InvalidStatusError