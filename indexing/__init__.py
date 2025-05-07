# --- sidekick/indexing/__init__.py ---
"""
Indexing and querying components.
"""

from .vector_store import ChromaVectorStoreAdapter
from .filters import (
    FilterOperator, FilterCondition, MetadataFilter, 
    MetadataFilters, create_filter, convert_to_llamaindex_filters
)
