# --- indexing/filters.py ---

from typing import Dict, Any, List, Optional, Union
from enum import Enum


class FilterOperator(str, Enum):
    """Operators for metadata filters."""
    EQ = "eq"  # Equal
    NE = "ne"  # Not equal
    GT = "gt"  # Greater than
    GTE = "gte"  # Greater than or equal
    LT = "lt"  # Less than
    LTE = "lte"  # Less than or equal
    IN = "in"  # In list
    NIN = "nin"  # Not in list
    CONTAINS = "contains"  # String contains
    STARTSWITH = "startswith"  # String starts with
    ENDSWITH = "endswith"  # String ends with


class FilterCondition(str, Enum):
    """Conditions for combining multiple filters."""
    AND = "and"
    OR = "or"


class MetadataFilter:
    """Single metadata filter condition."""
    
    def __init__(
        self,
        key: str,
        value: Any,
        operator: FilterOperator = FilterOperator.EQ
    ):
        """Initialize metadata filter.
        
        Args:
            key: Metadata key to filter on
            value: Value to compare against
            operator: Comparison operator
        """
        self.key = key
        self.value = value
        self.operator = operator
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary format.
        
        Returns:
            Filter as dictionary
        """
        return {
            "key": self.key,
            "value": self.value,
            "operator": self.operator.value
        }


class MetadataFilters:
    """Group of metadata filters with condition."""
    
    def __init__(
        self,
        filters: List[MetadataFilter],
        condition: FilterCondition = FilterCondition.AND
    ):
        """Initialize metadata filters.
        
        Args:
            filters: List of metadata filters
            condition: Condition for combining filters
        """
        self.filters = filters
        self.condition = condition
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filters to dictionary format.
        
        Returns:
            Filters as dictionary
        """
        return {
            "filters": [f.to_dict() for f in self.filters],
            "condition": self.condition.value
        }


def create_filter(
    metadata_dict: Dict[str, Any],
    condition: FilterCondition = FilterCondition.AND
) -> MetadataFilters:
    """Create metadata filters from a dictionary.
    
    Args:
        metadata_dict: Dictionary mapping keys to values
        condition: Condition for combining filters
        
    Returns:
        Metadata filters
    """
    filters = []
    for key, value in metadata_dict.items():
        filters.append(MetadataFilter(key=key, value=value))
    
    return MetadataFilters(filters=filters, condition=condition)


def convert_to_llamaindex_filters(metadata_filters: Union[MetadataFilters, Dict[str, Any]]) -> Dict[str, Any]:
    """Convert metadata filters to LlamaIndex filter format.
    
    Args:
        metadata_filters: Metadata filters or dictionary
        
    Returns:
        LlamaIndex-compatible filters dictionary
    """
    if isinstance(metadata_filters, dict):
        # Simple case: convert dictionary to operator filters
        filters_dict = {}
        for key, value in metadata_filters.items():
            filters_dict[key] = {"$eq": value}
        return filters_dict
    
    # Handle MetadataFilters object
    if isinstance(metadata_filters, MetadataFilters):
        # Convert to LlamaIndex's filter format
        if metadata_filters.condition == FilterCondition.AND:
            # For AND condition, merge all filters into a single dictionary
            filters_dict = {}
            for f in metadata_filters.filters:
                op_str = f"${f.operator.value}" if f.operator != FilterOperator.EQ else "$eq"
                filters_dict[f.key] = {op_str: f.value}
            return filters_dict
        
        # For OR condition, use $or operator with list of conditions
        or_conditions = []
        for f in metadata_filters.filters:
            op_str = f"${f.operator.value}" if f.operator != FilterOperator.EQ else "$eq"
            or_conditions.append({f.key: {op_str: f.value}})
        
        return {"$or": or_conditions}
    
    # Handle unexpected input
    raise ValueError(f"Unsupported filter type: {type(metadata_filters)}")
