"""
Docling metadata formatter for transforming raw Docling metadata into user-friendly formats.

This module provides a TransformComponent that processes nodes after chunking
but before indexing, formatting complex Docling metadata into clean, standardized
fields for both LLM and embedding contexts.
"""

import logging
from typing import List, Sequence, Dict, Any, Optional, Set

from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.bridge.pydantic import Field, BaseModel
from llama_index.core.schema import TransformComponent

logger = logging.getLogger(__name__)

# --- Configuration Model for Clarity ---
class FormattingConfig(BaseModel):
    """Configuration for which metadata to format and include."""
    include_in_llm: List[str] = Field(default_factory=lambda: [
        'formatted_source', 'formatted_location', 'formatted_headings', 
        'formatted_label', 'file_type', 'node_type', 'functional_title',
        'concise_summary', 'generated_questions_list'
    ])
    include_in_embed: List[str] = Field(default_factory=lambda: [
        'formatted_source', 'formatted_location', 'formatted_headings', 'formatted_label',
        'functional_title', 'concise_summary', 'generated_questions_list'
    ]) # Include enriched metadata for embedding too
    
    # Mapping from raw keys to new formatted keys
    source_keys: Dict[str, str] = Field(default_factory=lambda: {'origin': 'formatted_source'})
    location_keys: Dict[str, str] = Field(default_factory=lambda: {'prov': 'formatted_location'})
    headings_key: str = 'headings'
    formatted_headings_key: str = 'formatted_headings'
    label_keys: Dict[str, str] = Field(default_factory=lambda: {'doc_items': 'formatted_label'})
    
    heading_separator: str = " > "
    max_headings: Optional[int] = 3 # Limit number of headings included

# --- The Transformation Component ---
class DoclingMetadataFormatter(TransformComponent):
    """
    Formats raw Docling metadata into clean strings and configures
    nodes to include them for LLM and Embedding modes.
    
    This component:
    1. Creates human-readable formatted metadata fields from complex Docling metadata
    2. Configures node metadata exclusion for LLM and embedding contexts
    3. Ensures raw complex metadata is excluded while formatted fields are included
    """
    config: FormattingConfig = Field(default_factory=FormattingConfig)

    def __init__(self, config: Optional[FormattingConfig] = None):
        super().__init__()
        self.config = config or FormattingConfig()

    def _format_source(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Formats the 'origin' metadata."""
        key = next(iter(self.config.source_keys.keys())) # Get the raw key name (e.g., 'origin')
        
        origin_data = metadata.get(key)
        if isinstance(origin_data, dict):
            filename = origin_data.get('filename', 'N/A')
            mimetype = origin_data.get('mimetype')
            if mimetype:
                 return f"Source: {filename} (Type: {mimetype})"
            return f"Source: {filename}"
        logger.debug(f"Metadata key '{key}' not found or not a dict in node.")
        return None # Return None if formatting fails

    def _format_location(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Formats the location metadata from either top-level 'prov' or nested inside 'doc_items'."""
        # First try the direct prov field (as configured)
        key = next(iter(self.config.location_keys.keys())) # e.g., 'prov'
        
        # Try direct access first
        prov_data = metadata.get(key)
        if isinstance(prov_data, list) and prov_data:
            # Use the first provenance item
            prov_item = prov_data[0]
            if isinstance(prov_item, dict):
                page_no = prov_item.get('page_no')
                if page_no is not None:
                    return f"Page: {page_no}"
        
        # If that failed, try to find prov inside doc_items
        doc_items = metadata.get('doc_items')
        if isinstance(doc_items, list) and doc_items:
            for item in doc_items:
                if isinstance(item, dict) and 'prov' in item:
                    item_prov = item.get('prov')
                    if isinstance(item_prov, list) and item_prov:
                        prov_item = item_prov[0]
                        if isinstance(prov_item, dict):
                            page_no = prov_item.get('page_no')
                            if page_no is not None:
                                return f"Page: {page_no}"
                            
                            # If we find a bbox but no page_no, use that
                            bbox = prov_item.get('bbox')
                            if bbox:
                                # Format depending on what keys are present
                                if 't' in bbox and 'l' in bbox:
                                    return f"Position: top={bbox.get('t'):.1f}, left={bbox.get('l'):.1f}"
                                else:
                                    return f"Position: {str(bbox)}"
        
        logger.debug(f"Could not find page number or location information in metadata")
        return None

    def _format_headings(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Formats the 'headings' metadata."""
        key = self.config.headings_key
        
        headings = metadata.get(key)
        if isinstance(headings, list) and headings:
            # Limit the number of headings if configured
            if self.config.max_headings is not None:
                headings = headings[-self.config.max_headings:] # Take the last N headings (often most specific)
            return f"Section: {self.config.heading_separator.join(headings)}"
        return None # No headings or not a list

    def _format_label(self, metadata: Dict[str, Any]) -> Optional[str]:
         """Formats the 'doc_items' label metadata."""
         key = next(iter(self.config.label_keys.keys())) # e.g., 'doc_items'
         
         doc_items = metadata.get(key)
         if isinstance(doc_items, list) and doc_items:
             doc_item = doc_items[0] # Assuming one item per node based on input
             if isinstance(doc_item, dict):
                 label = doc_item.get('label')
                 if label:
                     return f"ContentType: {label}"
         return None

    def _debug_metadata_structure(self, node_id: str, metadata: Dict[str, Any]) -> None:
        """Debug log the structure of metadata to help troubleshoot formatting issues."""
        try:
            # Log the top-level keys
            logger.debug(f"Node {node_id} metadata keys: {list(metadata.keys())}")
            
            # Specifically examine keys we're interested in
            if 'doc_items' in metadata:
                doc_items = metadata['doc_items']
                if isinstance(doc_items, list) and doc_items:
                    logger.debug(f"doc_items[0] keys: {list(doc_items[0].keys()) if isinstance(doc_items[0], dict) else 'Not a dict'}")
                    # Check if prov is in the first doc_item
                    if isinstance(doc_items[0], dict) and 'prov' in doc_items[0]:
                        logger.debug(f"Found prov in doc_items[0]: {doc_items[0]['prov']}")
        except Exception as e:
            logger.debug(f"Error debugging metadata structure: {e}")
    
    def __call__(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        """
        Apply formatting and update exclusion keys.
        
        This method:
        1. Identifies Docling/Document nodes
        2. Formats complex metadata fields into human-readable strings
        3. Configures metadata exclusion for LLM and embedding contexts
        4. Ensures raw complex metadata is excluded while formatted fields are included
        """
        processed_nodes = []
        for node in nodes:
            try:
                # --- Add debug for metadata structure ---
                self._debug_metadata_structure(node.node_id, node.metadata)
                
                # --- Process only Docling/Document nodes ---
                # Broader detection of Docling nodes: check multiple indicators
                is_docling_node = False
                docling_indicators = [
                    # Check for direct indicators
                    node.metadata.get('file_type') == 'document',
                    node.metadata.get('node_type') == 'document',
                    'origin' in node.metadata,
                    'doc_items' in node.metadata,
                    # Check for schema name containing 'docling'
                    isinstance(node.metadata.get('schema_name', ''), str) and 'docling' in node.metadata.get('schema_name', '').lower()
                ]
                
                if any(docling_indicators):
                    is_docling_node = True
                    logger.debug(f"Detected Docling node: {node.node_id}")
                
                if is_docling_node:

                    # --- 1. Format Metadata ---
                    formatted_source = self._format_source(node.metadata)
                    if formatted_source:
                        node.metadata[self.config.source_keys['origin']] = formatted_source # Use the target key name

                    formatted_location = self._format_location(node.metadata)
                    if formatted_location:
                        # Use the target formatted key (from the dict value)
                        formatted_key = self.config.location_keys['prov']
                        node.metadata[formatted_key] = formatted_location
                        logger.debug(f"Added formatted location: {formatted_location} under key {formatted_key}")
                    else:
                        logger.debug(f"No location information found for node {node.node_id}")

                    formatted_headings = self._format_headings(node.metadata)
                    if formatted_headings:
                        node.metadata[self.config.formatted_headings_key] = formatted_headings
                        
                    formatted_label = self._format_label(node.metadata)
                    if formatted_label:
                         node.metadata[self.config.label_keys['doc_items']] = formatted_label

                    # Debugging output to help trace metadata processing
                    logger.debug(f"Formatted metadata: source={node.metadata.get(self.config.source_keys['origin'])}, " 
                               f"location={node.metadata.get(self.config.location_keys['prov'])}, " 
                               f"headings={node.metadata.get(self.config.formatted_headings_key)}, "
                               f"label={node.metadata.get(self.config.label_keys['doc_items'])}")

                    # --- 2. Update Exclusion Lists ---
                    # Start with all keys
                    all_keys = set(node.metadata.keys())

                    # Keys to potentially include in LLM
                    llm_include_keys = set(self.config.include_in_llm)
                    # Calculate keys to exclude for LLM
                    llm_exclude_keys = all_keys - llm_include_keys
                    node.excluded_llm_metadata_keys = sorted(list(llm_exclude_keys))

                    # Keys to potentially include in Embed
                    embed_include_keys = set(self.config.include_in_embed)
                    # Calculate keys to exclude for Embed
                    embed_exclude_keys = all_keys - embed_include_keys
                    node.excluded_embed_metadata_keys = sorted(list(embed_exclude_keys))

                    # --- 3. Ensure Raw Keys are Excluded (Important!) ---
                    # Make sure the original complex/raw keys are definitely excluded
                    raw_keys_to_exclude = {
                        next(iter(self.config.source_keys.keys())), # e.g. 'origin'
                        next(iter(self.config.location_keys.keys())), # e.g. 'prov'
                        'doc_items', # The raw list/dict structure
                        # Add any other raw keys you want to force exclude
                    }
                    
                    # Add raw keys to exclusion sets if they exist
                    node.excluded_llm_metadata_keys = sorted(list(set(node.excluded_llm_metadata_keys).union(raw_keys_to_exclude.intersection(all_keys))))
                    node.excluded_embed_metadata_keys = sorted(list(set(node.excluded_embed_metadata_keys).union(raw_keys_to_exclude.intersection(all_keys))))

                    # Apply template to format all enriched metadata into a single field
                    self._apply_template(node)
                
                # Always add the node to our result list, even if no formatting was done
                processed_nodes.append(node)

            except Exception as e:
                logger.error(f"Error processing node {node.node_id}: {e}", exc_info=True)
                # Append unprocessed node on error to ensure pipeline continues
                processed_nodes.append(node)

        return processed_nodes
        
    def _apply_template(self, node: BaseNode) -> None:
        """Apply a formatting template to create a consolidated view of enriched metadata.
        
        This ensures that enriched metadata like concise_summary, functional_title, etc.
        are visible to both LLM and embedding models.
        
        Args:
            node: The node to format
        """
        # Skip if not a docling node or if it doesn't have any enriched metadata
        if not node.metadata.get('file_type') == 'document':
            return
            
        # Collect enriched metadata fields that might be present
        enriched_fields = {}
        
        # Check for standard enrichment fields
        for field in [
            'concise_summary', 
            'functional_title', 
            'generated_questions_list',
            'formatted_source',
            'formatted_location',
            'formatted_headings',
            'formatted_label'
        ]:
            if field in node.metadata and node.metadata[field]:
                enriched_fields[field] = node.metadata[field]
        
        # If no enriched fields found, nothing to do
        if not enriched_fields:
            return
            
        # Build the formatted template with available fields
        formatted_sections = []
        
        # Add document info
        if 'formatted_source' in enriched_fields:
            formatted_sections.append(enriched_fields['formatted_source'])
            
        if 'formatted_location' in enriched_fields:
            formatted_sections.append(enriched_fields['formatted_location'])
            
        if 'formatted_headings' in enriched_fields:
            formatted_sections.append(enriched_fields['formatted_headings'])
            
        if 'formatted_label' in enriched_fields:
            formatted_sections.append(enriched_fields['formatted_label'])
        
        # Add enrichment data
        if 'functional_title' in enriched_fields:
            formatted_sections.append(f"Title: {enriched_fields['functional_title']}")
            
        if 'concise_summary' in enriched_fields:
            formatted_sections.append(f"Summary: {enriched_fields['concise_summary']}")
            
        if 'generated_questions_list' in enriched_fields and isinstance(enriched_fields['generated_questions_list'], list):
            questions = '\n'.join([f"- {q}" for q in enriched_fields['generated_questions_list']])
            formatted_sections.append(f"Questions:\n{questions}")
        
        # Create the consolidated template
        formatted_metadata = '\n'.join(formatted_sections)
        
        # Store the formatted metadata in a field that will be included for LLM and embedding
        node.metadata['formatted_metadata'] = formatted_metadata
        
        # Make sure this field is included in both LLM and embedding modes
        if 'formatted_metadata' not in self.config.include_in_llm:
            self.config.include_in_llm.append('formatted_metadata')
            
        if 'formatted_metadata' not in self.config.include_in_embed:
            self.config.include_in_embed.append('formatted_metadata')
            
        # Update exclusion lists to ensure formatted_metadata is not excluded
        if hasattr(node, 'excluded_llm_metadata_keys') and 'formatted_metadata' in node.excluded_llm_metadata_keys:
            node.excluded_llm_metadata_keys.remove('formatted_metadata')
            
        if hasattr(node, 'excluded_embed_metadata_keys') and 'formatted_metadata' in node.excluded_embed_metadata_keys:
            node.excluded_embed_metadata_keys.remove('formatted_metadata')
