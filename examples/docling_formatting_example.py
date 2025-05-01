"""
Example showing how to use the DoclingMetadataFormatter standalone.

This example demonstrates:
1. How to format Docling metadata for better readability
2. How to configure which metadata fields are included in LLM and embedding modes
3. How the formatter handles both Docling and non-Docling nodes
"""

import logging
import sys
from pathlib import Path

# Add the project root to the Python path to allow imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from llama_index.core.schema import Document, TextNode
from processors.document.docling_metadata_formatter import DoclingMetadataFormatter, FormattingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_docling_node():
    """Create a sample node with typical Docling metadata"""
    return TextNode(
        text="This is a sample paragraph from a PDF document...",
        metadata={
            "file_type": "document",
            "node_type": "document",
            "origin": {
                "filename": "technical_whitepaper.pdf",
                "mimetype": "application/pdf"
            },
            "prov": [
                {
                    "page_no": 5,
                    "bbox": {"t": 150.5, "l": 50.2, "w": 400.0, "h": 20.0}
                }
            ],
            "headings": ["Introduction", "Problem Statement", "Technical Approach"],
            "doc_items": [
                {
                    "label": "paragraph",
                    "conf": 0.95
                }
            ],
            # Additional complex metadata fields that would be excluded
            "raw_text": "Original unprocessed text...",
            "text_blocks": [{}, {}, {}],
            "embedding_model": "sentence-transformers/all-mpnet-base-v2"
        }
    )

def create_sample_code_node():
    """Create a sample code node (non-Docling) to demonstrate filtering"""
    return TextNode(
        text="def example_function():\n    return 'Hello, world!'",
        metadata={
            "file_type": "code",
            "node_type": "code",
            "language": "python",
            "file_path": "/path/to/example.py",
            "function_name": "example_function",
            "line_start": 10,
            "line_end": 11
        }
    )

def main():
    # Create sample nodes
    docling_node = create_sample_docling_node()
    code_node = create_sample_code_node()
    sample_nodes = [docling_node, code_node]
    
    logger.info("======= BEFORE FORMATTING =======")
    
    # Show original metadata that would be passed to LLM
    original_llm_metadata = {k: v for k, v in docling_node.metadata.items() 
                            if k not in docling_node.excluded_llm_metadata_keys}
    logger.info(f"Original LLM metadata keys: {list(original_llm_metadata.keys())}")
    
    # Initialize formatter with custom configuration
    formatter = DoclingMetadataFormatter(
        config=FormattingConfig(
            include_in_llm=[
                'formatted_source', 'formatted_location', 'formatted_headings', 
                'formatted_label', 'file_type', 'node_type'
            ],
            include_in_embed=[
                'formatted_source', 'formatted_location', 'formatted_headings'
            ],
            # Custom heading separator
            heading_separator=" | ",
            # Limit to 2 headings max
            max_headings=2
        )
    )
    
    # Apply formatting to both nodes
    formatted_nodes = formatter(sample_nodes)
    
    logger.info("\n======= AFTER FORMATTING =======")
    
    # Examine the formatted Docling node
    docling_result = formatted_nodes[0]
    
    # Show formatted metadata fields
    logger.info("Formatted metadata fields:")
    for key in ['formatted_source', 'formatted_location', 'formatted_headings', 'formatted_label']:
        if key in docling_result.metadata:
            logger.info(f"  {key}: {docling_result.metadata[key]}")
    
    # Show metadata that would be passed to LLM
    formatted_llm_metadata = {k: v for k, v in docling_result.metadata.items() 
                             if k not in docling_result.excluded_llm_metadata_keys}
    logger.info(f"Formatted LLM metadata keys: {list(formatted_llm_metadata.keys())}")
    
    # Show metadata that would be passed to embedding
    formatted_embed_metadata = {k: v for k, v in docling_result.metadata.items() 
                              if k not in docling_result.excluded_embed_metadata_keys}
    logger.info(f"Formatted Embedding metadata keys: {list(formatted_embed_metadata.keys())}")
    
    # Verify that code node was passed through unchanged
    code_result = formatted_nodes[1]
    logger.info("\nCode node metadata (should be unchanged):")
    logger.info(f"  Node type: {code_result.metadata.get('node_type')}")
    logger.info(f"  Metadata keys: {list(code_result.metadata.keys())}")

if __name__ == "__main__":
    main()
