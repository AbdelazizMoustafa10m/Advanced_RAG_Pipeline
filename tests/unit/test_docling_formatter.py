"""
Unit tests for the DoclingMetadataFormatter component.
"""

import unittest
from unittest.mock import MagicMock
import sys
import os
from pathlib import Path

# Add the project root to the Python path to allow imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from llama_index.core.schema import TextNode
from processors.document.docling_metadata_formatter import DoclingMetadataFormatter, FormattingConfig

class TestDoclingMetadataFormatter(unittest.TestCase):
    """Test the DoclingMetadataFormatter."""

    def setUp(self):
        """Set up the test fixtures."""
        # Create a sample Docling node
        self.docling_node = TextNode(
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
                # Additional metadata that should be excluded
                "raw_text": "Original unprocessed text...",
                "text_blocks": [{}, {}, {}]
            }
        )
        
        # Create a sample code node (non-Docling)
        self.code_node = TextNode(
            text="def example_function():\n    return 'Hello, world!'",
            metadata={
                "file_type": "code",
                "node_type": "code",
                "language": "python",
                "file_path": "/path/to/example.py",
                "function_name": "example_function"
            }
        )
        
        # Default formatter
        self.formatter = DoclingMetadataFormatter()

    def test_format_source(self):
        """Test formatting the source metadata."""
        # Apply formatting
        formatted = self.formatter([self.docling_node])[0]
        
        # Check if formatted_source was created
        self.assertIn('formatted_source', formatted.metadata)
        self.assertEqual(
            formatted.metadata['formatted_source'],
            "Source: technical_whitepaper.pdf (Type: application/pdf)"
        )

    def test_format_location(self):
        """Test formatting the location metadata."""
        # Apply formatting
        formatted = self.formatter([self.docling_node])[0]
        
        # Check if formatted_location was created
        self.assertIn('formatted_location', formatted.metadata)
        self.assertEqual(
            formatted.metadata['formatted_location'],
            "Page: 5"
        )

    def test_format_headings(self):
        """Test formatting the headings metadata."""
        # Apply formatting
        formatted = self.formatter([self.docling_node])[0]
        
        # Check if formatted_headings was created
        self.assertIn('formatted_headings', formatted.metadata)
        self.assertEqual(
            formatted.metadata['formatted_headings'],
            "Section: Introduction > Problem Statement > Technical Approach"
        )
        
        # Test with custom config (max 2 headings)
        custom_formatter = DoclingMetadataFormatter(
            config=FormattingConfig(max_headings=2)
        )
        formatted = custom_formatter([self.docling_node])[0]
        self.assertEqual(
            formatted.metadata['formatted_headings'],
            "Section: Problem Statement > Technical Approach"
        )

    def test_format_label(self):
        """Test formatting the label metadata."""
        # Apply formatting
        formatted = self.formatter([self.docling_node])[0]
        
        # Check if formatted_label was created
        self.assertIn('formatted_label', formatted.metadata)
        self.assertEqual(
            formatted.metadata['formatted_label'],
            "ContentType: paragraph"
        )

    def test_exclusion_keys(self):
        """Test that the correct metadata keys are excluded."""
        # Apply formatting
        formatted = self.formatter([self.docling_node])[0]
        
        # Check LLM exclusion
        # The raw complex fields should be excluded
        self.assertIn('origin', formatted.excluded_llm_metadata_keys)
        self.assertIn('prov', formatted.excluded_llm_metadata_keys)
        self.assertIn('doc_items', formatted.excluded_llm_metadata_keys)
        
        # But the formatted fields should not be excluded
        self.assertNotIn('formatted_source', formatted.excluded_llm_metadata_keys)
        self.assertNotIn('formatted_location', formatted.excluded_llm_metadata_keys)
        
        # Check embed exclusion
        self.assertIn('origin', formatted.excluded_embed_metadata_keys)
        self.assertIn('prov', formatted.excluded_embed_metadata_keys)
        
        # Custom fields should also be excluded as specified in config
        self.assertIn('raw_text', formatted.excluded_llm_metadata_keys)
        self.assertIn('text_blocks', formatted.excluded_llm_metadata_keys)

    def test_non_docling_node_passthrough(self):
        """Test that non-Docling nodes are passed through unchanged."""
        # Process both nodes
        formatted_nodes = self.formatter([self.docling_node, self.code_node])
        
        # The code node should be unchanged
        code_node_result = formatted_nodes[1]
        
        # It should not have any of the formatted fields
        self.assertNotIn('formatted_source', code_node_result.metadata)
        self.assertNotIn('formatted_location', code_node_result.metadata)
        self.assertNotIn('formatted_headings', code_node_result.metadata)
        
        # Its original metadata should be preserved
        self.assertEqual(code_node_result.metadata['file_type'], 'code')
        self.assertEqual(code_node_result.metadata['language'], 'python')
        self.assertEqual(code_node_result.metadata['function_name'], 'example_function')

    def test_custom_config(self):
        """Test with a custom configuration."""
        custom_config = FormattingConfig(
            include_in_llm=['formatted_source', 'formatted_location'],  # Only include these two
            include_in_embed=['formatted_source'],  # Only include source for embedding
            heading_separator=" | ",  # Custom separator
            max_headings=1  # Only keep the last heading
        )
        custom_formatter = DoclingMetadataFormatter(config=custom_config)
        
        # Apply formatting
        formatted = custom_formatter([self.docling_node])[0]
        
        # Check custom heading format
        self.assertEqual(
            formatted.metadata['formatted_headings'],
            "Section: Technical Approach"  # Only last heading
        )
        
        # Check custom exclusions for LLM
        llm_keys = [k for k in formatted.metadata.keys() 
                   if k not in formatted.excluded_llm_metadata_keys]
        self.assertEqual(set(llm_keys), {'formatted_source', 'formatted_location'})
        
        # Check custom exclusions for embedding
        embed_keys = [k for k in formatted.metadata.keys() 
                     if k not in formatted.excluded_embed_metadata_keys]
        self.assertEqual(set(embed_keys), {'formatted_source'})

    def test_error_handling(self):
        """Test that errors don't crash the formatter."""
        # Create a node with invalid metadata
        bad_node = TextNode(
            text="Invalid metadata node",
            metadata={
                "file_type": "document",
                "origin": "not-a-dict",  # This should cause an error when formatting
                "prov": "not-a-list"  # This should cause an error when formatting
            }
        )
        
        # This should not raise an exception
        result = self.formatter([bad_node, self.code_node])
        
        # We should get both nodes back
        self.assertEqual(len(result), 2)
        
        # The bad node should be in the result unchanged
        self.assertEqual(result[0].text, "Invalid metadata node")

if __name__ == '__main__':
    unittest.main()
