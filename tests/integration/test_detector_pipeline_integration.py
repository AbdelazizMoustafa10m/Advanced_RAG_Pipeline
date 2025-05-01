"""
Integration tests for the document detector with the pipeline orchestrator.
Tests how the detector integrates with the pipeline for document processing.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import shutil
import sys

# Mock the magic module before importing any modules that depend on it
sys.modules['magic'] = MagicMock()

from core.config import DocumentType, DetectorConfig, UnifiedConfig
from detectors.enhanced_detector_service import EnhancedDetectorService
from pipeline.orchestrator import PipelineOrchestrator
from llama_index.core.schema import Document, TextNode


class TestDetectorPipelineIntegration(unittest.TestCase):
    """Test the integration between the document detector and pipeline orchestrator."""
    
    def setUp(self):
        """Set up the test environment."""
        self.config = UnifiedConfig()
        self.orchestrator = PipelineOrchestrator(self.config)
        
        # Create mock processors
        self.orchestrator.code_processor = MagicMock()
        self.orchestrator.code_processor.process.return_value = [
            TextNode(text="Processed code", metadata={"file_type": "code"})
        ]
        
        self.orchestrator.document_processor = MagicMock()
        self.orchestrator.document_processor.process.return_value = [
            TextNode(text="Processed document", metadata={"file_type": "document"})
        ]
        
        # Create a mock loader
        self.orchestrator.loader = MagicMock()
    
    @patch.object(EnhancedDetectorService, 'detect_type')
    def test_orchestrator_uses_detector_for_document_routing(self, mock_detect):
        """Test that the orchestrator uses the detector for document routing."""
        # Arrange
        test_files = ["/path/to/test.py", "/path/to/document.pdf"]
        
        # Mock the loader to return documents
        self.orchestrator.loader.load_documents.return_value = [
            Document(text="Code content", metadata={"source": test_files[0]}),
            Document(text="Document content", metadata={"source": test_files[1]})
        ]
        
        # Mock the detector to identify document types
        mock_detect.side_effect = [DocumentType.CODE, DocumentType.DOCUMENT]
        
        # Mock _group_document_parts to return the documents grouped by source
        self.orchestrator._group_document_parts = MagicMock()
        self.orchestrator._group_document_parts.return_value = {
            test_files[0]: [Document(text="Code content", metadata={"source": test_files[0]})],
            test_files[1]: [Document(text="Document content", metadata={"source": test_files[1]})]
        }
        
        # Act
        nodes = self.orchestrator.run()
        
        # Assert
        self.orchestrator.loader.load_documents.assert_called_once()
        self.orchestrator._group_document_parts.assert_called_once()
        
        # Verify that the code processor was called with the code document
        code_call_args = self.orchestrator.code_processor.process.call_args[0]
        self.assertEqual(code_call_args[0][0].text, "Code content")
        
        # Verify that the document processor was called with the document document
        doc_call_args = self.orchestrator.document_processor.process.call_args[0]
        self.assertEqual(doc_call_args[0][0].text, "Document content")
        
        # Verify that we got the processed nodes
        self.assertEqual(len(nodes), 2)
        self.assertIn("Processed code", [node.text for node in nodes])
        self.assertIn("Processed document", [node.text for node in nodes])
    
    @patch.object(EnhancedDetectorService, 'batch_detect')
    def test_orchestrator_uses_detector_for_batch_processing(self, mock_batch_detect):
        """Test that the orchestrator uses the detector for batch processing."""
        # Arrange
        test_files = ["/path/to/test1.py", "/path/to/test2.py", "/path/to/document.pdf"]
        
        # Mock the loader to return documents
        self.orchestrator.loader.load_documents.return_value = [
            Document(text="Code content 1", metadata={"source": test_files[0]}),
            Document(text="Code content 2", metadata={"source": test_files[1]}),
            Document(text="Document content", metadata={"source": test_files[2]})
        ]
        
        # Mock the detector to identify document types
        mock_batch_detect.return_value = {
            test_files[0]: DocumentType.CODE,
            test_files[1]: DocumentType.CODE,
            test_files[2]: DocumentType.DOCUMENT
        }
        
        # Mock _group_document_parts to return the documents grouped by source
        self.orchestrator._group_document_parts = MagicMock()
        self.orchestrator._group_document_parts.return_value = {
            test_files[0]: [Document(text="Code content 1", metadata={"source": test_files[0]})],
            test_files[1]: [Document(text="Code content 2", metadata={"source": test_files[1]})],
            test_files[2]: [Document(text="Document content", metadata={"source": test_files[2]})]
        }
        
        # Act
        nodes = self.orchestrator.run()
        
        # Assert
        self.orchestrator.loader.load_documents.assert_called_once()
        self.orchestrator._group_document_parts.assert_called_once()
        
        # Verify that the code processor was called with the code documents
        self.assertEqual(self.orchestrator.code_processor.process.call_count, 2)
        
        # Verify that the document processor was called with the document document
        self.assertEqual(self.orchestrator.document_processor.process.call_count, 1)
        
        # Verify that we got the processed nodes
        self.assertEqual(len(nodes), 3)  # 2 code nodes + 1 document node
    
    def test_orchestrator_handles_document_parts(self):
        """Test that the orchestrator correctly handles document parts."""
        # Arrange
        original_doc = "/path/to/document.pdf"
        
        # Create document parts with metadata indicating they're parts of the same document
        doc_parts = [
            Document(text="Part 1", metadata={"source": f"{original_doc}#page=1"}),
            Document(text="Part 2", metadata={"source": f"{original_doc}#page=2"}),
            Document(text="Part 3", metadata={"source": f"{original_doc}#page=3"})
        ]
        
        # Mock the loader to return document parts
        self.orchestrator.loader.load_documents.return_value = doc_parts
        
        # Act
        with patch.object(self.orchestrator, '_group_document_parts') as mock_group:
            # Call the actual implementation to test it
            mock_group.side_effect = self.orchestrator._group_document_parts
            nodes = self.orchestrator.run()
        
        # Assert
        self.orchestrator.loader.load_documents.assert_called_once()
        
        # Verify that the document processor was called once with all parts
        self.assertEqual(self.orchestrator.document_processor.process.call_count, 1)
        doc_call_args = self.orchestrator.document_processor.process.call_args[0]
        self.assertEqual(len(doc_call_args[0]), 3)  # All 3 parts should be processed together
        
        # Verify that we got the processed node
        self.assertEqual(len(nodes), 1)  # 1 document node (parts are combined)
        self.assertEqual(nodes[0].text, "Processed document")
    
    def test_orchestrator_handles_mixed_document_types(self):
        """Test that the orchestrator correctly handles mixed document types."""
        # Arrange
        code_file = "/path/to/test.py"
        doc_file = "/path/to/document.pdf"
        unknown_file = "/path/to/unknown.xyz"
        
        # Create documents of different types
        documents = [
            Document(text="Code content", metadata={"source": code_file}),
            Document(text="Document content", metadata={"source": doc_file}),
            Document(text="Unknown content", metadata={"source": unknown_file})
        ]
        
        # Mock the loader to return mixed documents
        self.orchestrator.loader.load_documents.return_value = documents
        
        # Mock _group_document_parts to return the documents grouped by source
        self.orchestrator._group_document_parts = MagicMock()
        self.orchestrator._group_document_parts.return_value = {
            code_file: [documents[0]],
            doc_file: [documents[1]],
            unknown_file: [documents[2]]
        }
        
        # Mock the detector to identify document types
        with patch.object(self.orchestrator.detector, 'detect_type') as mock_detect:
            mock_detect.side_effect = [
                DocumentType.CODE,
                DocumentType.DOCUMENT,
                DocumentType.UNKNOWN
            ]
            
            # Act
            nodes = self.orchestrator.run()
        
        # Assert
        self.orchestrator.loader.load_documents.assert_called_once()
        self.orchestrator._group_document_parts.assert_called_once()
        
        # Verify that the code processor was called with the code document
        self.orchestrator.code_processor.process.assert_called_once()
        code_call_args = self.orchestrator.code_processor.process.call_args[0]
        self.assertEqual(code_call_args[0][0].text, "Code content")
        
        # Verify that the document processor was called with the document document
        self.orchestrator.document_processor.process.assert_called_once()
        doc_call_args = self.orchestrator.document_processor.process.call_args[0]
        self.assertEqual(doc_call_args[0][0].text, "Document content")
        
        # Verify that we got the processed nodes (unknown file is skipped)
        self.assertEqual(len(nodes), 2)
    
    def test_orchestrator_error_handling(self):
        """Test that the orchestrator handles errors during processing."""
        # Arrange
        test_files = ["/path/to/test.py", "/path/to/error.py"]
        
        # Mock the loader to return documents
        self.orchestrator.loader.load_documents.return_value = [
            Document(text="Code content", metadata={"source": test_files[0]}),
            Document(text="Error content", metadata={"source": test_files[1]})
        ]
        
        # Mock _group_document_parts to return the documents grouped by source
        self.orchestrator._group_document_parts = MagicMock()
        self.orchestrator._group_document_parts.return_value = {
            test_files[0]: [Document(text="Code content", metadata={"source": test_files[0]})],
            test_files[1]: [Document(text="Error content", metadata={"source": test_files[1]})]
        }
        
        # Mock the detector to identify document types
        with patch.object(self.orchestrator.detector, 'detect_type') as mock_detect:
            mock_detect.side_effect = [DocumentType.CODE, DocumentType.CODE]
            
            # Make the second document cause an error during processing
            self.orchestrator.code_processor.process.side_effect = [
                [TextNode(text="Processed code", metadata={"file_type": "code"})],
                Exception("Processing error")
            ]
            
            # Mock document registry if it's being used
            if hasattr(self.orchestrator, 'document_registry') and self.orchestrator.document_registry:
                self.orchestrator.document_registry = MagicMock()
            
            # Act
            nodes = self.orchestrator.run()
        
        # Assert
        self.orchestrator.loader.load_documents.assert_called_once()
        self.orchestrator._group_document_parts.assert_called_once()
        
        # Verify that the code processor was called twice
        self.assertEqual(self.orchestrator.code_processor.process.call_count, 2)
        
        # Verify that we got only the successfully processed node
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].text, "Processed code")
        
        # If document registry is used, verify it was updated with the error
        if hasattr(self.orchestrator, 'document_registry') and self.orchestrator.document_registry:
            self.orchestrator.document_registry.update_status.assert_called()


class TestEndToEndPipelineIntegration(unittest.TestCase):
    """End-to-end tests for the pipeline integration (optional)."""
    
    def setUp(self):
        """Set up the test environment with real temporary files."""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create test files
        self.python_file = os.path.join(self.test_dir, "test.py")
        with open(self.python_file, "w") as f:
            f.write("print('Hello, world!')")
        
        self.text_file = os.path.join(self.test_dir, "test.txt")
        with open(self.text_file, "w") as f:
            f.write("This is a test document.")
        
        # Create a configuration for the orchestrator
        self.config = UnifiedConfig()
        # Set the input directory to our test directory
        self.config.loader.input_dir = self.test_dir
    
    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.test_dir)
    
    @unittest.skip("This is an end-to-end test that should be run manually")
    def test_end_to_end_pipeline(self):
        """Test the end-to-end pipeline with real files."""
        # Create an orchestrator with the test configuration
        orchestrator = PipelineOrchestrator(self.config)
        
        # Run the pipeline
        nodes = orchestrator.run()
        
        # Verify that we got nodes for both files
        self.assertGreaterEqual(len(nodes), 2)
        
        # Check that we have nodes for both file types
        code_nodes = [node for node in nodes if node.metadata.get("file_type") == "code"]
        text_nodes = [node for node in nodes if node.metadata.get("file_type") == "document"]
        
        self.assertGreaterEqual(len(code_nodes), 1)
        self.assertGreaterEqual(len(text_nodes), 1)


if __name__ == '__main__':
    unittest.main()
