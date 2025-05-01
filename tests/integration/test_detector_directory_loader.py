"""
Integration tests for the document detector with the enhanced directory loader.
Tests how the detector integrates with the directory loader for file type detection.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import shutil
import sys

# Mock the magic module before importing any modules that depend on it
sys.modules['magic'] = MagicMock()

from core.config import DocumentType, DetectorConfig, LoaderConfig
from detectors.enhanced_detector_service import EnhancedDetectorService
from loaders.enhanced_directory_loader import EnhancedDirectoryLoader
from llama_index.core import Document


class TestDetectorDirectoryLoaderIntegration(unittest.TestCase):
    """Test the integration between the document detector and directory loader."""
    
    def setUp(self):
        """Set up the test environment."""
        self.detector_config = DetectorConfig()
        self.loader_config = LoaderConfig()
        
        # Create a mock DoclingReader
        self.mock_docling_reader = MagicMock()
        self.mock_docling_reader.load_data.return_value = [
            Document(text="Document content", metadata={"source": "test.pdf"})
        ]
        
        # Create a loader with the mock reader
        self.loader = EnhancedDirectoryLoader(
            docling_reader=self.mock_docling_reader,
            detector_config=self.detector_config,
            loader_config=self.loader_config
        )
    
    @patch.object(EnhancedDetectorService, 'detect_type')
    @patch.object(EnhancedDirectoryLoader, '_load_file_by_type')
    def test_loader_uses_detector_for_single_file(self, mock_load_file, mock_detect):
        """Test that the loader uses the detector for a single file."""
        # Arrange
        test_file = "/path/to/test.py"
        mock_detect.return_value = DocumentType.CODE
        mock_load_file.return_value = [Document(text="Code content", metadata={"source": test_file})]
        
        # Act
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=True):
            docs = self.loader.load_documents(test_file)
        
        # Assert
        mock_detect.assert_called_once_with(test_file)
        mock_load_file.assert_called_once_with(test_file, DocumentType.CODE)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].text, "Code content")
    
    @patch.object(EnhancedDetectorService, 'batch_detect')
    @patch.object(EnhancedDirectoryLoader, '_get_files_from_directory')
    @patch.object(EnhancedDirectoryLoader, '_load_file_by_type')
    def test_loader_uses_detector_for_directory(self, mock_load_file, mock_get_files, mock_batch_detect):
        """Test that the loader uses the detector for a directory."""
        # Arrange
        test_dir = "/path/to/dir"
        test_files = ["/path/to/dir/test.py", "/path/to/dir/doc.pdf"]
        
        mock_get_files.return_value = test_files
        mock_batch_detect.return_value = {
            test_files[0]: DocumentType.CODE,
            test_files[1]: DocumentType.DOCUMENT
        }
        
        # Set up _load_file_by_type to return different documents based on file type
        def load_file_side_effect(file_path, doc_type):
            if doc_type == DocumentType.CODE:
                return [Document(text="Code content", metadata={"source": file_path})]
            else:
                return [Document(text="Document content", metadata={"source": file_path})]
        
        mock_load_file.side_effect = load_file_side_effect
        
        # Act
        with patch('os.path.exists', return_value=True), \
             patch('os.path.isfile', return_value=False), \
             patch('os.path.isdir', return_value=True):
            docs = self.loader.load_documents(test_dir)
        
        # Assert
        mock_get_files.assert_called_once_with(test_dir)
        mock_batch_detect.assert_called_once_with(test_files)
        self.assertEqual(mock_load_file.call_count, 2)
        self.assertEqual(len(docs), 2)
    
    @patch('os.path.exists')
    @patch('os.path.isfile')
    @patch('os.path.isdir')
    def test_loader_handles_nonexistent_source(self, mock_isdir, mock_isfile, mock_exists):
        """Test that the loader handles nonexistent sources."""
        # Arrange
        test_source = "/path/to/nonexistent"
        mock_exists.return_value = False
        
        # Act & Assert
        with self.assertRaises(ValueError):
            self.loader.load_documents(test_source)
    
    def test_load_file_by_type_uses_correct_loader(self):
        """Test that _load_file_by_type uses the correct loader based on file type."""
        # Arrange
        code_file = "/path/to/test.py"
        doc_file = "/path/to/test.pdf"
        unknown_file = "/path/to/unknown.xyz"
        
        # Mock the specialized loaders
        self.loader.code_loader = MagicMock()
        self.loader.code_loader._load_file.return_value = [
            Document(text="Code content", metadata={"source": code_file})
        ]
        
        self.loader.fallback_loader = MagicMock()
        self.loader.fallback_loader._load_file.return_value = [
            Document(text="Unknown content", metadata={"source": unknown_file})
        ]
        
        # Act
        with patch('os.path.exists', return_value=True):
            code_docs = self.loader._load_file_by_type(code_file, DocumentType.CODE)
            doc_docs = self.loader._load_file_by_type(doc_file, DocumentType.DOCUMENT)
            unknown_docs = self.loader._load_file_by_type(unknown_file, DocumentType.UNKNOWN)
        
        # Assert
        self.loader.code_loader._load_file.assert_called_once_with(code_file)
        self.mock_docling_reader.load_data.assert_called_once_with(file_path=[doc_file])
        self.loader.fallback_loader._load_file.assert_called_once_with(unknown_file)
        
        self.assertEqual(len(code_docs), 1)
        self.assertEqual(code_docs[0].metadata["file_type"], "code")
        
        self.assertEqual(len(doc_docs), 1)
        self.assertEqual(doc_docs[0].metadata["file_type"], "document")
        
        self.assertEqual(len(unknown_docs), 1)
        self.assertEqual(unknown_docs[0].metadata["file_type"], "unknown")
    
    @patch('os.path.exists')
    @patch('os.path.isfile')
    @patch('os.path.isdir')
    @patch('os.walk')
    def test_get_files_from_directory_applies_filters(self, mock_walk, mock_isdir, mock_isfile, mock_exists):
        """Test that _get_files_from_directory applies include/exclude patterns."""
        # Arrange
        test_dir = "/path/to/dir"
        mock_exists.return_value = True
        mock_isfile.return_value = False
        mock_isdir.return_value = True
        
        # Set up mock os.walk to return a list of files
        mock_walk.return_value = [
            (test_dir, [], ["test.py", "test.pdf", "test.log", "node_modules/file.js"])
        ]
        
        # Configure loader with include/exclude patterns
        self.loader.loader_config.include_patterns = [".py", ".pdf"]
        self.loader.loader_config.exclude_patterns = ["node_modules", ".log"]
        
        # Act
        files = self.loader._get_files_from_directory(test_dir)
        
        # Assert
        self.assertEqual(len(files), 2)
        self.assertIn(os.path.join(test_dir, "test.py"), files)
        self.assertIn(os.path.join(test_dir, "test.pdf"), files)
        self.assertNotIn(os.path.join(test_dir, "test.log"), files)
        self.assertNotIn(os.path.join(test_dir, "node_modules/file.js"), files)


class TestDetectorDirectoryLoaderWithRealFiles(unittest.TestCase):
    """Test the integration with real files (optional)."""
    
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
        
        # Create a subdirectory with a file
        self.sub_dir = os.path.join(self.test_dir, "subdir")
        os.makedirs(self.sub_dir)
        self.sub_file = os.path.join(self.sub_dir, "subfile.py")
        with open(self.sub_file, "w") as f:
            f.write("print('Hello from subdir!')")
        
        # Create a detector and loader
        self.detector_config = DetectorConfig()
        self.loader_config = LoaderConfig(recursive=True)
        self.loader = EnhancedDirectoryLoader(
            detector_config=self.detector_config,
            loader_config=self.loader_config
        )
    
    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.test_dir)
    
    @unittest.skip("This test uses real files and should be run manually")
    def test_real_file_detection(self):
        """Test detection with real files."""
        # Act
        detector = EnhancedDetectorService(self.detector_config)
        python_type = detector.detect_type(self.python_file)
        text_type = detector.detect_type(self.text_file)
        
        # Assert
        self.assertEqual(python_type, DocumentType.CODE)
        self.assertEqual(text_type, DocumentType.DOCUMENT)
    
    @unittest.skip("This test uses real files and should be run manually")
    def test_real_directory_loading(self):
        """Test loading a real directory."""
        # Act
        docs = self.loader.load_documents(self.test_dir)
        
        # Assert
        self.assertEqual(len(docs), 3)  # 3 files total
        
        # Check that we have the correct document types
        code_docs = [doc for doc in docs if doc.metadata.get("file_type") == "code"]
        text_docs = [doc for doc in docs if doc.metadata.get("file_type") == "document"]
        
        self.assertEqual(len(code_docs), 2)  # 2 Python files
        self.assertEqual(len(text_docs), 1)  # 1 text file


if __name__ == '__main__':
    unittest.main()
