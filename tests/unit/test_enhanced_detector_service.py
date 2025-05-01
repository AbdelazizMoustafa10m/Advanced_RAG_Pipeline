"""
Unit tests for the enhanced detector service.
Tests the service layer that manages document detection operations.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor

# Mock the magic module before importing any modules that depend on it
sys.modules['magic'] = MagicMock()

from core.config import DocumentType, DetectorConfig
from detectors.document_detector import DocumentDetector, DetectionResult
from detectors.enhanced_detector_service import EnhancedDetectorService


class TestEnhancedDetectorService(unittest.TestCase):
    """Test the EnhancedDetectorService class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.config = DetectorConfig(parallel_processing=True)
        self.service = EnhancedDetectorService(self.config)
    
    @patch.object(DocumentDetector, 'detect_type')
    def test_detect_type_delegates_to_detector(self, mock_detect):
        """Test that detect_type delegates to the underlying detector."""
        # Arrange
        test_file = "/path/to/test.py"
        mock_detect.return_value = DocumentType.CODE
        
        # Act
        with patch('os.path.exists', return_value=True):
            result = self.service.detect_type(test_file)
        
        # Assert
        self.assertEqual(result, DocumentType.CODE)
        mock_detect.assert_called_once_with(test_file)
    
    @patch.object(DocumentDetector, 'detect_with_metadata')
    def test_detect_with_metadata_delegates_to_detector(self, mock_detect):
        """Test that detect_with_metadata delegates to the underlying detector."""
        # Arrange
        test_file = "/path/to/test.py"
        expected_result = DetectionResult(
            document_type=DocumentType.CODE,
            confidence=0.9,
            detected_format="python"
        )
        mock_detect.return_value = expected_result
        
        # Act
        with patch('os.path.exists', return_value=True):
            result = self.service.detect_with_metadata(test_file)
        
        # Assert
        self.assertEqual(result, expected_result)
        mock_detect.assert_called_once_with(test_file)
    
    @patch.object(DocumentDetector, 'batch_detect')
    def test_batch_detect_filters_nonexistent_files(self, mock_batch_detect):
        """Test that batch_detect filters out nonexistent files."""
        # Arrange
        test_files = ["/path/to/test1.py", "/path/to/nonexistent.py"]
        mock_batch_detect.return_value = {test_files[0]: DocumentType.CODE}
        
        # Act
        with patch('os.path.exists', side_effect=[True, False]):
            results = self.service.batch_detect(test_files)
        
        # Assert
        self.assertEqual(len(results), 1)
        mock_batch_detect.assert_called_once_with([test_files[0]])
    
    @patch.object(DocumentDetector, 'batch_detect_with_metadata')
    def test_batch_detect_with_metadata_filters_nonexistent_files(self, mock_batch_detect):
        """Test that batch_detect_with_metadata filters out nonexistent files."""
        # Arrange
        test_files = ["/path/to/test1.py", "/path/to/nonexistent.py"]
        mock_batch_detect.return_value = {
            test_files[0]: DetectionResult(
                document_type=DocumentType.CODE,
                confidence=0.9,
                detected_format="python"
            )
        }
        
        # Act
        with patch('os.path.exists', side_effect=[True, False]):
            results = self.service.batch_detect_with_metadata(test_files)
        
        # Assert
        self.assertEqual(len(results), 1)
        mock_batch_detect.assert_called_once_with([test_files[0]])
    
    def test_filter_by_type(self):
        """Test filtering files by document type."""
        # Arrange
        test_files = ["/path/to/test1.py", "/path/to/test2.py", "/path/to/document.pdf"]
        
        # Act
        with patch.object(self.service, 'batch_detect') as mock_batch_detect:
            mock_batch_detect.return_value = {
                test_files[0]: DocumentType.CODE,
                test_files[1]: DocumentType.CODE,
                test_files[2]: DocumentType.DOCUMENT
            }
            
            code_files = self.service.filter_by_type(test_files, DocumentType.CODE)
            doc_files = self.service.filter_by_type(test_files, DocumentType.DOCUMENT)
        
        # Assert
        self.assertEqual(len(code_files), 2)
        self.assertEqual(len(doc_files), 1)
        self.assertIn(test_files[0], code_files)
        self.assertIn(test_files[1], code_files)
        self.assertIn(test_files[2], doc_files)
    
    def test_group_by_type(self):
        """Test grouping files by document type."""
        # Arrange
        test_files = ["/path/to/test1.py", "/path/to/document.pdf", "/path/to/unknown.xyz"]
        
        # Act
        with patch.object(self.service, 'batch_detect') as mock_batch_detect:
            mock_batch_detect.return_value = {
                test_files[0]: DocumentType.CODE,
                test_files[1]: DocumentType.DOCUMENT,
                test_files[2]: DocumentType.UNKNOWN
            }
            
            grouped = self.service.group_by_type(test_files)
        
        # Assert
        self.assertEqual(len(grouped[DocumentType.CODE]), 1)
        self.assertEqual(len(grouped[DocumentType.DOCUMENT]), 1)
        self.assertEqual(len(grouped[DocumentType.UNKNOWN]), 1)
        self.assertIn(test_files[0], grouped[DocumentType.CODE])
        self.assertIn(test_files[1], grouped[DocumentType.DOCUMENT])
        self.assertIn(test_files[2], grouped[DocumentType.UNKNOWN])
    
    @patch('utils.parallel.get_executor')
    def test_parallel_batch_detect(self, mock_get_executor):
        """Test parallel batch detection."""
        # Arrange
        test_files = [f"/path/to/test{i}.py" for i in range(5)]
        
        # Set up the mock executor
        mock_executor = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = DocumentType.CODE
        mock_executor.__enter__.return_value.submit.return_value = mock_future
        mock_get_executor.return_value = mock_executor
        
        # Act
        with patch('os.path.exists', return_value=True):
            results = self.service._parallel_batch_detect(test_files)
        
        # Assert
        self.assertEqual(len(results), 5)
        for file_path in test_files:
            self.assertEqual(results[file_path], DocumentType.CODE)
        
        # Verify executor was used correctly
        mock_get_executor.assert_called_once()
        self.assertEqual(mock_executor.__enter__.return_value.submit.call_count, 5)
    
    @patch('utils.parallel.get_executor')
    def test_parallel_batch_detect_with_metadata(self, mock_get_executor):
        """Test parallel batch detection with metadata."""
        # Arrange
        test_files = [f"/path/to/test{i}.py" for i in range(5)]
        
        # Set up the mock executor
        mock_executor = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = DetectionResult(
            document_type=DocumentType.CODE,
            confidence=0.9,
            detected_format="python"
        )
        mock_executor.__enter__.return_value.submit.return_value = mock_future
        mock_get_executor.return_value = mock_executor
        
        # Act
        with patch('os.path.exists', return_value=True):
            results = self.service._parallel_batch_detect_with_metadata(test_files)
        
        # Assert
        self.assertEqual(len(results), 5)
        for file_path in test_files:
            self.assertEqual(results[file_path].document_type, DocumentType.CODE)
            self.assertEqual(results[file_path].detected_format, "python")
            self.assertEqual(results[file_path].confidence, 0.9)
        
        # Verify executor was used correctly
        mock_get_executor.assert_called_once()
        self.assertEqual(mock_executor.__enter__.return_value.submit.call_count, 5)
    
    @patch('utils.parallel.get_executor')
    def test_parallel_batch_detect_handles_errors(self, mock_get_executor):
        """Test that parallel batch detection handles errors gracefully."""
        # Arrange
        test_files = ["/path/to/test1.py", "/path/to/error.py"]
        
        # Set up the mock executor
        mock_executor = MagicMock()
        mock_future1 = MagicMock()
        mock_future1.result.return_value = DocumentType.CODE
        
        mock_future2 = MagicMock()
        mock_future2.result.side_effect = Exception("Test error")
        
        mock_executor.__enter__.return_value.submit.side_effect = [mock_future1, mock_future2]
        mock_get_executor.return_value = mock_executor
        
        # Act
        with patch('os.path.exists', return_value=True):
            results = self.service._parallel_batch_detect(test_files)
        
        # Assert
        self.assertEqual(len(results), 2)
        self.assertEqual(results[test_files[0]], DocumentType.CODE)
        self.assertEqual(results[test_files[1]], DocumentType.UNKNOWN)  # Should default to UNKNOWN on error
    
    def test_sequential_vs_parallel_processing(self):
        """Test switching between sequential and parallel processing."""
        # Arrange
        test_files = [f"/path/to/test{i}.py" for i in range(5)]
        
        # Act - Sequential
        with patch.object(self.service, '_parallel_batch_detect') as mock_parallel, \
             patch.object(self.service.detector, 'batch_detect') as mock_sequential, \
             patch('os.path.exists', return_value=True):
            
            # Set parallel_processing to False
            self.service.config.parallel_processing = False
            self.service.batch_detect(test_files)
            
            # Set parallel_processing to True
            self.service.config.parallel_processing = True
            self.service.batch_detect(test_files)
        
        # Assert
        mock_sequential.assert_called_once_with(test_files)
        mock_parallel.assert_called_once_with(test_files)


if __name__ == '__main__':
    unittest.main()
