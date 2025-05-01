"""
Unit tests for the document detector module.
Tests individual detector components and their functionality.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import shutil
import sys

# Mock the magic module before importing any modules that depend on it
sys.modules['magic'] = MagicMock()

from core.config import DocumentType, DetectorConfig
from detectors.document_detector import (
    FileExtensionDetector, 
    MagicNumberDetector,
    ContentDetector,
    DocumentDetector,
    DetectionResult,
    DetectionConfidence
)


class TestFileExtensionDetector(unittest.TestCase):
    """Test the FileExtensionDetector class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.config = DetectorConfig()
        self.detector = FileExtensionDetector(self.config)
        
    def test_detect_python_file(self):
        """Test detection of Python files."""
        # Arrange
        test_file = "/path/to/test.py"
        
        # Act
        with patch('os.path.exists', return_value=True):
            result = self.detector.detect(test_file)
            
        # Assert
        self.assertEqual(result.document_type, DocumentType.CODE)
        self.assertEqual(result.detected_format, "python")
        self.assertGreaterEqual(result.confidence, 0.7)
        
    def test_detect_pdf_file(self):
        """Test detection of PDF files."""
        # Arrange
        test_file = "/path/to/document.pdf"
        
        # Act
        with patch('os.path.exists', return_value=True):
            result = self.detector.detect(test_file)
            
        # Assert
        self.assertEqual(result.document_type, DocumentType.DOCUMENT)
        self.assertEqual(result.detected_format, "pdf")
        self.assertGreaterEqual(result.confidence, 0.7)
        
    def test_detect_docx_file(self):
        """Test detection of DOCX files."""
        # Arrange
        test_file = "/path/to/document.docx"
        
        # Act
        with patch('os.path.exists', return_value=True):
            result = self.detector.detect(test_file)
            
        # Assert
        self.assertEqual(result.document_type, DocumentType.DOCUMENT)
        self.assertEqual(result.detected_format, "docx")
        self.assertGreaterEqual(result.confidence, 0.7)
        
    def test_detect_csv_file(self):
        """Test detection of CSV files."""
        # Arrange
        test_file = "/path/to/data.csv"
        
        # Act
        with patch('os.path.exists', return_value=True):
            result = self.detector.detect(test_file)
            
        # Assert
        self.assertEqual(result.document_type, DocumentType.DOCUMENT)
        self.assertEqual(result.detected_format, "csv")
        self.assertGreaterEqual(result.confidence, 0.7)
        
    def test_detect_html_file(self):
        """Test detection of HTML files."""
        # Arrange
        test_file = "/path/to/webpage.html"
        
        # Act
        with patch('os.path.exists', return_value=True):
            result = self.detector.detect(test_file)
            
        # Assert
        self.assertEqual(result.document_type, DocumentType.DOCUMENT)
        self.assertEqual(result.detected_format, "html")
        self.assertGreaterEqual(result.confidence, 0.7)
        
    def test_detect_unknown_extension(self):
        """Test detection of files with unknown extensions."""
        # Arrange
        test_file = "/path/to/unknown.xyz"
        
        # Act
        with patch('os.path.exists', return_value=True):
            result = self.detector.detect(test_file)
            
        # Assert
        self.assertEqual(result.document_type, DocumentType.UNKNOWN)
        self.assertLessEqual(result.confidence, 0.4)
        
    def test_nonexistent_file(self):
        """Test handling of nonexistent files."""
        # Arrange
        test_file = "/path/to/nonexistent.py"
        
        # Act & Assert
        with patch('os.path.exists', return_value=False):
            with self.assertRaises(FileNotFoundError):
                self.detector.detect(test_file)


class TestMagicNumberDetector(unittest.TestCase):
    """Test the MagicNumberDetector class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.config = DetectorConfig()
        self.detector = MagicNumberDetector(self.config)
    
    @patch('magic.Magic')
    def test_detect_pdf_by_magic(self, mock_magic):
        """Test detection of PDF files by magic number."""
        # Arrange
        test_file = "/path/to/file_without_extension"
        mock_magic_instance = MagicMock()
        mock_magic_instance.from_file.return_value = "PDF document, version 1.5"
        mock_magic.return_value = mock_magic_instance
        
        # Act
        with patch('os.path.exists', return_value=True):
            result = self.detector.detect(test_file)
        
        # Assert
        self.assertEqual(result.document_type, DocumentType.DOCUMENT)
        self.assertEqual(result.detected_format, "pdf")
        self.assertGreaterEqual(result.confidence, 0.8)
    
    @patch('magic.Magic')
    def test_detect_python_by_magic(self, mock_magic):
        """Test detection of Python files by magic number."""
        # Arrange
        test_file = "/path/to/script_without_extension"
        mock_magic_instance = MagicMock()
        mock_magic_instance.from_file.return_value = "Python script, ASCII text executable"
        mock_magic.return_value = mock_magic_instance
        
        # Act
        with patch('os.path.exists', return_value=True):
            result = self.detector.detect(test_file)
        
        # Assert
        self.assertEqual(result.document_type, DocumentType.CODE)
        self.assertEqual(result.detected_format, "python")
        self.assertGreaterEqual(result.confidence, 0.8)
        
    @patch('magic.Magic')
    def test_detect_docx_by_magic(self, mock_magic):
        """Test detection of DOCX files by magic number."""
        # Arrange
        test_file = "/path/to/doc_without_extension"
        mock_magic_instance = MagicMock()
        mock_magic_instance.from_file.return_value = "Microsoft Word 2007+"
        mock_magic.return_value = mock_magic_instance
        
        # Act
        with patch('os.path.exists', return_value=True):
            result = self.detector.detect(test_file)
        
        # Assert
        self.assertEqual(result.document_type, DocumentType.DOCUMENT)
        self.assertEqual(result.detected_format, "docx")
        self.assertGreaterEqual(result.confidence, 0.8)
        
    @patch('magic.Magic')
    def test_detect_binary_by_magic(self, mock_magic):
        """Test detection of binary files by magic number."""
        # Arrange
        test_file = "/path/to/binary_file"
        mock_magic_instance = MagicMock()
        mock_magic_instance.from_file.return_value = "data"
        mock_magic.return_value = mock_magic_instance
        
        # Act
        with patch('os.path.exists', return_value=True):
            result = self.detector.detect(test_file)
        
        # Assert
        self.assertEqual(result.document_type, DocumentType.UNKNOWN)
        self.assertLessEqual(result.confidence, 0.5)


class TestContentDetector(unittest.TestCase):
    """Test the ContentDetector class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.config = DetectorConfig()
        self.detector = ContentDetector(self.config)
    
    def test_detect_python_by_content(self):
        """Test detection of Python files by content."""
        # Arrange
        test_file = "/path/to/unknown_file"
        python_content = """
import os
import sys

def main():
    print("Hello, world!")

if __name__ == "__main__":
    main()
"""
        
        # Act
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', unittest.mock.mock_open(read_data=python_content)):
            result = self.detector.detect(test_file)
        
        # Assert
        self.assertEqual(result.document_type, DocumentType.CODE)
        self.assertEqual(result.detected_format, "python")
        self.assertGreaterEqual(result.confidence, 0.6)
    
    def test_detect_html_by_content(self):
        """Test detection of HTML files by content."""
        # Arrange
        test_file = "/path/to/unknown_file"
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Test Document</title>
</head>
<body>
    <h1>Hello, world!</h1>
</body>
</html>
"""
        
        # Act
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', unittest.mock.mock_open(read_data=html_content)):
            result = self.detector.detect(test_file)
        
        # Assert
        self.assertEqual(result.document_type, DocumentType.DOCUMENT)
        self.assertEqual(result.detected_format, "html")
        self.assertGreaterEqual(result.confidence, 0.6)
        
    def test_detect_markdown_by_content(self):
        """Test detection of Markdown files by content."""
        # Arrange
        test_file = "/path/to/unknown_file"
        md_content = """
# Heading 1

## Heading 2

This is a paragraph with **bold** and *italic* text.

- List item 1
- List item 2

```python
def hello():
    print("Hello, world!")
```
"""
        
        # Act
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', unittest.mock.mock_open(read_data=md_content)):
            result = self.detector.detect(test_file)
        
        # Assert
        self.assertEqual(result.document_type, DocumentType.DOCUMENT)
        self.assertEqual(result.detected_format, "markdown")
        self.assertGreaterEqual(result.confidence, 0.6)
        
    def test_detect_csv_by_content(self):
        """Test detection of CSV files by content."""
        # Arrange
        test_file = "/path/to/unknown_file"
        csv_content = """
id,name,age,email
1,John Doe,30,john@example.com
2,Jane Smith,25,jane@example.com
3,Bob Johnson,40,bob@example.com
"""
        
        # Act
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', unittest.mock.mock_open(read_data=csv_content)):
            result = self.detector.detect(test_file)
        
        # Assert
        self.assertEqual(result.document_type, DocumentType.DOCUMENT)
        self.assertEqual(result.detected_format, "csv")
        self.assertGreaterEqual(result.confidence, 0.6)
        
    def test_detect_empty_file(self):
        """Test detection of empty files."""
        # Arrange
        test_file = "/path/to/empty_file"
        
        # Act
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', unittest.mock.mock_open(read_data="")), \
             patch('os.path.getsize', return_value=0):
            result = self.detector.detect(test_file)
        
        # Assert
        self.assertEqual(result.document_type, DocumentType.UNKNOWN)
        self.assertEqual(result.confidence, DetectionConfidence.NONE.value)


class TestDocumentDetector(unittest.TestCase):
    """Test the DocumentDetector class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.config = DetectorConfig()
        self.detector = DocumentDetector(self.config)
    
    @patch('detectors.document_detector.FileExtensionDetector.detect')
    @patch('detectors.document_detector.MagicNumberDetector.detect')
    @patch('detectors.document_detector.ContentDetector.detect')
    def test_detect_type_combines_results(self, mock_content_detect, mock_magic_detect, mock_extension_detect):
        """Test that detect_type combines results from all detectors."""
        # Arrange
        test_file = "/path/to/test.py"
        
        # Set up detector mocks with different confidence levels
        mock_extension_detect.return_value = DetectionResult(
            document_type=DocumentType.CODE,
            confidence=0.7,
            detected_format="python"
        )
        
        mock_magic_detect.return_value = DetectionResult(
            document_type=DocumentType.CODE,
            confidence=0.9,
            detected_format="python"
        )
        
        mock_content_detect.return_value = DetectionResult(
            document_type=DocumentType.CODE,
            confidence=0.6,
            detected_format="python"
        )
        
        # Act
        with patch('os.path.exists', return_value=True):
            result = self.detector.detect_with_metadata(test_file)
        
        # Assert
        self.assertEqual(result.document_type, DocumentType.CODE)
        self.assertEqual(result.detected_format, "python")
        self.assertEqual(result.confidence, 0.9)  # Should take highest confidence
        
    @patch('detectors.document_detector.FileExtensionDetector.detect')
    @patch('detectors.document_detector.MagicNumberDetector.detect')
    @patch('detectors.document_detector.ContentDetector.detect')
    def test_detect_type_conflicting_results(self, mock_content_detect, mock_magic_detect, mock_extension_detect):
        """Test handling of conflicting detection results."""
        # Arrange
        test_file = "/path/to/ambiguous.txt"
        
        # Set up detector mocks with conflicting results
        mock_extension_detect.return_value = DetectionResult(
            document_type=DocumentType.DOCUMENT,
            confidence=0.5,
            detected_format="text"
        )
        
        mock_magic_detect.return_value = DetectionResult(
            document_type=DocumentType.DOCUMENT,
            confidence=0.6,
            detected_format="text"
        )
        
        mock_content_detect.return_value = DetectionResult(
            document_type=DocumentType.CODE,
            confidence=0.8,  # Highest confidence
            detected_format="python"
        )
        
        # Act
        with patch('os.path.exists', return_value=True):
            result = self.detector.detect_with_metadata(test_file)
        
        # Assert
        self.assertEqual(result.document_type, DocumentType.CODE)  # Should use highest confidence result
        self.assertEqual(result.detected_format, "python")
        self.assertEqual(result.confidence, 0.8)
        
    def test_batch_detect(self):
        """Test batch detection of multiple files."""
        # Arrange
        test_files = ["/path/to/test.py", "/path/to/document.pdf"]
        
        # Act
        with patch('os.path.exists', return_value=True), \
             patch.object(self.detector, 'detect_type') as mock_detect:
            mock_detect.side_effect = [DocumentType.CODE, DocumentType.DOCUMENT]
            results = self.detector.batch_detect(test_files)
        
        # Assert
        self.assertEqual(len(results), 2)
        self.assertEqual(results[test_files[0]], DocumentType.CODE)
        self.assertEqual(results[test_files[1]], DocumentType.DOCUMENT)
        
    def test_batch_detect_with_metadata(self):
        """Test batch detection with metadata for multiple files."""
        # Arrange
        test_files = ["/path/to/test.py", "/path/to/document.pdf"]
        
        # Act
        with patch('os.path.exists', return_value=True), \
             patch.object(self.detector, 'detect_with_metadata') as mock_detect:
            mock_detect.side_effect = [
                DetectionResult(document_type=DocumentType.CODE, confidence=0.9, detected_format="python"),
                DetectionResult(document_type=DocumentType.DOCUMENT, confidence=0.9, detected_format="pdf")
            ]
            results = self.detector.batch_detect_with_metadata(test_files)
        
        # Assert
        self.assertEqual(len(results), 2)
        self.assertEqual(results[test_files[0]].document_type, DocumentType.CODE)
        self.assertEqual(results[test_files[0]].detected_format, "python")
        self.assertEqual(results[test_files[1]].document_type, DocumentType.DOCUMENT)
        self.assertEqual(results[test_files[1]].detected_format, "pdf")
        
    def test_nonexistent_file(self):
        """Test handling of nonexistent files."""
        # Arrange
        test_file = "/path/to/nonexistent.py"
        
        # Act & Assert
        with patch('os.path.exists', return_value=False):
            with self.assertRaises(FileNotFoundError):
                self.detector.detect_type(test_file)


class TestDetectionResult(unittest.TestCase):
    """Test the DetectionResult class."""
    
    def test_comparison(self):
        """Test comparison of detection results by confidence."""
        # Arrange
        result1 = DetectionResult(document_type=DocumentType.CODE, confidence=0.8)
        result2 = DetectionResult(document_type=DocumentType.DOCUMENT, confidence=0.6)
        result3 = DetectionResult(document_type=DocumentType.UNKNOWN, confidence=0.3)
        
        # Assert
        self.assertLess(result3, result2)
        self.assertLess(result2, result1)
        self.assertFalse(result1 < result2)
        
    def test_equality(self):
        """Test equality of detection results."""
        # Arrange
        result1 = DetectionResult(document_type=DocumentType.CODE, confidence=0.8)
        result2 = DetectionResult(document_type=DocumentType.CODE, confidence=0.8)
        result3 = DetectionResult(document_type=DocumentType.DOCUMENT, confidence=0.8)
        
        # Assert
        self.assertFalse(result1 < result2)
        self.assertFalse(result2 < result1)
        # Different document types but same confidence should not be less than each other
        self.assertFalse(result1 < result3)
        self.assertFalse(result3 < result1)


if __name__ == '__main__':
    unittest.main()
