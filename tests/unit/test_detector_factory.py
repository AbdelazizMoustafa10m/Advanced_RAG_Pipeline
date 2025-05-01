"""
Unit tests for the detector factory.
Tests the factory pattern implementation for creating detector instances.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys

# Mock the magic module before importing any modules that depend on it
sys.modules['magic'] = MagicMock()

from core.config import DetectorConfig
from core.interfaces import IDocumentDetector
from detectors.detector_factory import DetectorFactory
from detectors.file_extension import FileExtensionDetector
from detectors.document_detector import DocumentDetector
from detectors.enhanced_detector_service import EnhancedDetectorService


class TestDetectorFactory(unittest.TestCase):
    """Test the DetectorFactory class."""
    
    def test_create_detector_basic(self):
        """Test creating a basic detector."""
        # Act
        detector = DetectorFactory.create_detector('basic')
        
        # Assert
        self.assertIsInstance(detector, FileExtensionDetector)
        self.assertIsInstance(detector, IDocumentDetector)
    
    def test_create_detector_enhanced(self):
        """Test creating an enhanced detector."""
        # Act
        detector = DetectorFactory.create_detector('enhanced')
        
        # Assert
        self.assertIsInstance(detector, EnhancedDetectorService)
        self.assertIsInstance(detector, IDocumentDetector)
    
    def test_create_detector_comprehensive(self):
        """Test creating a comprehensive detector."""
        # Act
        detector = DetectorFactory.create_detector('comprehensive')
        
        # Assert
        self.assertIsInstance(detector, DocumentDetector)
        self.assertIsInstance(detector, IDocumentDetector)
    
    def test_create_detector_unknown_type(self):
        """Test handling of unknown detector types."""
        # Act & Assert
        with self.assertRaises(ValueError):
            DetectorFactory.create_detector('unknown_type')
    
    def test_create_default_detector(self):
        """Test creating the default detector."""
        # Act
        detector = DetectorFactory.create_default_detector()
        
        # Assert
        self.assertIsInstance(detector, EnhancedDetectorService)
        self.assertIsInstance(detector, IDocumentDetector)
    
    def test_create_detector_with_config(self):
        """Test creating a detector with a custom configuration."""
        # Arrange
        config = DetectorConfig(
            parallel_processing=True,
            min_confidence=0.75
        )
        
        # Act
        detector = DetectorFactory.create_detector('enhanced', config)
        
        # Assert
        self.assertIsInstance(detector, EnhancedDetectorService)
        self.assertEqual(detector.config.parallel_processing, True)
        self.assertEqual(detector.config.min_confidence, 0.75)
    
    def test_create_default_detector_with_config(self):
        """Test creating the default detector with a custom configuration."""
        # Arrange
        config = DetectorConfig(
            parallel_processing=False,
            min_confidence=0.6
        )
        
        # Act
        detector = DetectorFactory.create_default_detector(config)
        
        # Assert
        self.assertIsInstance(detector, EnhancedDetectorService)
        self.assertEqual(detector.config.parallel_processing, False)
        self.assertEqual(detector.config.min_confidence, 0.6)


if __name__ == '__main__':
    unittest.main()
