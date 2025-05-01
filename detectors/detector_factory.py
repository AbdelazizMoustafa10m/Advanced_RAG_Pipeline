# --- detectors/detector_factory.py ---

from typing import Optional, Type

from core.interfaces import IDocumentDetector
from core.config import DetectorConfig
from detectors.document_detector import DocumentDetector
from detectors.enhanced_detector_service import EnhancedDetectorService


class DetectorFactory:
    """Factory for creating document detectors."""
    
    @staticmethod
    def create_detector(detector_type: str, config: Optional[DetectorConfig] = None) -> IDocumentDetector:
        """Create a document detector of the specified type.
        
        Args:
            detector_type: Type of detector to create ('basic', 'enhanced', or 'comprehensive')
            config: Optional detector configuration
            
        Returns:
            Document detector instance
        """
        config = config or DetectorConfig()
        
        if detector_type.lower() == 'basic' or detector_type.lower() == 'simple':
            # Use the enhanced detector service for basic detection too
            # since FileExtensionDetector has been removed
            return EnhancedDetectorService(config)
        elif detector_type.lower() == 'enhanced':
            return EnhancedDetectorService(config)
        elif detector_type.lower() == 'comprehensive':
            return DocumentDetector(config)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
    
    @staticmethod
    def create_default_detector(config: Optional[DetectorConfig] = None) -> IDocumentDetector:
        """Create the default document detector.
        
        Args:
            config: Optional detector configuration
            
        Returns:
            Document detector instance
        """
        config = config or DetectorConfig()
        
        # Use the enhanced detector as the default
        return EnhancedDetectorService(config)
