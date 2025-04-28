# --- detectors/detector_service.py ---

from typing import List, Dict, Optional
import os
import logging

from core.interfaces import IDocumentDetector
from core.config import DocumentType, DetectorConfig
from detectors.file_extension import FileExtensionDetector

logger = logging.getLogger(__name__)


class DetectorService:
    """Service for detecting document types using a chain of detectors."""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize detector service with configuration.
        
        Args:
            config: Optional detector configuration
        """
        self.config = config or DetectorConfig()
        self.detectors: List[IDocumentDetector] = []
        
        # Add detectors based on configuration
        if self.config.use_file_extension:
            self.detectors.append(FileExtensionDetector(self.config))
        
        # Add content-based detector if enabled
        if self.config.use_content_analysis:
            # TODO: Implement content-based detector
            pass
        
        if not self.detectors:
            logger.warning("No detectors configured. Using FileExtensionDetector by default.")
            self.detectors.append(FileExtensionDetector(self.config))
    
    def detect_type(self, file_path: str) -> DocumentType:
        """Detect document type using all configured detectors.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Detected document type
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Try each detector in sequence
        for detector in self.detectors:
            try:
                doc_type = detector.detect_type(file_path)
                if doc_type != DocumentType.UNKNOWN:
                    return doc_type
            except Exception as e:
                logger.error(f"Error in detector {detector.__class__.__name__}: {str(e)}")
        
        # If all detectors fail or return UNKNOWN, return UNKNOWN
        return DocumentType.UNKNOWN
    
    def batch_detect(self, file_paths: List[str]) -> Dict[str, DocumentType]:
        """Detect document types for multiple files.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary mapping file paths to detected types
        """
        results = {}
        for file_path in file_paths:
            try:
                results[file_path] = self.detect_type(file_path)
            except Exception as e:
                logger.error(f"Error detecting type for {file_path}: {str(e)}")
                results[file_path] = DocumentType.UNKNOWN
        
        return results
