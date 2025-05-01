# --- detectors/enhanced_detector_service.py ---

from typing import List, Dict, Optional, Set, Tuple
import os
import logging
from concurrent.futures import ThreadPoolExecutor

from core.interfaces import IDocumentDetector
from core.config import DocumentType, DetectorConfig
from detectors.document_detector import DocumentDetector, DetectionResult
from utils.parallel import get_executor

logger = logging.getLogger(__name__)


class EnhancedDetectorService:
    """Enhanced service for detecting document types with detailed metadata."""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize detector service with configuration.
        
        Args:
            config: Optional detector configuration
        """
        self.config = config or DetectorConfig()
        self.detector = DocumentDetector(self.config)
    
    def detect_type(self, file_path: str) -> DocumentType:
        """Detect document type using the comprehensive detector.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Detected document type
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return self.detector.detect_type(file_path)
    
    def detect_with_metadata(self, file_path: str) -> DetectionResult:
        """Detect document type and return detailed metadata.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Detection result with document type, confidence, and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return self.detector.detect_with_metadata(file_path)
    
    def batch_detect(self, file_paths: List[str]) -> Dict[str, DocumentType]:
        """Detect document types for multiple files.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary mapping file paths to detected types
        """
        # Filter out non-existent files
        valid_paths = [path for path in file_paths if os.path.exists(path)]
        if len(valid_paths) < len(file_paths):
            logger.warning(f"Skipping {len(file_paths) - len(valid_paths)} non-existent files")
        
        if not valid_paths:
            return {}
        
        # Use parallel processing if enabled
        if self.config.parallel_processing and len(valid_paths) > 1:
            return self._parallel_batch_detect(valid_paths)
        
        # Otherwise use sequential processing
        return self.detector.batch_detect(valid_paths)
    
    def batch_detect_with_metadata(self, file_paths: List[str]) -> Dict[str, DetectionResult]:
        """Detect document types with metadata for multiple files.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary mapping file paths to detection results
        """
        # Filter out non-existent files
        valid_paths = [path for path in file_paths if os.path.exists(path)]
        if len(valid_paths) < len(file_paths):
            logger.warning(f"Skipping {len(file_paths) - len(valid_paths)} non-existent files")
        
        if not valid_paths:
            return {}
        
        # Use parallel processing if enabled
        if self.config.parallel_processing and len(valid_paths) > 1:
            return self._parallel_batch_detect_with_metadata(valid_paths)
        
        # Otherwise use sequential processing
        return self.detector.batch_detect_with_metadata(valid_paths)
    
    def _parallel_batch_detect(self, file_paths: List[str]) -> Dict[str, DocumentType]:
        """Detect document types for multiple files in parallel.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary mapping file paths to detected types
        """
        results = {}
        
        with get_executor() as executor:
            # Submit all detection tasks
            future_to_path = {
                executor.submit(self.detect_type, path): path 
                for path in file_paths
            }
            
            # Collect results as they complete
            for future in future_to_path:
                path = future_to_path[future]
                try:
                    results[path] = future.result()
                except Exception as e:
                    logger.error(f"Error detecting type for {path}: {str(e)}")
                    results[path] = DocumentType.UNKNOWN
        
        return results
    
    def _parallel_batch_detect_with_metadata(self, file_paths: List[str]) -> Dict[str, DetectionResult]:
        """Detect document types with metadata for multiple files in parallel.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary mapping file paths to detection results
        """
        results = {}
        
        with get_executor() as executor:
            # Submit all detection tasks
            future_to_path = {
                executor.submit(self.detect_with_metadata, path): path 
                for path in file_paths
            }
            
            # Collect results as they complete
            for future in future_to_path:
                path = future_to_path[future]
                try:
                    results[path] = future.result()
                except Exception as e:
                    logger.error(f"Error detecting type for {path}: {str(e)}")
                    results[path] = DetectionResult(
                        document_type=DocumentType.UNKNOWN,
                        confidence=0.0
                    )
        
        return results
    
    def filter_by_type(self, file_paths: List[str], document_type: DocumentType) -> List[str]:
        """Filter files by document type.
        
        Args:
            file_paths: List of file paths
            document_type: Document type to filter by
            
        Returns:
            List of file paths matching the document type
        """
        results = self.batch_detect(file_paths)
        return [path for path, type_ in results.items() if type_ == document_type]
    
    def group_by_type(self, file_paths: List[str]) -> Dict[DocumentType, List[str]]:
        """Group files by document type.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary mapping document types to lists of file paths
        """
        results = self.batch_detect(file_paths)
        grouped = {doc_type: [] for doc_type in DocumentType}
        
        for path, type_ in results.items():
            grouped[type_].append(path)
        
        return grouped
