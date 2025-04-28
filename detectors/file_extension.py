# --- detectors/file_extension.py ---

import os
from typing import List, Dict, Optional
from pathlib import Path

from core.interfaces import IDocumentDetector
from core.config import DocumentType, DetectorConfig


class FileExtensionDetector(IDocumentDetector):
    """Detector that determines document type based on file extension."""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize detector with configuration.
        
        Args:
            config: Optional detector configuration
        """
        self.config = config or DetectorConfig()
        self.code_extensions = set(self.config.code_extensions)
        self.document_extensions = set(self.config.document_extensions)
    
    def detect_type(self, file_path: str) -> DocumentType:
        """Detect document type from file path extension.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Detected document type
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in self.code_extensions:
            return DocumentType.CODE
        elif file_ext in self.document_extensions:
            return DocumentType.DOCUMENT
        else:
            return DocumentType.UNKNOWN
