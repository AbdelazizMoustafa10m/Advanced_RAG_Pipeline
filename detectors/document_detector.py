# --- detectors/document_detector.py ---

import os
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Any

# Try to import magic, but make it optional
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    # Create a dummy magic module for type checking
    class DummyMagic:
        def from_file(self, file_path: str) -> str:
            return ""
    
    class magic:
        @staticmethod
        def Magic() -> Any:
            return DummyMagic()

from core.interfaces import IDocumentDetector
from core.config import DocumentType, DetectorConfig

logger = logging.getLogger(__name__)


class DetectionConfidence(Enum):
    """Confidence levels for document type detection."""
    HIGH = 0.9
    MEDIUM = 0.7
    LOW = 0.4
    NONE = 0.0


@dataclass
class DetectionResult:
    """Result of a document type detection."""
    document_type: DocumentType
    confidence: float
    detected_format: Optional[str] = None
    detected_language: Optional[str] = None
    metadata: Optional[Dict] = None

    def __lt__(self, other):
        """Compare detection results by confidence."""
        if not isinstance(other, DetectionResult):
            return NotImplemented
        return self.confidence < other.confidence


class BaseDetector(ABC):
    """Base class for all document type detectors."""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize detector with configuration.
        
        Args:
            config: Optional detector configuration
        """
        self.config = config or DetectorConfig()
        
    @abstractmethod
    def detect(self, file_path: str) -> DetectionResult:
        """Detect document type from file.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Detection result with document type and confidence
        """
        pass


class FileExtensionDetector(BaseDetector):
    """Detector that determines document type based on file extension."""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize detector with configuration.
        
        Args:
            config: Optional detector configuration
        """
        super().__init__(config)
        self.code_extensions = set(self.config.code_extensions)
        self.document_extensions = set(self.config.document_extensions)
        
        # Extended mappings for specific file types
        self.extension_format_map = {
            # Code files
            ".py": ("python", DocumentType.CODE),
            ".js": ("javascript", DocumentType.CODE),
            ".ts": ("typescript", DocumentType.CODE),
            ".java": ("java", DocumentType.CODE),
            ".cpp": ("cpp", DocumentType.CODE),
            ".c": ("c", DocumentType.CODE),
            ".h": ("c-header", DocumentType.CODE),
            ".hpp": ("cpp-header", DocumentType.CODE),
            ".go": ("go", DocumentType.CODE),
            ".cs": ("csharp", DocumentType.CODE),
            ".rb": ("ruby", DocumentType.CODE),
            ".php": ("php", DocumentType.CODE),
            ".rs": ("rust", DocumentType.CODE),
            ".swift": ("swift", DocumentType.CODE),
            ".kt": ("kotlin", DocumentType.CODE),
            ".scala": ("scala", DocumentType.CODE),
            
            # Document files
            ".pdf": ("pdf", DocumentType.DOCUMENT),
            ".docx": ("docx", DocumentType.DOCUMENT),
            ".doc": ("doc", DocumentType.DOCUMENT),
            ".md": ("markdown", DocumentType.DOCUMENT),
            ".markdown": ("markdown", DocumentType.DOCUMENT),
            ".txt": ("text", DocumentType.DOCUMENT),
            ".rtf": ("rtf", DocumentType.DOCUMENT),
            ".odt": ("odt", DocumentType.DOCUMENT),
            
            # Data files
            ".csv": ("csv", DocumentType.DOCUMENT),
            ".json": ("json", DocumentType.DOCUMENT),
            ".xml": ("xml", DocumentType.DOCUMENT),
            ".yaml": ("yaml", DocumentType.DOCUMENT),
            ".yml": ("yaml", DocumentType.DOCUMENT),
            
            # Presentation files
            ".pptx": ("pptx", DocumentType.DOCUMENT),
            ".ppt": ("ppt", DocumentType.DOCUMENT),
            
            # Web files
            ".html": ("html", DocumentType.DOCUMENT),
            ".htm": ("html", DocumentType.DOCUMENT),
            ".css": ("css", DocumentType.CODE),
        }
    
    def detect(self, file_path: str) -> DetectionResult:
        """Detect document type from file extension.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Detection result with document type and confidence
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Check specific format mapping first
        if file_ext in self.extension_format_map:
            format_info, doc_type = self.extension_format_map[file_ext]
            return DetectionResult(
                document_type=doc_type,
                confidence=DetectionConfidence.HIGH.value,
                detected_format=format_info
            )
        
        # Fall back to general extension sets
        if file_ext in self.code_extensions:
            return DetectionResult(
                document_type=DocumentType.CODE,
                confidence=DetectionConfidence.MEDIUM.value
            )
        elif file_ext in self.document_extensions:
            return DetectionResult(
                document_type=DocumentType.DOCUMENT,
                confidence=DetectionConfidence.MEDIUM.value
            )
        
        # Unknown extension
        return DetectionResult(
            document_type=DocumentType.UNKNOWN,
            confidence=DetectionConfidence.NONE.value
        )


class MagicNumberDetector(BaseDetector):
    """Detector that determines document type based on file magic numbers/MIME type."""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize detector with configuration.
        
        Args:
            config: Optional detector configuration
        """
        super().__init__(config)
        self.mime_type_map = {
            # Document types
            "application/pdf": (DocumentType.DOCUMENT, "pdf", DetectionConfidence.HIGH.value),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": 
                (DocumentType.DOCUMENT, "docx", DetectionConfidence.HIGH.value),
            "application/msword": (DocumentType.DOCUMENT, "doc", DetectionConfidence.HIGH.value),
            "text/markdown": (DocumentType.DOCUMENT, "markdown", DetectionConfidence.HIGH.value),
            "text/plain": (DocumentType.DOCUMENT, "text", DetectionConfidence.MEDIUM.value),
            "application/rtf": (DocumentType.DOCUMENT, "rtf", DetectionConfidence.HIGH.value),
            "application/vnd.oasis.opendocument.text": (DocumentType.DOCUMENT, "odt", DetectionConfidence.HIGH.value),
            
            # Data files
            "text/csv": (DocumentType.DOCUMENT, "csv", DetectionConfidence.HIGH.value),
            "application/json": (DocumentType.DOCUMENT, "json", DetectionConfidence.HIGH.value),
            "application/xml": (DocumentType.DOCUMENT, "xml", DetectionConfidence.HIGH.value),
            "text/xml": (DocumentType.DOCUMENT, "xml", DetectionConfidence.HIGH.value),
            "application/yaml": (DocumentType.DOCUMENT, "yaml", DetectionConfidence.HIGH.value),
            "text/yaml": (DocumentType.DOCUMENT, "yaml", DetectionConfidence.HIGH.value),
            
            # Presentation files
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": 
                (DocumentType.DOCUMENT, "pptx", DetectionConfidence.HIGH.value),
            "application/vnd.ms-powerpoint": (DocumentType.DOCUMENT, "ppt", DetectionConfidence.HIGH.value),
            
            # Web files
            "text/html": (DocumentType.DOCUMENT, "html", DetectionConfidence.HIGH.value),
            "text/css": (DocumentType.CODE, "css", DetectionConfidence.HIGH.value),
            
            # Code files - these are typically detected as text/plain,
            # so we'll handle them in the content detector
        }
    
    def detect(self, file_path: str) -> DetectionResult:
        """Detect document type from file magic numbers/MIME type.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Detection result with document type and confidence
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if magic module is available
        if not MAGIC_AVAILABLE:
            # Return low confidence result when magic is not available
            logger.warning("Magic module not available. Skipping magic number detection.")
            return DetectionResult(
                document_type=DocumentType.UNKNOWN,
                confidence=DetectionConfidence.NONE.value,
                detected_format=None
            )
        
        try:
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(file_path)
            
            if mime_type in self.mime_type_map:
                doc_type, format_type, confidence = self.mime_type_map[mime_type]
                return DetectionResult(
                    document_type=doc_type,
                    confidence=confidence,
                    detected_format=format_type
                )
            
            # For text files, we'll return a low confidence result
            # and let the content detector refine it
            if mime_type.startswith("text/"):
                return DetectionResult(
                    document_type=DocumentType.DOCUMENT,
                    confidence=DetectionConfidence.LOW.value,
                    detected_format="text"
                )
            
            # For binary files we can't specifically identify
            return DetectionResult(
                document_type=DocumentType.UNKNOWN,
                confidence=DetectionConfidence.LOW.value
            )
            
        except Exception as e:
            logger.warning(f"Error in magic number detection: {str(e)}")
            return DetectionResult(
                document_type=DocumentType.UNKNOWN,
                confidence=DetectionConfidence.NONE.value
            )


class ContentDetector(BaseDetector):
    """Detector that determines document type based on file content analysis."""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize detector with configuration.
        
        Args:
            config: Optional detector configuration
        """
        super().__init__(config)
        
        # Patterns to identify code files
        self.code_patterns = {
            # Python
            r"^(import|from)\s+\w+": ("python", DetectionConfidence.HIGH.value),
            r"^(def|class)\s+\w+": ("python", DetectionConfidence.HIGH.value),
            
            # JavaScript/TypeScript
            r"(const|let|var)\s+\w+\s*=": ("javascript", DetectionConfidence.HIGH.value),
            r"function\s+\w+\s*\(": ("javascript", DetectionConfidence.HIGH.value),
            r"export\s+(default\s+)?(class|function|const)": ("javascript", DetectionConfidence.HIGH.value),
            r"import\s+.*\s+from\s+['\"]": ("javascript", DetectionConfidence.HIGH.value),
            
            # Java/C#
            r"public\s+(class|void|int)": ("java", DetectionConfidence.HIGH.value),
            r"private\s+(class|void|int)": ("java", DetectionConfidence.HIGH.value),
            r"package\s+[\w\.]+;": ("java", DetectionConfidence.HIGH.value),
            r"using\s+[\w\.]+;": ("csharp", DetectionConfidence.HIGH.value),
            
            # C/C++
            r"#include\s+[<\"][\w\.]+[>\"]": ("cpp", DetectionConfidence.HIGH.value),
            r"int\s+main\s*\(": ("cpp", DetectionConfidence.MEDIUM.value),
            
            # Go
            r"package\s+\w+": ("go", DetectionConfidence.HIGH.value),
            r"func\s+\w+\s*\(": ("go", DetectionConfidence.HIGH.value),
            r"import\s+\(": ("go", DetectionConfidence.HIGH.value),
            
            # Ruby
            r"require\s+['\"][\w\/]+['\"]": ("ruby", DetectionConfidence.HIGH.value),
            r"def\s+\w+": ("ruby", DetectionConfidence.MEDIUM.value),
            r"class\s+\w+(\s+<\s+\w+)?": ("ruby", DetectionConfidence.HIGH.value),
            
            # PHP
            r"<\?php": ("php", DetectionConfidence.HIGH.value),
            r"function\s+\w+\s*\(": ("php", DetectionConfidence.MEDIUM.value),
            r"namespace\s+[\w\\]+;": ("php", DetectionConfidence.HIGH.value),
            
            # Rust
            r"fn\s+\w+\s*\(": ("rust", DetectionConfidence.HIGH.value),
            r"use\s+[\w:]+;": ("rust", DetectionConfidence.HIGH.value),
            r"struct\s+\w+\s*\{": ("rust", DetectionConfidence.HIGH.value),
            
            # Swift
            r"import\s+\w+": ("swift", DetectionConfidence.MEDIUM.value),
            r"class\s+\w+\s*:": ("swift", DetectionConfidence.HIGH.value),
            r"func\s+\w+\s*\(": ("swift", DetectionConfidence.HIGH.value),
            
            # Kotlin
            r"fun\s+\w+\s*\(": ("kotlin", DetectionConfidence.HIGH.value),
            r"class\s+\w+(\(.*\))?(\s*:\s*\w+)?": ("kotlin", DetectionConfidence.HIGH.value),
            r"package\s+[\w\.]+": ("kotlin", DetectionConfidence.HIGH.value),
        }
        
        # Patterns to identify document files
        self.document_patterns = {
            # Markdown
            r"^#\s+\w+": ("markdown", DetectionConfidence.HIGH.value),
            r"\[.*\]\(.*\)": ("markdown", DetectionConfidence.MEDIUM.value),
            
            # HTML
            r"<!DOCTYPE\s+html>": ("html", DetectionConfidence.HIGH.value),
            r"<html[>\s]": ("html", DetectionConfidence.HIGH.value),
            r"<body[>\s]": ("html", DetectionConfidence.MEDIUM.value),
            
            # XML
            r"<\?xml\s+version=": ("xml", DetectionConfidence.HIGH.value),
            r"<[a-zA-Z0-9]+:[a-zA-Z0-9]+": ("xml", DetectionConfidence.MEDIUM.value),
            
            # CSV
            r"^[\w\s]+,[\w\s]+,[\w\s]+": ("csv", DetectionConfidence.MEDIUM.value),
        }
    
    def detect(self, file_path: str) -> DetectionResult:
        """Detect document type from file content.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Detection result with document type and confidence
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # First check if it's a text file
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(file_path)
            
            if not mime_type.startswith("text/") and not mime_type in [
                "application/json", "application/xml", "application/javascript"
            ]:
                # Not a text file, skip content analysis
                return DetectionResult(
                    document_type=DocumentType.UNKNOWN,
                    confidence=DetectionConfidence.NONE.value
                )
            
            # Read a sample of the file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read up to 50 lines or 8KB, whichever comes first
                lines = []
                for _ in range(50):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)
                
                content_sample = ''.join(lines)
                if not content_sample:
                    return DetectionResult(
                        document_type=DocumentType.UNKNOWN,
                        confidence=DetectionConfidence.NONE.value
                    )
            
            # Check for code patterns
            for pattern, (language, confidence) in self.code_patterns.items():
                if re.search(pattern, content_sample, re.MULTILINE):
                    return DetectionResult(
                        document_type=DocumentType.CODE,
                        confidence=confidence,
                        detected_language=language
                    )
            
            # Check for document patterns
            for pattern, (format_type, confidence) in self.document_patterns.items():
                if re.search(pattern, content_sample, re.MULTILINE):
                    return DetectionResult(
                        document_type=DocumentType.DOCUMENT,
                        confidence=confidence,
                        detected_format=format_type
                    )
            
            # If no specific pattern matched, but it's a text file,
            # it's likely a generic document
            return DetectionResult(
                document_type=DocumentType.DOCUMENT,
                confidence=DetectionConfidence.LOW.value,
                detected_format="text"
            )
            
        except Exception as e:
            logger.warning(f"Error in content detection: {str(e)}")
            return DetectionResult(
                document_type=DocumentType.UNKNOWN,
                confidence=DetectionConfidence.NONE.value
            )


class DocumentDetector(IDocumentDetector):
    """Comprehensive document detector using multiple detection strategies."""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        """Initialize the document detector with configuration.
        
        Args:
            config: Optional detector configuration
        """
        self.config = config or DetectorConfig()
        self.detectors: List[BaseDetector] = []
        
        # Initialize detectors in order of preference
        self.detectors = [
            FileExtensionDetector(self.config)
        ]
        
        # Add MagicNumberDetector only if magic module is available
        if MAGIC_AVAILABLE:
            self.detectors.append(MagicNumberDetector(self.config))
        else:
            logger.warning("Magic module not available. MagicNumberDetector will not be used.")
            
        # Add ContentDetector only if content analysis is enabled
        if self.config.use_content_analysis:
            self.detectors.append(ContentDetector(self.config))
    
    def detect_type(self, file_path: str) -> DocumentType:
        """Detect document type using all configured detectors.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Detected document type
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        detection_results = []
        
        # Try each detector
        for detector in self.detectors:
            try:
                result = detector.detect(file_path)
                if result.confidence > 0:
                    detection_results.append(result)
            except Exception as e:
                logger.error(f"Error in detector {detector.__class__.__name__}: {str(e)}")
        
        # If no results, return UNKNOWN
        if not detection_results:
            return DocumentType.UNKNOWN
        
        # Sort by confidence (highest first)
        detection_results.sort(reverse=True)
        
        # Return the document type with highest confidence
        return detection_results[0].document_type
    
    def detect_with_metadata(self, file_path: str) -> DetectionResult:
        """Detect document type and return detailed metadata.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Detection result with document type, confidence, and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        detection_results = []
        
        # Try each detector
        for detector in self.detectors:
            try:
                result = detector.detect(file_path)
                if result.confidence > 0:
                    detection_results.append(result)
            except Exception as e:
                logger.error(f"Error in detector {detector.__class__.__name__}: {str(e)}")
        
        # If no results, return UNKNOWN with no confidence
        if not detection_results:
            return DetectionResult(
                document_type=DocumentType.UNKNOWN,
                confidence=DetectionConfidence.NONE.value
            )
        
        # Sort by confidence (highest first)
        detection_results.sort(reverse=True)
        
        # Get the highest confidence result
        best_result = detection_results[0]
        
        # Merge metadata from other high-confidence results
        merged_metadata = {}
        for result in detection_results:
            # Only consider results with reasonable confidence
            if result.confidence >= self.config.min_confidence:
                # Add format if not already set
                if not best_result.detected_format and result.detected_format:
                    best_result.detected_format = result.detected_format
                
                # Add language if not already set
                if not best_result.detected_language and result.detected_language:
                    best_result.detected_language = result.detected_language
                
                # Merge any additional metadata
                if result.metadata:
                    merged_metadata.update(result.metadata)
        
        # Add merged metadata
        if merged_metadata:
            best_result.metadata = merged_metadata
        
        return best_result
    
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
    
    def batch_detect_with_metadata(self, file_paths: List[str]) -> Dict[str, DetectionResult]:
        """Detect document types with metadata for multiple files.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dictionary mapping file paths to detection results
        """
        results = {}
        for file_path in file_paths:
            try:
                results[file_path] = self.detect_with_metadata(file_path)
            except Exception as e:
                logger.error(f"Error detecting type for {file_path}: {str(e)}")
                results[file_path] = DetectionResult(
                    document_type=DocumentType.UNKNOWN,
                    confidence=DetectionConfidence.NONE.value
                )
        
        return results
