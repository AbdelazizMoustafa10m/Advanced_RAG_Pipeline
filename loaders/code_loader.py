# --- loaders/code_loader.py ---

import os
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

from llama_index.core import Document
from llama_index.core.schema import TextNode

from core.interfaces import IDocumentLoader
from core.config import LoaderConfig, CodeLanguage

logger = logging.getLogger(__name__)


class CodeLoader(IDocumentLoader):
    """Specialized loader for code files."""
    
    def __init__(self, config: Optional[LoaderConfig] = None):
        """Initialize code loader with configuration.
        
        Args:
            config: Optional loader configuration
        """
        self.config = config or LoaderConfig()
    
    def supports_source(self, source: str) -> bool:
        """Check if the loader supports a given source.
        
        Args:
            source: Path to file or directory
            
        Returns:
            True if the source is a file with a code extension, False otherwise
        """
        if not os.path.isfile(source):
            return False
        
        # Get file extension
        file_ext = os.path.splitext(source)[1].lower()
        
        # Check if extension is in code extensions list
        from core.config import DetectorConfig
        detector_config = DetectorConfig()
        return file_ext in detector_config.code_extensions
    
    def load_documents(self, source: str) -> List[Document]:
        """Load code documents from a source.
        
        Args:
            source: Path to file or directory containing code
            
        Returns:
            List of loaded documents
            
        Raises:
            ValueError: If source doesn't exist or isn't supported
        """
        if not os.path.exists(source):
            raise ValueError(f"Source path does not exist: {source}")
        
        # If source is a directory, process all supported files
        if os.path.isdir(source):
            documents = []
            for root, _, files in os.walk(source):
                for file in files:
                    file_path = os.path.join(root, file)
                    if self.supports_source(file_path):
                        try:
                            documents.extend(self._load_file(file_path))
                        except Exception as e:
                            logger.error(f"Error loading code from {file_path}: {str(e)}")
            return documents
        
        # If source is a file, process just that file
        if not self.supports_source(source):
            raise ValueError(f"Source file not supported: {source}")
        
        return self._load_file(source)
    
    def _load_file(self, file_path: str) -> List[Document]:
        """Load a single code file.
        
        Args:
            file_path: Path to the code file
            
        Returns:
            List containing the loaded document
        """
        try:
            # Detect language based on file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            language = self._detect_language(file_ext)
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create metadata
            metadata = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_type": "code",
                "language": language.value,
                "size_bytes": os.path.getsize(file_path),
                "line_count": content.count('\n') + 1
            }
            
            # Create document
            document = Document(
                text=content,
                metadata=metadata,
                id_=str(Path(file_path).absolute())
            )
            
            logger.info(f"Loaded code document from {file_path}")
            return [document]
            
        except Exception as e:
            logger.error(f"Error loading code from {file_path}: {str(e)}")
            raise
    
    def _detect_language(self, file_ext: str) -> CodeLanguage:
        """Detect code language based on file extension.
        
        Args:
            file_ext: File extension
            
        Returns:
            Detected code language
        """
        ext_to_language = {
            ".py": CodeLanguage.PYTHON,
            ".js": CodeLanguage.JAVASCRIPT,
            ".ts": CodeLanguage.TYPESCRIPT,
            ".java": CodeLanguage.JAVA,
            ".cpp": CodeLanguage.CPP,
            ".c": CodeLanguage.C,
            ".go": CodeLanguage.GO,
            ".cs": CodeLanguage.CSHARP,
            ".rb": CodeLanguage.RUBY,
            ".php": CodeLanguage.PHP,
            ".rs": CodeLanguage.RUST,
            ".swift": CodeLanguage.SWIFT,
            ".kt": CodeLanguage.KOTLIN
        }
        
        return ext_to_language.get(file_ext, CodeLanguage.UNKNOWN)
