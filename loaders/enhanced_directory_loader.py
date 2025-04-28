# --- enhanced_directory_loader.py ---

import os
import glob
import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict

from llama_index.core import Document
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import CodeSplitter

from core.interfaces import IDocumentLoader
from core.config import LoaderConfig, DocumentType, DetectorConfig, CodeProcessorConfig
from detectors.detector_service import DetectorService
from loaders.code_loader import CodeLoader
from loaders.directory_loader import UnifiedDirectoryLoader as DirectoryLoader
from processors.code.code_processor import CodeProcessor

logger = logging.getLogger(__name__)


class EnhancedDirectoryLoader(IDocumentLoader):
    """Enhanced loader that routes files to specialized loaders based on file type."""
    
    def __init__(
        self, 
        docling_reader: Optional[DoclingReader] = None,
        code_loader: Optional[CodeLoader] = None,
        detector_config: Optional[DetectorConfig] = None,
        loader_config: Optional[LoaderConfig] = None,
        code_processor_config: Optional[CodeProcessorConfig] = None,
        llm = None
    ):
        """Initialize enhanced directory loader with configuration.
        
        Args:
            docling_reader: Optional DoclingReader for document files
            code_loader: Optional CodeLoader for code files
            detector_config: Optional detector configuration
            loader_config: Optional loader configuration
            code_processor_config: Optional code processor configuration
            llm: Optional LLM instance for metadata enrichment
        """
        self.detector_config = detector_config or DetectorConfig()
        self.loader_config = loader_config or LoaderConfig()
        self.code_processor_config = code_processor_config or CodeProcessorConfig()
        self.llm = llm
        
        # Create detector service
        self.detector_service = DetectorService(self.detector_config)
        
        # Initialize specialized loaders
        self.docling_reader = docling_reader
        self.code_loader = code_loader or CodeLoader(self.loader_config)
        self.fallback_loader = DirectoryLoader(self.loader_config)
        
        # Initialize code processor
        self.code_processor = CodeProcessor(self.llm, self.code_processor_config)
    
    def supports_source(self, source: str) -> bool:
        """Check if the loader supports a given source.
        
        Args:
            source: Path to file or directory
            
        Returns:
            True if the source is a directory or file, False otherwise
        """
        return os.path.isdir(source) or os.path.isfile(source)
    
    def load_documents(self, source: str) -> List[Document]:
        """Load documents from a source using specialized loaders based on file type.
        
        Args:
            source: Path to directory or file
            
        Returns:
            List of loaded documents
            
        Raises:
            ValueError: If source doesn't exist
        """
        if not os.path.exists(source):
            raise ValueError(f"Source path does not exist: {source}")
        
        # If source is a file, detect its type and use appropriate loader
        if os.path.isfile(source):
            doc_type = self.detector_service.detect_type(source)
            return self._load_file_by_type(source, doc_type)
        
        # If source is a directory, get all files
        all_files = self._get_files_from_directory(source)
        
        # Detect file types
        file_types = self.detector_service.batch_detect(all_files)
        
        # Group files by type
        files_by_type = defaultdict(list)
        for file_path, doc_type in file_types.items():
            files_by_type[doc_type].append(file_path)
        
        # Load documents using appropriate loaders
        all_documents = []
        
        # Load document files using DoclingReader if available
        if DocumentType.DOCUMENT in files_by_type and self.docling_reader is not None:
            document_files = files_by_type[DocumentType.DOCUMENT]
            logger.info(f"Loading {len(document_files)} document files with DoclingReader")
            try:
                document_docs = self.docling_reader.load_data(file_path=document_files)
                # Add document type metadata
                for doc in document_docs:
                    doc.metadata["file_type"] = "document"
                all_documents.extend(document_docs)
                logger.info(f"Loaded {len(document_docs)} documents with DoclingReader")
            except Exception as e:
                logger.error(f"Error loading documents with DoclingReader: {str(e)}")
                # Fallback to DirectoryLoader for document files
                for file_path in document_files:
                    try:
                        docs = self.fallback_loader._load_file(file_path)
                        for doc in docs:
                            doc.metadata["file_type"] = "document"
                        all_documents.extend(docs)
                    except Exception as inner_e:
                        logger.error(f"Error loading document {file_path} with fallback loader: {str(inner_e)}")
        
        # Load code files using CodeLoader
        if DocumentType.CODE in files_by_type:
            code_files = files_by_type[DocumentType.CODE]
            logger.info(f"Loading {len(code_files)} code files with CodeLoader")
            for file_path in code_files:
                try:
                    code_docs = self.code_loader._load_file(file_path)
                    # Ensure file_type is set to code
                    for doc in code_docs:
                        doc.metadata["file_type"] = "code"
                    all_documents.extend(code_docs)
                except Exception as e:
                    logger.error(f"Error loading code file {file_path}: {str(e)}")
        
        # Load unknown files using fallback loader
        if DocumentType.UNKNOWN in files_by_type:
            unknown_files = files_by_type[DocumentType.UNKNOWN]
            logger.info(f"Loading {len(unknown_files)} unknown files with fallback loader")
            for file_path in unknown_files:
                try:
                    unknown_docs = self.fallback_loader._load_file(file_path)
                    for doc in unknown_docs:
                        doc.metadata["file_type"] = "unknown"
                    all_documents.extend(unknown_docs)
                except Exception as e:
                    logger.error(f"Error loading unknown file {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(all_documents)} documents in total")
        return all_documents
    
    def _get_files_from_directory(self, directory: str) -> List[str]:
        """Get all files from a directory, applying include/exclude patterns.
        
        Args:
            directory: Path to directory
            
        Returns:
            List of file paths
        """
        all_files = []
        
        # Walk through directory
        for root, _, files in os.walk(directory):
            # Skip excluded directories
            if any(excluded in root for excluded in self.loader_config.exclude_patterns):
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip excluded files
                if any(excluded in file for excluded in self.loader_config.exclude_patterns):
                    continue
                
                # Include only specific patterns if defined
                if self.loader_config.include_patterns:
                    file_ext = os.path.splitext(file)[1].lower()
                    if not any(file_ext.endswith(pattern) for pattern in self.loader_config.include_patterns):
                        continue
                
                all_files.append(file_path)
                
            # Stop if not recursive
            if not self.loader_config.recursive:
                break
        
        return all_files
    
    def _load_file_by_type(self, file_path: str, doc_type: DocumentType) -> List[Document]:
        """Load a file using the appropriate loader based on its type.
        
        Args:
            file_path: Path to the file
            doc_type: Detected document type
            
        Returns:
            List of loaded documents
        """
        try:
            if doc_type == DocumentType.DOCUMENT and self.docling_reader is not None:
                # Use DoclingReader for document files
                docs = self.docling_reader.load_data(file_path=[file_path])
                for doc in docs:
                    doc.metadata["file_type"] = "document"
                return docs
            elif doc_type == DocumentType.CODE:
                # Use CodeLoader for code files
                docs = self.code_loader._load_file(file_path)
                for doc in docs:
                    doc.metadata["file_type"] = "code"
                return docs
            else:
                # Use fallback loader for unknown files
                docs = self.fallback_loader._load_file(file_path)
                for doc in docs:
                    doc.metadata["file_type"] = "unknown"
                return docs
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            # Try fallback loader if primary loader fails
            try:
                docs = self.fallback_loader._load_file(file_path)
                for doc in docs:
                    doc.metadata["file_type"] = "unknown"
                return docs
            except Exception as inner_e:
                logger.error(f"Error loading file {file_path} with fallback loader: {str(inner_e)}")
                return []
