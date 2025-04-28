# loader/directory_loader.py
import os
import logging
from typing import List, Dict
from pathlib import Path
from llama_index.core import Document, SimpleDirectoryReader

from core.interfaces import IDocumentLoader
from core.config import LoaderConfig, DocumentType # Use Enums

logger = logging.getLogger(__name__)

class UnifiedDirectoryLoader(IDocumentLoader):
    """Loads documents from a directory and adds basic type metadata."""

    def __init__(self, config: LoaderConfig):
        self.config = config

    def supports_source(self, source: str) -> bool:
        # Supports directories
        return os.path.isdir(source)

    def load_documents(self, source: str) -> List[Document]:
        if not self.supports_source(source):
            raise ValueError(f"Source path must be a directory: {source}")
        if not os.path.exists(source):
            raise ValueError(f"Source path does not exist: {source}")

        all_docs: List[Document] = []
        input_dir_path = Path(source).resolve()

        logger.info(f"Scanning directory: {input_dir_path}")
        # Use SimpleDirectoryReader's file metadata feature
        def get_file_metadata(file_path: str) -> Dict:
            file_path_obj = Path(file_path).resolve()
            rel_path = file_path_obj.relative_to(input_dir_path)
            content_type = DocumentType.UNKNOWN
            # Basic routing based on top-level folder (adjust as needed)
            if rel_path.parts[0] == 'code_repository': # Example subdir name
                 content_type = DocumentType.CODE
            elif rel_path.parts[0] == 'technical_docs': # Example subdir name
                 content_type = DocumentType.DOCUMENT

            return {
                "file_path": str(file_path_obj), # Store absolute path
                "source_content_type": content_type.value # Store the string value
            }

        try:
            reader = SimpleDirectoryReader(
                input_dir=str(input_dir_path),
                recursive=self.config.recursive,
                file_extractor=self.config.file_extractors,
                exclude_hidden=True, # Usually good practice
                filename_as_id=True, # Use file path as ID
                required_exts=self.config.include_patterns if self.config.include_patterns else None,
                exclude=self.config.exclude_patterns if self.config.exclude_patterns else None,
                file_metadata=get_file_metadata # Apply our metadata function
            )
            all_docs = reader.load_data(show_progress=True)
            logger.info(f"Loaded {len(all_docs)} documents from {input_dir_path}")

        except Exception as e:
            logger.error(f"Error loading documents from {input_dir_path}: {e}", exc_info=True)
            # Depending on requirements, you might want to return partial results or raise
            raise

        return all_docs