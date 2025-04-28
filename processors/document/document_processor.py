# processors/document/document_processor.py
import logging
import time
from typing import List, Optional, Dict, Set
from llama_index.core import Document
from llama_index.core.schema import TextNode
from core.interfaces import IDocumentProcessor, IDocumentChunker, IMetadataEnricher 
from core.config import DocumentType, DoclingConfig
from processors.document.docling_chunker import DoclingChunker
from processors.document.metadata_generator import DoclingMetadataGenerator
from llm.providers import ILLMProvider 

logger = logging.getLogger(__name__)

class TechnicalDocumentProcessor(IDocumentProcessor):
    """Processor for technical documents (PDF, DOCX, etc.)."""

    def __init__(self, config: DoclingConfig, llm_provider: Optional[ILLMProvider] = None, llm_config=None):
        """Initialize with configuration."""
        self.config = config
        self.llm_provider = llm_provider
        self.llm_config = llm_config
        
        # Initialize chunker
        self.chunker: IDocumentChunker = DoclingChunker(config)
        
        # Initialize metadata enricher if LLM is available
        self.enricher: Optional[IMetadataEnricher] = None
        
        # Check if document enrichment is enabled in LLM config
        enrich_documents = True  # Default if not specified
        if llm_config:
            enrich_documents = getattr(llm_config, 'enrich_documents', True)
            
        if enrich_documents and llm_provider:
            try:
                metadata_llm = llm_provider.get_metadata_llm()
                if metadata_llm:
                    self.enricher = DoclingMetadataGenerator(metadata_llm)
                else:
                    logger.info("Metadata LLM is disabled in configuration. Skipping enrichment.")
            except Exception as e:
                logger.warning(f"Failed to initialize metadata enricher: {e}")
                # Continue without enricher
        else:
            logger.info("Document metadata enrichment is disabled in configuration.")

    def supports_document_type(self, document_type: DocumentType) -> bool:
        return document_type == DocumentType.DOCUMENT

    def _get_original_file_path(self, document: Document) -> str:
        """Extract the original file path from a document."""
        file_path = document.metadata.get('file_path', document.doc_id or '')
        
        # Handle document parts (e.g., file.pdf_part_0)
        if '_part_' in file_path:
            # Extract the original file path without the part suffix
            original_path = file_path.split('_part_')[0]
            return original_path
        
        return file_path

    def process_document(self, document: Document) -> List[TextNode]:
        file_path = document.metadata.get('file_path', document.doc_id or '')
        logger.info(f"Processing document: {file_path}")
        
        # 1. Chunk document
        nodes = self.chunker.chunk_document(document)
        if not nodes:
             logger.warning(f"No nodes generated after chunking document: {file_path}")
             return []

        # 2. Enrich nodes with metadata if enricher is available
        if self.enricher:
            _ = self.enricher.enrich(nodes)
        else:
            logger.info("Skipping metadata enrichment as no LLM is configured")

        logger.info(f"Enriched {len(nodes)} nodes for document: {file_path}")
        return nodes

    def process_documents(self, documents: List[Document]) -> List[TextNode]:
        all_nodes = []
        for document in documents:
            if self.supports_document_type(DocumentType(document.metadata.get("source_content_type", "unknown"))):
                 try:
                     nodes = self.process_document(document)
                     all_nodes.extend(nodes)
                 except Exception as e:
                     logger.error(f"Failed to process document {document.doc_id}: {e}", exc_info=True)
            else:
                 logger.warning(f"Skipping document {document.doc_id}, type not supported by this processor.")
        return all_nodes