# --- processors/code/code_processor.py ---

from typing import List, Optional
import logging
import os

from llama_index.core import Document
from llama_index.core.schema import TextNode

from core.interfaces import IDocumentProcessor
from core.config import DocumentType, CodeProcessorConfig
from processors.code.code_splitter import CodeSplitterAdapter
from processors.code.metadata_generator import CodeMetadataGenerator
from llm.providers import ILLMProvider # Import provider interface

logger = logging.getLogger(__name__)


class CodeProcessor(IDocumentProcessor):
    """Processor for code documents."""
    
    def __init__(
        self, 
        llm_provider: ILLMProvider,
        config: Optional[CodeProcessorConfig] = None,
        llm_config=None
    ):
        """Initialize code processor with configuration.
        
        Args:
            config: Optional code processor configuration
            llm_provider: Optional LLM provider instance
            llm_config: Optional LLM configuration with enrichment settings
        """
        self.config = config or CodeProcessorConfig()
        self.llm_provider = llm_provider # Store provider
        self.llm_config = llm_config
        
        # Initialize components
        self.code_splitter = CodeSplitterAdapter(self.config)
        
        # Check if code enrichment is enabled in LLM config
        enrich_code = True  # Default if not specified
        if llm_config:
            enrich_code = getattr(llm_config, 'enrich_code', True)
        
        # Make metadata generator optional when LLM is not available or enrichment is disabled
        self.metadata_generator = None
        if enrich_code:
            try:
                metadata_llm = self.llm_provider.get_metadata_llm()
                if metadata_llm:
                    self.metadata_generator = CodeMetadataGenerator(metadata_llm)
                else:
                    logger.info("Metadata LLM is disabled in configuration. Skipping code enrichment.")
            except ValueError as e:
                logger.warning(f"Metadata generation will be skipped: {e}")
        else:
            logger.info("Code metadata enrichment is disabled in configuration.")
    
    def supports_document_type(self, document_type: DocumentType) -> bool:
        """Check if the processor supports a given document type.
        
        Args:
            document_type: The document type to check
            
        Returns:
            True if document_type is CODE, False otherwise
        """
        return document_type == DocumentType.CODE
    
    def process_document(self, document: Document) -> List[TextNode]:
        """Process a code document into text nodes.
        
        Args:
            document: The code document to process
            
        Returns:
            List of text nodes
        """
        # Split document into code chunks
        nodes = self.code_splitter.split(document)
        
        # Enrich nodes with metadata if generator is available
        if self.metadata_generator:
            self.metadata_generator.enrich(nodes)
        else:
            logger.info("Skipping metadata enrichment as no LLM is configured")
        
        # Apply formatting template
        self._apply_template(nodes)
        
        return nodes
    
    def process_documents(self, documents: List[Document]) -> List[TextNode]:
        """Process multiple code documents into text nodes.
        
        Args:
            documents: The code documents to process
            
        Returns:
            List of text nodes
        """
        all_nodes = []
        for document in documents:
            try:
                nodes = self.process_document(document)
                all_nodes.extend(nodes)
            except Exception as e:
                logger.error(f"Error processing document {document.doc_id}: {str(e)}")
        
        return all_nodes
    
    def _apply_template(self, nodes: List[TextNode]) -> None:
        """Apply formatting template to nodes.
        
        Args:
            nodes: The nodes to format
        """
        template = """TITLE: {title}
DESCRIPTION: {description}
SOURCE: {source}
LANGUAGE: {language}
CODE:
```{language}
{content}
```"""
        
        for node in nodes:
            if node.metadata.get("context7_format"):
                metadata_str = template.format(
                    title=node.metadata.get("title", "Untitled Code Snippet"),
                    description=node.metadata.get("description", "No description available"),
                    source=node.metadata.get("source", "Unknown source"),
                    language=node.metadata.get("language", "text"),
                    content=node.text
                )
                
                # Store the formatted metadata
                node.metadata["formatted_metadata"] = metadata_str
