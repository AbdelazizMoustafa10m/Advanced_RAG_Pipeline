# processors/document/docling_chunker.py

import os
import logging
from typing import List, Optional, Dict, Any

# --- Docling Core Imports ---
# Assuming 'docling' library is installed
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import PdfFormatOption
    from docling.datamodel.pipeline_options import AcceleratorDevice
    from docling.datamodel.pipeline_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import OcrOptions
    from docling.datamodel.pipeline_options import smolvlm_picture_description
    from docling.chunking import HybridChunker
    from docling.chunking import HierarchicalChunker
    _docling_installed = True
except ImportError:
    _docling_installed = False
    # Define dummy classes if docling is not installed to avoid import errors elsewhere
    # These won't actually work but allow the structure to load.
    class DocumentConverter: pass
    class PdfPipelineOptions: pass
    class TableFormerMode: ACCURATE="ACCURATE"; FAST="FAST" # Dummy enum values
    class InputFormat: PDF="pdf" # Dummy enum values
    class PdfFormatOption: pass
    class AcceleratorDevice: pass
    class AcceleratorOptions: pass
    class OcrOptions: pass
    class smolvlm_picture_description: pass

# --- LlamaIndex Imports ---
from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core.node_parser import SentenceSplitter


# --- Local Project Imports ---
from core.interfaces import IDocumentChunker
from core.config import DoclingConfig, DocumentFormat # Import Enums

logger = logging.getLogger(__name__)

class DoclingChunker(IDocumentChunker):
    """
    Chunks documents (PDF, DOCX etc.) using the Docling library via
    LlamaIndex readers and node parsers.
    """

    def __init__(self, config: Optional[DoclingConfig] = None):
        """
        Initializes the DoclingChunker with specific configuration.

        Args:
            config: Configuration object for Docling settings.
        """
        if not _docling_installed:
            logger.error("The 'docling-io' library is not installed. DoclingChunker cannot function.")
            logger.error("Please install it using: pip install docling-io")
            # Or handle this more gracefully, maybe disable this chunker
            raise ImportError("Docling library not found. Please install docling-io.")

        self.config = config or DoclingConfig()
        self.reader: Optional[DoclingReader] = None
        self.node_parser: Optional[DoclingNodeParser] = None
        
        # Extract chunking parameters from config with defaults
        self.chunk_size = getattr(self.config, 'chunk_size', 1000)
        self.chunk_overlap = getattr(self.config, 'chunk_overlap', 200)
        
        # Cache to store processed files to avoid reprocessing
        self.processed_files_cache: Dict[str, List[TextNode]] = {}

        try:
            # --- Initialize Docling Components based on config ---
            logger.info("Initializing Docling components...")
            pipeline_options = PdfPipelineOptions() # Assuming PDF focus for now
            
            # Safely access config attributes with defaults if not present
            if hasattr(self.config, 'do_ocr'):
                pipeline_options.do_ocr = self.config.do_ocr
            else:
                pipeline_options.do_ocr = True  # Default
                
            if hasattr(self.config, 'do_table_structure'):
                pipeline_options.do_table_structure = self.config.do_table_structure
            else:
                pipeline_options.do_table_structure = True  # Default
                
            # Map string config to Enum if necessary (depends on Docling library version)
            table_mode = getattr(self.config, 'table_structure_mode', 'ACCURATE')
            pipeline_options.table_structure_options.mode = (
                TableFormerMode.ACCURATE if table_mode == "ACCURATE"
                else TableFormerMode.FAST
            )
            
            if hasattr(self.config, 'do_code_enrichment'):
                pipeline_options.do_code_enrichment = self.config.do_code_enrichment
            else:
                pipeline_options.do_code_enrichment = True  # Default
                
            if hasattr(self.config, 'do_formula_enrichment'):
                pipeline_options.do_formula_enrichment = self.config.do_formula_enrichment
            else:
                pipeline_options.do_formula_enrichment = True  # Default
            
            # Configure OCR language to include English
            pipeline_options.ocr_options.lang = ["en"]
            
            # Configure accelerator options for performance
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=8, device=AcceleratorDevice.AUTO
            )
            
            # Enable picture description
            pipeline_options.do_picture_description = True
            
            # Set image scaling factor
            pipeline_options.images_scale = 1.0
            
            # Configure picture description options
            pipeline_options.picture_description_options = smolvlm_picture_description

            doc_converter = DocumentConverter(
                format_options={
                    # Configure primarily for PDF, add others if Docling supports them well
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    # Add DOCX, HTML etc. config here if needed/supported
                }
            )

            self.reader = DoclingReader(
                export_type=DoclingReader.ExportType.JSON, # Required for DoclingNodeParser
                doc_converter=doc_converter
            )

            # Determine which chunker to use based on the chunking_strategy configuration
            chunking_strategy = getattr(self.config, 'chunking_strategy', 'hybrid').lower()
            logger.info(f"Using chunking strategy: {chunking_strategy}")
            
            if chunking_strategy == 'hierarchical':
                chunker = HierarchicalChunker()
                logger.info("Using HierarchicalChunker for document processing")
            else:  # Default to 'hybrid'
                chunker = HybridChunker()
                logger.info("Using HybridChunker for document processing")
            
            # Initialize the node parser that understands Docling's JSON output with the selected chunker
            self.node_parser = DoclingNodeParser(chunker=chunker)
            logger.info("Docling components initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize Docling components: {e}", exc_info=True)
            # Set components to None so chunking will fail gracefully later
            self.reader = None
            self.node_parser = None
            # Depending on requirements, you might re-raise the exception

    def chunk_document(self, document: Document) -> List[TextNode]:
        """Chunk a document into text nodes.
        
        Args:
            document: Document to chunk
            
        Returns:
            List of text nodes
        """
        try:
            logger.info(f"Chunking document: {document.id_}")
            
            # Use the Docling node parser to create semantically meaningful chunks
            nodes = self.node_parser.get_nodes_from_documents([document])
            
            logger.info(f"Created {len(nodes)} chunks from document {document.id_}")
            return nodes
            
        except Exception as e:
            logger.error(f"Error chunking document {document.id_}: {str(e)}")
            
            # Fallback to sentence splitter if Docling parsing fails
            logger.warning(f"Falling back to sentence splitter for document {document.id_}")
            
            sentence_splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            logger.info(f"Using SentenceSplitter with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
            
            return sentence_splitter.get_nodes_from_documents([document])