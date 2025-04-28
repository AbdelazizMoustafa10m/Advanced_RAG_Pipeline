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

# --- LlamaIndex Imports ---
from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.readers.docling import DoclingReader
from llama_index.node_parser.docling import DoclingNodeParser

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

            # Initialize the node parser that understands Docling's JSON output
            self.node_parser = DoclingNodeParser(
                 # include_metadata=True # Default True - keeps Docling metadata
                 # include_prev_next_rel=True # Default True - adds relationships
            )
            logger.info("Docling components initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize Docling components: {e}", exc_info=True)
            # Set components to None so chunking will fail gracefully later
            self.reader = None
            self.node_parser = None
            # Depending on requirements, you might re-raise the exception

    def chunk_document(self, document: Document) -> List[TextNode]:
        """
        Chunks a single document using the configured Docling pipeline.

        Args:
            document: The llama_index Document object to process.
                      Requires 'file_path' in its metadata.

        Returns:
            A list of TextNode objects representing the structured chunks.
            Returns an empty list if processing fails.
        """
        if self.reader is None or self.node_parser is None:
            logger.error("DoclingChunker is not properly initialized. Cannot chunk.")
            return [] # Return empty list on initialization failure

        file_path = document.metadata.get("file_path")
        if not file_path or not os.path.exists(file_path):
             logger.error(f"DoclingChunker requires a valid 'file_path' in metadata. Skipping document ID: {document.doc_id or 'N/A'}")
             # Fallback: return single node with original text? Or empty list?
             # Returning empty list avoids potential issues downstream if metadata is crucial
             return []
        
        # Extract the original file path without any _part_X suffix
        # This handles cases where the loader splits a file into multiple parts
        original_file_path = file_path
        if "_part_" in file_path:
            # Extract the base file path without the _part_X suffix
            original_file_path = file_path.split("_part_")[0]
            logger.info(f"Detected document part: {file_path}, using original file: {original_file_path}")
        
        # Check if we've already processed this file
        if original_file_path in self.processed_files_cache:
            logger.info(f"Using cached chunks for {original_file_path}")
            cached_nodes = self.processed_files_cache[original_file_path]
            
            # Create a copy of the nodes with updated document-specific metadata
            document_specific_nodes = []
            for i, node in enumerate(cached_nodes):
                # Create a new node with the same content but updated metadata
                new_node = TextNode(
                    text=node.text,
                    metadata=dict(node.metadata),  # Copy metadata
                    id_=f"{document.doc_id}_docling_chunk_{i}"
                )
                # Update document-specific metadata
                new_node.metadata["doc_id"] = document.doc_id
                document_specific_nodes.append(new_node)
                
            return document_specific_nodes

        doc_format_str = os.path.splitext(original_file_path)[1].lower().lstrip('.')
        doc_format = DocumentFormat(doc_format_str) if doc_format_str in [fmt.value for fmt in DocumentFormat] else DocumentFormat.UNKNOWN

        logger.info(f"Starting Docling chunking for: {original_file_path}")
        try:
            # --- Step 1: Use DoclingReader to process the file ---
            # DoclingReader.load_data expects a list of file paths
            # It re-processes the file using the configured doc_converter
            docling_docs = self.reader.load_data(file_path=[original_file_path])

            if not docling_docs:
                logger.warning(f"DoclingReader returned no documents for {original_file_path}")
                return []

            # --- Step 2: Use DoclingNodeParser to extract structured nodes ---
            # This parser understands the JSON output from DoclingReader
            nodes = self.node_parser.get_nodes_from_documents(docling_docs)

            # --- Step 3: Post-process Nodes (Add/Standardize Metadata) ---
            for i, node in enumerate(nodes):
                # Ensure essential metadata from original doc is present
                node.metadata["file_path"] = original_file_path
                node.metadata["source_uri"] = original_file_path # Standardize source identifier
                node.metadata["content_type"] = "document" # Set content type
                node.metadata["document_format"] = doc_format.value # Add format
                node.metadata["doc_id"] = document.doc_id
                # Copy other relevant metadata if needed
                if "creation_date" in document.metadata:
                    node.metadata["document_creation_date"] = document.metadata["creation_date"]
                # DoclingNodeParser should already add 'page_label', 'headings', 'doc_items' etc.

                # Ensure node has a unique ID (DoclingNodeParser usually assigns one)
                if not node.node_id:
                     node.id_ = f"{document.doc_id}_docling_chunk_{i}"


            logger.info(f"Successfully chunked {original_file_path} into {len(nodes)} nodes using Docling.")
            
            # Cache the processed nodes to avoid reprocessing the same file
            self.processed_files_cache[original_file_path] = nodes
            
            return nodes

        except ImportError as ie:
             # Catch specific Docling/dependency import errors if they occur at runtime
             logger.error(f"Import error during Docling processing of {file_path}: {ie}. Ensure all Docling dependencies are installed.", exc_info=True)
             return []
        except Exception as e:
            logger.error(f"Error chunking {file_path} with Docling: {e}", exc_info=True)
            # Fallback or error reporting strategy:
            # Option 1: Return empty list
            # return []
            # Option 2: Return a single node with the original text + error metadata
            error_node = TextNode(
                 id_=document.doc_id + "_chunk_error",
                 text=f"Error processing document with Docling. Original text preview:\n\n{document.text[:500]}...",
                 metadata={**document.metadata, "processing_error": str(e)}
            )
            return [error_node]