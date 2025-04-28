# pipeline/orchestrator.py
import logging
import os
import time
import re
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
from llama_index.core.schema import Document, TextNode

from registry.document_registry import DocumentRegistry
from registry.status import ProcessingStatus

from core.config import UnifiedConfig, DocumentType
from core.interfaces import IDocumentLoader, IDocumentProcessor # Adjust imports
from detectors.detector_service import DetectorService
from loaders.enhanced_directory_loader import EnhancedDirectoryLoader # Use enhanced loader
from llama_index.readers.docling import DoclingReader
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption
from processors.code.code_processor import CodeProcessor
from processors.document.document_processor import TechnicalDocumentProcessor
from llm.providers import DefaultLLMProvider # Example provider
from llama_index.core.ingestion import IngestionPipeline
from llama_index.node_parser.docling import DoclingNodeParser
from processors.document_router import DocumentTypeRouter
from processors.code.metadata_generator import CodeMetadataGenerator
from processors.document.metadata_generator import DoclingMetadataGenerator

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Orchestrates the unified document processing pipeline."""

    def __init__(self, config: UnifiedConfig, document_registry=None, llm_provider=None):
        """Initialize the orchestrator with configuration."""
        self.config = config
        self.llm_provider = llm_provider or DefaultLLMProvider(config.llm)
        
        # Initialize document registry if provided
        self.document_registry = document_registry
        
        # Initialize DoclingReader for document files
        docling_reader = self._initialize_docling_reader(config.docling)
        
        # Initialize enhanced loader with specialized loaders
        self.loader = EnhancedDirectoryLoader(
            docling_reader=docling_reader,
            detector_config=config.detector,
            loader_config=config.loader,
            code_processor_config=config.code_processor,
            llm=self.llm_provider
        )
        
        # Initialize detector service
        self.detector = DetectorService(config.detector)
        
        # Initialize processors
        self.code_processor = CodeProcessor(
            self.llm_provider,
            config.code_processor,
            llm_config=config.llm  # Pass LLM config for enrichment flags
        )
        
        # Pass the correct config object to the document processor
        self.document_processor = TechnicalDocumentProcessor(
            config.docling,  # This is the DoclingConfig object
            self.llm_provider,
            llm_config=config.llm  # Pass LLM config for enrichment flags
        )
        
        # Store processed nodes for recovery in case of timeout or error
        self.processed_nodes: List[TextNode] = []
        
        # Track processing progress
        self.total_documents = 0
        self.processed_documents = 0
        self.processing_start_time = 0

        # Add more processors if needed

    def _initialize_docling_reader(self, docling_config) -> Optional[DoclingReader]:
        """Initialize the DoclingReader with appropriate configuration."""
        try:
            # Initialize Docling components based on config
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = docling_config.do_ocr
            pipeline_options.do_table_structure = docling_config.do_table_structure
            pipeline_options.table_structure_options.mode = (
                TableFormerMode.ACCURATE if docling_config.table_structure_mode == "ACCURATE"
                else TableFormerMode.FAST
            )
            pipeline_options.do_code_enrichment = docling_config.do_code_enrichment
            pipeline_options.do_formula_enrichment = docling_config.do_formula_enrichment

            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    # Add other format options as needed
                }
            )

            return DoclingReader(
                export_type=DoclingReader.ExportType.JSON,
                doc_converter=doc_converter
            )
        except Exception as e:
            logger.error(f"Failed to initialize DoclingReader: {e}")
            return None
            
    def _is_document_part(self, doc: Document) -> bool:
        """Check if a document is a part of a larger document."""
        file_path = doc.metadata.get('file_path', '')
        return '_part_' in file_path
        
    def _get_original_document_path(self, doc: Document) -> str:
        """Extract the original document path from a document part."""
        # For debugging purposes, log some information about the document
        logger.debug(f"Extracting path from document with ID: {doc.doc_id}")
        logger.debug(f"Document metadata keys: {list(doc.metadata.keys())}")
        
        # First check if this is a Docling document based on schema name
        schema_name = doc.metadata.get('schema_name', '')
        if schema_name and 'docling' in schema_name.lower():
            logger.debug(f"Found Docling document with schema: {schema_name}")
            
            # Check for origin metadata (this is where Docling stores filename)
            origin = doc.metadata.get('origin', {})
            if isinstance(origin, dict) and 'filename' in origin:
                filename = origin['filename']
                logger.info(f"Extracted filename '{filename}' from Docling document origin")
                return os.path.join(self.config.input_directory, filename)
            
            # If we couldn't find it in origin, check headings for potential document title
            headings = doc.metadata.get('headings', [])
            if headings and isinstance(headings, list) and len(headings) > 0:
                heading = str(headings[0])
                # Convert heading to a valid filename
                valid_filename = re.sub(r'[^\w\-_\.]', '_', heading) + '.pdf'
                logger.info(f"Using heading '{heading}' as filename: {valid_filename}")
                return os.path.join(self.config.input_directory, valid_filename)
        
        # Try standard metadata fields for file path
        for field in ['file_path', 'original_file_path', 'source']:
            file_path = doc.metadata.get(field, '')
            if file_path:
                logger.debug(f"Found path in {field}: {file_path}")
                # If we found a non-empty path, process it
                if '_part_' in file_path:
                    # Extract the original file path without the part suffix
                    match = re.match(r'^(.+?)_part_\d+$', file_path)
                    if match:
                        return match.group(1)
                return file_path
        
        # If we get here, no valid file path was found
        # Special handling for code files which often have language info
        language = doc.metadata.get('language', '')
        if language and language.lower() in ['python', 'javascript', 'java', 'c++', 'go']:
            file_name = doc.metadata.get('file_name', '')
            if file_name:
                logger.info(f"Using file_name for code document: {file_name}")
                return os.path.join(self.config.input_directory, file_name)
                    
        # Try to use the document ID or a fallback
        doc_id = doc.metadata.get('doc_id', '') or doc.id_
        if doc_id:
            logger.warning(f"Falling back to document ID for path: unknown_path_{doc_id}")
            return f"unknown_path_{doc_id}"
            
        # Last resort fallback
        fallback = f"unknown_path_{hash(doc.text[:100])}"
        logger.warning(f"Using last resort fallback path: {fallback}")
        return fallback
        
    def _group_document_parts(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Group document parts by their original document path."""
        document_groups = defaultdict(list)
        unknown_doc_counter = 0
        
        # Create a document tracker to associate document IDs with file types
        # This helps us maintain correct grouping without hardcoding filenames
        doc_type_tracker = {}
        pdf_paths_by_id = {}
        code_paths_by_id = {}
        
        # First, make an initial pass to identify document types
        for idx, doc in enumerate(documents):
            doc_id = doc.doc_id
            file_type = doc.metadata.get("file_type", "").lower()
            file_path = doc.metadata.get("file_path", "")
            
            # Determine document type
            if file_type == "code" or (file_path and any(file_path.lower().endswith(ext) for ext in [".py", ".js", ".java", ".c", ".cpp"])):
                doc_type_tracker[doc_id] = "code"
                if file_path and os.path.exists(file_path):
                    code_paths_by_id[doc_id] = file_path
            else:
                # Assume PDF/document if not code
                doc_type_tracker[doc_id] = "document"
        
        # Now look at filesystem to match documents with actual files
        pdf_files = []
        code_files = []
        
        for root, _, files in os.walk(self.config.input_directory):
            for file in files:
                full_path = os.path.join(root, file)
                if file.lower().endswith(".pdf"):
                    pdf_files.append(full_path)
                elif file.lower().endswith((".py", ".js", ".java", ".c", ".cpp")):
                    code_files.append(full_path)
        
        logger.info(f"Found {len(pdf_files)} PDF files and {len(code_files)} code files in directory")
        
        # Now group documents with their source files
        for doc in documents:
            doc_id = doc.doc_id
            doc_type = doc_type_tracker.get(doc_id, "unknown")
            
            if doc_type == "code":
                # For code documents: use existing path if available
                if doc_id in code_paths_by_id:
                    original_path = code_paths_by_id[doc_id]
                    logger.info(f"Using existing code path for document {doc_id}: {original_path}")
                else:
                    # Try to match with available code files
                    if code_files:
                        original_path = code_files[0]  # Use the first code file as default
                        # Try to find a better match by checking content
                        doc_content = str(doc.text)[:200]
                        for file_path in code_files:
                            if os.path.basename(file_path) in doc_content:
                                original_path = file_path
                                break
                    else:
                        # No code files found, create a placeholder
                        unknown_doc_counter += 1
                        original_path = f"code_document_{unknown_doc_counter}.py"
                
                logger.info(f"Using code path for document {doc_id}: {original_path}")
            else:  # document type
                # For PDF documents
                if pdf_files:
                    # Use the first PDF file we found - we'll group all PDF documents this way
                    original_path = pdf_files[0]
                else:
                    # Create a document ID based group instead if no PDF files found
                    original_path = f"document_{doc_id}.pdf"
                    
                logger.info(f"Using document path for document {doc_id}: {original_path}")
            
            # Set consistent metadata
            doc.metadata["file_type"] = doc_type
            doc.metadata["file_path"] = original_path
            doc.metadata["original_file_path"] = original_path
            
            # Add the document to its group
            document_groups[original_path].append(doc)
            
        return document_groups
    
    def run(self):
        """Runs the full ingestion pipeline."""
        start_time = time.time()
        self.processing_start_time = start_time
        logger.info(f"Starting unified pipeline run at {start_time}")
        
        # Reset any stalled documents if registry is available
        if self.document_registry and self.config.registry.enabled:
            reset_count = self.document_registry.reset_stalled_processing(
                max_processing_time=self.config.registry.reset_stalled_after_seconds
            )
            if reset_count > 0:
                logger.info(f"Reset {reset_count} stalled documents in registry")
        
        # 1. First, get a list of files without loading content
        input_dir = self.config.input_directory
        logger.info(f"Scanning for documents in {input_dir}...")
        
        documents_to_load = []
        skipped_files = []
        
        # For directory input, get all files
        if os.path.isdir(input_dir):
            all_files = []
            for root, _, files in os.walk(input_dir):
                for f in files:
                    file_path = os.path.join(root, f)
                    # Skip hidden files and directories
                    if not os.path.basename(file_path).startswith('.') and os.path.isfile(file_path):
                        # Store the absolute path for loading
                        all_files.append(file_path)
            
            # If registry is available, check each file before loading
            if self.document_registry and self.config.registry.enabled:
                documents_to_load = []
                for abs_path in all_files:
                    # For registry, we need the path format './data/file.ext'
                    # Extract just the base directory (data) and filename
                    base_dir = os.path.basename(input_dir)
                    rel_filename = os.path.basename(abs_path)
                    reg_path = f"./{base_dir}/{rel_filename}"
                    
                    try:
                        # Get file stats for a quick check
                        file_stats = os.stat(abs_path)
                        file_size = file_stats.st_size
                        file_mtime = file_stats.st_mtime
                        
                        # Get document status from registry
                        doc_status = self.document_registry.get_document_status(reg_path)
                        
                        if doc_status and doc_status.get("status") == ProcessingStatus.COMPLETED.value:
                            # Document exists and is completed, check if should process based on modification time
                            last_processed = doc_status.get("last_processed", 0)
                            
                            if last_processed > file_mtime:  # File hasn't changed since last processing
                                skipped_files.append(reg_path)
                                logger.info(f"Skipping already processed file (cached): {reg_path}")
                                continue
                            else:
                                logger.info(f"File has been modified since last processing: {reg_path}")
                    except Exception as e:
                        logger.warning(f"Error checking file status, will process: {reg_path}: {e}")
                    
                    # If we got here, we need to process this file
                    documents_to_load.append(abs_path)
            else:
                # No registry, load all files
                documents_to_load = all_files
        else:
            # Single file case
            documents_to_load = [input_dir]
        
        logger.info(f"Identified {len(documents_to_load)} documents to load, skipped {len(skipped_files)} already processed documents")
        
        # Skip loading if nothing to process
        if not documents_to_load:
            logger.info("No new documents to process. Pipeline completed with 0 nodes.")
            return []
            
        # Now load only the documents we need to process
        logger.info(f"Loading {len(documents_to_load)} documents...")
        documents = []
        for doc_path in documents_to_load:
            try:
                docs = self.loader.load_documents(doc_path)
                if docs:
                    documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading document {doc_path}: {e}")
        
        if not documents:
            logger.warning("No documents loaded successfully. Pipeline completed with 0 nodes.")
            return []
            
        self.total_documents = len(documents)
        logger.info(f"Loaded {len(documents)} documents")
        
        # Group document parts by their original document
        # This helps avoid redundant processing of the same document
        grouped_documents = self._group_document_parts(documents)
        
        # Count original documents (not parts)
        original_document_count = len(grouped_documents)
        logger.info(f"Detected {original_document_count} original documents (some split into parts)")
        
        # 2. Process documents using the document router
        all_processed_nodes = []
        
        #------------------------ Create an ingestion pipeline with the document router---------------------------#

        # Initialize DoclingNodeParser for document files
        docling_parser = DoclingNodeParser()
        
        # Create metadata enrichers if LLM is available and enabled
        document_metadata_enricher = None
        code_metadata_enricher = None
        
        if self.llm_provider:
            try:
                # Get the LLM instance
                metadata_llm = self.llm_provider.get_metadata_llm()
                if metadata_llm:
                    # Check enrichment flags in config
                    enrich_documents = getattr(self.config.llm, 'enrich_documents', True)
                    enrich_code = getattr(self.config.llm, 'enrich_code', True)
                    
                    # Create document metadata enricher if document enrichment is enabled
                    if enrich_documents:
                        document_metadata_enricher = DoclingMetadataGenerator(metadata_llm, num_questions=2)
                        logger.info("Initialized DoclingMetadataGenerator for document nodes")
                    else:
                        logger.info("Document metadata enrichment is disabled in configuration")
                    
                    # Create code metadata enricher if code enrichment is enabled
                    if enrich_code:
                        code_metadata_enricher = CodeMetadataGenerator(metadata_llm)
                        logger.info("Initialized CodeMetadataGenerator for code nodes")
                    else:
                        logger.info("Code metadata enrichment is disabled in configuration")
            except Exception as e:
                logger.warning(f"Failed to initialize metadata enrichers: {e}")
        
        # Create the router with our processors and enrichers
        document_router = DocumentTypeRouter(
            docling_parser=docling_parser, 
            code_processor=self.code_processor,
            document_metadata_enricher=document_metadata_enricher,
            code_metadata_enricher=code_metadata_enricher
        )
        
        # Create the pipeline with the router
        pipeline = IngestionPipeline(
            transformations=[
                document_router,  # This will route documents to the appropriate parser
            ]
        )
        
        # Process documents by group to maintain the grouping benefits
        for original_path, doc_group in grouped_documents.items():
            try:
                logger.info(f"Processing document group: {original_path}")
                logger.info(f"This group contains {len(doc_group)} document parts")
                
                # Check document registry if available
                if self.document_registry and self.config.registry.enabled:
                    # Take the first document to get content for hash check
                    sample_doc = doc_group[0]
                    doc_content = sample_doc.text
                    
                    # Check if we should process this document
                    if not self.document_registry.should_process(original_path, content=doc_content):
                        logger.info(f"Skipping already processed document: {original_path}")
                        continue
                    
                    # Register document and mark as processing
                    # Look for document type in various metadata fields with priority
                    doc_type = sample_doc.metadata.get("node_type", 
                                sample_doc.metadata.get("file_type", 
                                    sample_doc.metadata.get("source_content_type", "unknown")))
                    
                    # Normalize document type - ensure it's either 'code' or 'document'
                    if doc_type.lower() == "code" or ".py" in original_path.lower():
                        doc_type = "code"
                    else:
                        doc_type = "document"
                        
                    self.document_registry.register_document(
                        doc_id=original_path,
                        content=doc_content,
                        document_type=doc_type,
                        source=sample_doc.metadata.get("source", original_path),
                        metadata={
                            "document_type": doc_type,
                            "node_type": doc_type,  # Ensure consistency
                            "language": sample_doc.metadata.get("language", "unknown"),
                            "processor": sample_doc.metadata.get("processor", "unknown")
                        }
                    )
                    self.document_registry.update_status(original_path, ProcessingStatus.PROCESSING)
                
                # Process each document part through the router
                for doc in doc_group:
                    # Set the file_type in metadata to help the router
                    doc_type = doc.metadata.get("node_type", doc.metadata.get("file_type", doc.metadata.get("source_content_type", "unknown")))
                    if doc_type == "code":
                        doc.metadata["file_type"] = "code"
                        doc.metadata["node_type"] = "code"  # Ensure node_type is set
                    else:
                        doc.metadata["file_type"] = "document"
                        doc.metadata["node_type"] = "document"  # Ensure node_type is set
                
                # Process the document group through the pipeline
                processed_nodes = pipeline.run(documents=doc_group)
                
                # Add the processed nodes to our results
                all_processed_nodes.extend(processed_nodes)
                self.processed_nodes.extend(processed_nodes)  # Store for recovery
                
                # Update document registry with successful processing
                if self.document_registry and self.config.registry.enabled:
                    # Get the document type for metadata - use the same logic as when registering
                    doc_type = doc_group[0].metadata.get("node_type", 
                                doc_group[0].metadata.get("file_type", 
                                    doc_group[0].metadata.get("source_content_type", "unknown")))
                    
                    # Normalize document type
                    if doc_type.lower() == "code" or ".py" in original_path.lower():
                        doc_type = "code"
                        processor_name = "CodeProcessor"
                    else:
                        doc_type = "document"
                        processor_name = "TechnicalDocumentProcessor"
                    
                    # Update metadata with processing results
                    metadata_updates = {
                        "chunk_count": len(processed_nodes),
                        "processor": processor_name,
                        "processing_time": time.time() - self.processing_start_time
                    }
                    
                    # Mark document as successfully completed
                    self.document_registry.update_status(
                        original_path, 
                        ProcessingStatus.COMPLETED,
                        metadata_updates=metadata_updates
                    )
                    logger.info(f"Updated document registry for {original_path}: COMPLETED")
                
                # Update progress
                self.processed_documents += 1
                elapsed = time.time() - self.processing_start_time
                logger.info(f"Progress: {self.processed_documents}/{self.total_documents} documents processed in {elapsed:.2f}s")
                
                # Log estimated time remaining
                if self.processed_documents > 1:
                    docs_per_second = self.processed_documents / elapsed
                    remaining_docs = original_document_count - self.processed_documents
                    estimated_remaining = remaining_docs / docs_per_second if docs_per_second > 0 else 0
                    logger.info(f"Estimated time remaining: {estimated_remaining:.2f} seconds")
                
            except Exception as e:
                # Update document registry with failure status
                if self.document_registry and self.config.registry.enabled:
                    self.document_registry.update_status(
                        original_path,
                        ProcessingStatus.FAILED,
                        error_message=str(e)
                    )
                    logger.info(f"Updated document registry for {original_path}: FAILED")
                
                logger.error(f"Error processing document group {original_path}: {e}")
                # Continue with next document group
                continue
        
        # 3. Optional: Apply common post-processing/standardization enrichers if needed
        # enricher = Standardizer(...)
        # final_nodes = enricher.enrich(all_processed_nodes)
        
        final_nodes = all_processed_nodes  # Assuming no separate standardizer for now
        self.processed_nodes = final_nodes  # Store final result for recovery
        
        end_time = time.time()
        total_time = end_time - start_time
        nodes_per_second = len(final_nodes) / total_time if total_time > 0 else 0
        logger.info(f"Pipeline run completed. Generated {len(final_nodes)} nodes in {total_time:.2f} seconds.")
        logger.info(f"Processing efficiency: {nodes_per_second:.2f} nodes/second")
        
        return final_nodes