# pipeline/orchestrator.py
import logging
import os
import time
import re
from typing import List, Dict, Optional, Set, Tuple, Any
from collections import defaultdict
from llama_index.core.schema import Document, TextNode

from registry.document_registry import DocumentRegistry
from registry.status import ProcessingStatus

from core.config import UnifiedConfig, DocumentType
from core.interfaces import IDocumentLoader, IDocumentProcessor # Adjust imports
from detectors.enhanced_detector_service import EnhancedDetectorService
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
from processors.document.docling_metadata_formatter import DoclingMetadataFormatter, FormattingConfig

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """
    Orchestrates the unified document ingestion pipeline.
    
    This class handles the entire document ingestion process: loading documents,
    detecting document types, processing documents through appropriate processors,
    generating embeddings, and indexing in a vector store. It follows the single
    responsibility principle by focusing solely on document ingestion without
    handling query operations.
    
    The pipeline follows these key steps:
    1. Document loading: Loads documents from the specified input directory
    2. Document detection: Determines document types using detection strategies
    3. Document routing: Routes documents to appropriate processors based on type
    4. Document processing: Processes documents into nodes with metadata
    5. Embedding: Generates vector embeddings for processed nodes
    6. Indexing: Stores nodes in a vector database for later retrieval
    
    The orchestrator returns both the processed nodes, the created index, and the
    vector store, allowing other components (like a separate query pipeline) to use
    these artifacts for retrieval and generation tasks.
    """

    def __init__(self, config: UnifiedConfig, document_registry=None, llm_provider=None):
        """Initialize the orchestrator with configuration."""
        self.config = config
        self.llm_provider = llm_provider or DefaultLLMProvider(config.llm)
        
        # Initialize document registry if provided
        self.document_registry = document_registry

        # Track processing progress
        self.total_documents = 0
        self.processed_documents = 0
        self.processing_start_time = 0

        # Store processed nodes for recovery in case of timeout or error
        self.processed_nodes: List[TextNode] = []

        logger.info("Initializing pipeline components...")
        
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
        logger.info("Initialized EnhancedDirectoryLoader")
        
        # Initialize enhanced detector service
        self.detector = EnhancedDetectorService(config.detector)
        logger.info("Initialized EnhancedDetectorService")
        
        # Initialize processors
        self.code_processor = CodeProcessor(
            self.llm_provider,
            config.code_processor,
            llm_config=config.llm  # Pass LLM config for enrichment flags
        )
        logger.info("Initialized CodeProcessor")
        # Pass the correct config object to the document processor
        self.document_processor = TechnicalDocumentProcessor(
            config.docling,  # This is the DoclingConfig object
            self.llm_provider,
            llm_config=config.llm  # Pass LLM config for enrichment flags
        )
        logger.info("Initialized TechnicalDocumentProcessor")

        # Initialize DoclingNodeParser for document files
        self.docling_parser = DoclingNodeParser()
        logger.info("Initialized DoclingNodeParser")
        
        # Initialize embedder service
        try:
            from embedders.embedder_factory import EmbedderFactory
            from indexing.vector_store import VectorStoreFactory
            self.embedder = EmbedderFactory.create_embedder(config.embedder)
            logger.info(f"Initialized Embedder Service with provider: {config.embedder.provider}, model: {config.embedder.model_name}")
            
            # Initialize vector store
            self.vector_store = VectorStoreFactory.create_vector_store(config.vector_store)
            logger.info(f"Initialized Vector Store with engine: {config.vector_store.engine}")
        except ImportError as e:
            logger.warning(f"Embedder or Vector Store modules not available: {str(e)}")
            self.embedder = None
            self.vector_store = None
        except Exception as e:
            logger.error(f"Error initializing embedder or vector store: {str(e)}")
            self.embedder = None
            self.vector_store = None

        # Get the enrichers from the respective processors
        document_metadata_enricher = getattr(self.document_processor, 'enricher', None)
        code_metadata_enricher = getattr(self.code_processor, 'metadata_generator', None)    

        # Log what enrichers we found
        if document_metadata_enricher:
            logger.info(f"Using document metadata enricher: {type(document_metadata_enricher).__name__}")
        else:
            logger.info("No document metadata enricher available (enrich_documents=False or LLM disabled)")

        if code_metadata_enricher:
            logger.info(f"Using code metadata enricher: {type(code_metadata_enricher).__name__}")
        else:
            logger.info("No code metadata enricher available (enrich_code=False or LLM disabled)")

        # Initialize the Docling metadata formatter (runs BEFORE router/enrichment)
        self.docling_formatter = DoclingMetadataFormatter(
            config=FormattingConfig(
                include_in_llm=['formatted_source', 'formatted_location', 'formatted_headings', 'formatted_label', 'file_type', 'node_type'],
                include_in_embed=['formatted_source', 'formatted_location', 'formatted_headings', 'formatted_label']
            )
        )
        logger.info("Initialized DoclingMetadataFormatter")    

        # Create the router with our processors and enrichers
        # The router itself will call the appropriate processor's process_document,
        # which internally handles enrichment if the enricher instance exists.
        self.document_router = DocumentTypeRouter(
            document_processor=self.document_processor,
            code_processor=self.code_processor, # Pass the whole processor
            # Pass enricher instances; router will check if they are None before calling enrich
            document_metadata_enricher=document_metadata_enricher,
            code_metadata_enricher=code_metadata_enricher
        )
        logger.info("Initialized DocumentTypeRouter")

        # Create the pipeline with the correct order of transformations
        self.pipeline = IngestionPipeline(
            transformations=[
                self.document_router,    # This will route documents to the appropriate parser
                self.docling_formatter,  # This will format Docling metadata after parsing
            ]
        )
        logger.info("Initialized complete IngestionPipeline")

        # Log initialization completion
        logger.info("PipelineOrchestrator initialization complete")

        # --- Start of Method Definitions ---

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
        
    def _load_documents(self, input_dir):
        """Load documents from the input directory with registry checking.
        
        Args:
            input_dir: Path to the input directory or file to process.
            
        Returns:
            tuple: (loaded_documents, skipped_count)
                - loaded_documents: List of loaded Document objects
                - skipped_count: Number of skipped files (already processed)
        """
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
                    # Get the proper relative path from input_dir to the file
                    rel_path = os.path.relpath(abs_path, os.path.dirname(input_dir))
                    reg_path = f"./{rel_path}"
                    
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
            return [], len(skipped_files)
            
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
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents, len(skipped_files)
        
    def _process_document_groups(self, grouped_documents: Dict[str, List[Document]], original_document_count: int) -> List[TextNode]:
        """Process document groups through the pipeline.
        
        Args:
            grouped_documents: Dictionary mapping original document paths to lists of document parts
            original_document_count: Total number of original documents (not parts)
            
        Returns:
            List[TextNode]: All processed nodes from all document groups
        """
        # Initialize result container
        all_processed_nodes = []
        
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
                processed_nodes = self.pipeline.run(documents=doc_group)
                
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
        
        return all_processed_nodes
            
    def _group_document_parts(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Group document parts by their original document path."""
        document_groups = defaultdict(list)
        unknown_doc_counter = 0
        
        # Get all files in the input directory
        all_files = []
        for root, _, files in os.walk(self.config.input_directory):
            for file in files:
                full_path = os.path.join(root, file)
                if os.path.isfile(full_path) and not os.path.basename(full_path).startswith('.'):
                    all_files.append(full_path)
        
        # Use the detector service to identify file types
        file_type_results = self.detector.batch_detect_with_metadata(all_files)
        
        # Group files by type for easier lookup
        files_by_type = defaultdict(list)
        for file_path, result in file_type_results.items():
            doc_type = result.document_type
            files_by_type[doc_type].append(file_path)
        
        # Log file counts by type
        type_counts = {doc_type.name: len(files) for doc_type, files in files_by_type.items()}
        logger.info(f"Detected file types: {type_counts}")
        
        # Create a mapping from document ID to file path and type
        doc_path_mapping = {}
        doc_type_mapping = {}
        
        # First, check if documents already have file paths
        for doc in documents:
            doc_id = doc.doc_id
            file_path = doc.metadata.get("file_path", "")
            
            if file_path and os.path.exists(file_path):
                # Use existing file path if available
                doc_path_mapping[doc_id] = file_path
                
                # Detect the type using our detector service
                if file_path in file_type_results:
                    doc_type = file_type_results[file_path].document_type
                else:
                    # If not in results, detect it now
                    detection_result = self.detector.detect_with_metadata(file_path)
                    doc_type = detection_result.document_type
                
                doc_type_mapping[doc_id] = doc_type
        
        # Now group documents with their source files
        for doc in documents:
            doc_id = doc.doc_id
            
            # If we already mapped this document, use that information
            if doc_id in doc_path_mapping:
                original_path = doc_path_mapping[doc_id]
                doc_type = doc_type_mapping[doc_id]
                logger.info(f"Using existing path for document {doc_id}: {original_path}")
            else:
                # Try to match with detected files based on content
                doc_content = str(doc.text)[:500].lower()
                matched = False
                
                # Try to find a matching file by content
                for doc_type, file_paths in files_by_type.items():
                    for file_path in file_paths:
                        file_name = os.path.basename(file_path).lower()
                        if file_name in doc_content:
                            original_path = file_path
                            matched = True
                            logger.info(f"Matched document {doc_id} to file {original_path} by content")
                            break
                    if matched:
                        break
                
                # If no match by content, try to match by type
                if not matched:
                    # Get the document type from metadata if available
                    metadata_type = doc.metadata.get("file_type", "").lower()
                    
                    # Map metadata type to DocumentType enum
                    if metadata_type == "code":
                        target_type = DocumentType.CODE
                    elif metadata_type == "markdown":
                        target_type = DocumentType.MARKDOWN
                    elif metadata_type == "pdf" or metadata_type == "document":
                        target_type = DocumentType.DOCUMENT
                    else:
                        # Try to detect from content
                        if "```" in doc_content or "def " in doc_content or "class " in doc_content:
                            target_type = DocumentType.CODE
                        elif "#" in doc_content[:10] or "markdown" in doc_content:
                            target_type = DocumentType.MARKDOWN
                        else:
                            target_type = DocumentType.DOCUMENT
                    
                    # Find a file of the matching type
                    if target_type in files_by_type and files_by_type[target_type]:
                        original_path = files_by_type[target_type][0]
                        doc_type = target_type
                        logger.info(f"Matched document {doc_id} to file {original_path} by type")
                    else:
                        # Create a placeholder path
                        unknown_doc_counter += 1
                        if target_type == DocumentType.CODE:
                            original_path = f"code_document_{unknown_doc_counter}.py"
                        elif target_type == DocumentType.MARKDOWN:
                            original_path = f"markdown_document_{unknown_doc_counter}.md"
                        else:
                            original_path = f"document_{unknown_doc_counter}.txt"
                        doc_type = target_type
                        logger.info(f"Created placeholder path for document {doc_id}: {original_path}")
            
            # Set consistent metadata
            doc.metadata["document_type"] = doc_type.name.lower()
            doc.metadata["file_path"] = original_path
            doc.metadata["original_file_path"] = original_path
            
            # Add file extension to metadata for better type identification
            file_ext = os.path.splitext(original_path)[1].lower()
            if file_ext:
                doc.metadata["file_extension"] = file_ext
            
            # Add the document to its group
            document_groups[original_path].append(doc)
            
        return document_groups
    
    def run(self):
        """
        Runs the full ingestion pipeline.
        
        This method orchestrates the entire document ingestion process:
        1. Loads documents from the configured input directory
        2. Groups document parts by their original source
        3. Processes document groups through appropriate processors
        4. Embeds the resulting nodes (if embedder is available)
        5. Indexes the nodes in a vector store (if available)
        
        Returns:
            tuple: (final_nodes, index, vector_store)
                - final_nodes (List[TextNode]): All processed nodes
                - index (VectorStoreIndex or None): The created index (if indexing was successful)
                - vector_store (IVectorStore or None): The vector store instance (if available)
        """
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
        
        # 1. Load documents with registry checking
        documents, skipped_count = self._load_documents(self.config.input_directory)
        
        # Return early if no documents to process
        if not documents:
            logger.info("No documents to process. Pipeline completed with 0 nodes.")
            return []
            
        self.total_documents = len(documents)
        
        # Group document parts by their original document
        # This helps avoid redundant processing of the same document
        grouped_documents = self._group_document_parts(documents)
        
        # Count original documents (not parts)
        original_document_count = len(grouped_documents)
        logger.info(f"Detected {original_document_count} original documents (some split into parts)")
        
        # 2. Process documents using the document router
        all_processed_nodes = self._process_document_groups(grouped_documents, original_document_count)
        
        final_nodes = all_processed_nodes  # Assuming no separate standardizer for now
        
        # 3. Embed nodes if embedder is available
        if self.embedder and final_nodes:
            try:
                logger.info(f"Embedding {len(final_nodes)} nodes with {self.config.embedder.provider} provider")
                embed_start_time = time.time()
                
                # Embed the nodes
                embedded_nodes = self.embedder.embed_nodes(final_nodes)
                
                embed_end_time = time.time()
                embed_time = embed_end_time - embed_start_time
                embed_nodes_per_second = len(embedded_nodes) / embed_time if embed_time > 0 else 0
                
                logger.info(f"Embedding completed in {embed_time:.2f} seconds")
                logger.info(f"Embedding efficiency: {embed_nodes_per_second:.2f} nodes/second")
                
                # Count nodes with embeddings
                nodes_with_embeddings = sum(1 for node in embedded_nodes if node.embedding is not None and len(node.embedding) > 0)
                logger.info(f"Nodes with embeddings: {nodes_with_embeddings}/{len(embedded_nodes)}")
                
                final_nodes = embedded_nodes
                
                # 4. Index nodes in vector store if available
                if self.vector_store and nodes_with_embeddings > 0:
                    try:
                        index_start_time = time.time()
                        logger.info(f"Indexing {nodes_with_embeddings} nodes in vector store with engine: {self.config.vector_store.engine}")
                        
                        # Create index from nodes
                        index = self.vector_store.create_index(final_nodes)
                        
                        # Persist the index
                        self.vector_store.persist()
                        
                        index_end_time = time.time()
                        index_time = index_end_time - index_start_time
                        index_nodes_per_second = nodes_with_embeddings / index_time if index_time > 0 else 0
                        
                        logger.info(f"Indexing completed in {index_time:.2f} seconds")
                        logger.info(f"Indexing efficiency: {index_nodes_per_second:.2f} nodes/second")
                    except Exception as e:
                        logger.error(f"Error indexing nodes in vector store: {str(e)}")
                        # Continue without indexing
                elif self.vector_store and nodes_with_embeddings == 0:
                    logger.warning("No nodes with embeddings to index in vector store")
                elif not self.vector_store:
                    logger.warning("Vector store not available. Nodes will not be indexed.")
            except Exception as e:
                logger.error(f"Error embedding nodes: {str(e)}")
                # Continue with unembedded nodes
        elif not self.embedder:
            logger.warning("Embedder not available. Nodes will not be embedded.")
            # Cannot index without embeddings
            logger.warning("Vector store indexing skipped due to missing embeddings.")
        
        self.processed_nodes = final_nodes  # Store final result for recovery
        
        end_time = time.time()
        total_time = end_time - start_time
        nodes_per_second = len(final_nodes) / total_time if total_time > 0 else 0
        logger.info(f"Pipeline run completed. Generated {len(final_nodes)} nodes in {total_time:.2f} seconds.")
        logger.info(f"Processing efficiency: {nodes_per_second:.2f} nodes/second")
        
        # Return the processed nodes, the created index, and the vector store for use by query components
        return final_nodes, index, self.vector_store