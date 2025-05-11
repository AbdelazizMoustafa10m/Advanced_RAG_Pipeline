"""Document router for routing documents to appropriate processors."""

from typing import Any, List, Optional
import logging
import os
from llama_index.core.schema import TransformComponent, Document, TextNode
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

class DocumentTypeRouter(TransformComponent):
    """Routes documents to the appropriate parser based on their type."""
    
    document_processor: Any = Field(description="Processor for document files")
    code_processor: Any = Field(description="Processor for code files")
    document_metadata_enricher: Optional[Any] = Field(default=None, description="Metadata enricher for document files")
    code_metadata_enricher: Optional[Any] = Field(default=None, description="Metadata enricher for code files")
    
    def __call__(self, nodes: List[Document], **kwargs) -> List[TextNode]:
        """Route documents to the appropriate parser.
        
        Args:
            nodes: List of documents to process
            
        Returns:
            List of processed text nodes
        """
        all_nodes = []
        document_nodes = []
        code_nodes = []
        
        for doc in nodes:
            # Check document type from metadata and file extension
            doc_type = doc.metadata.get("file_type", "unknown")
            
            # Additional check for code files based on file extension
            file_path = doc.metadata.get("file_path", "")
            file_ext = os.path.splitext(file_path)[1].lower() if file_path else ""
            
            # Override doc_type for known code file extensions
            code_extensions = [".py", ".js", ".java", ".cpp", ".c", ".h", ".cs", ".go", ".rb", ".php"]
            if file_ext in code_extensions:
                doc_type = "code"
                doc.metadata["file_type"] = "code"
                
            if doc_type == "document":
                # Use document processor for document files
                try:
                    parsed_nodes = self.document_processor.process_document(doc)
                    logger.info(f"Successfully processed document {doc.doc_id} with TechnicalDocumentProcessor")
                    
                    # Add node type metadata for enrichment
                    for node in parsed_nodes:
                        node.metadata["node_type"] = "document"
                    
                    # Add to document nodes for later enrichment
                    document_nodes.extend(parsed_nodes)
                    all_nodes.extend(parsed_nodes)
                except Exception as e:
                    logger.error(f"Error processing document {doc.doc_id} with TechnicalDocumentProcessor: {str(e)}")
                    # Fallback to code processor if docling parser fails
                    try:
                        parsed_nodes = self.code_processor.process_document(doc)
                        # Add to code nodes for later enrichment
                        code_nodes.extend(parsed_nodes)
                        all_nodes.extend(parsed_nodes)
                    except Exception as code_e:
                        logger.error(f"Fallback to code processor also failed: {str(code_e)}")
                        # Create a simple text node as final fallback
                        fallback_node = TextNode(text=doc.text, metadata=doc.metadata)
                        fallback_node.metadata["node_type"] = "unknown"
                        all_nodes.append(fallback_node)
            elif doc_type == "code" or doc_type == "unknown":
                # Use our enhanced code processor for code files and unknown files
                try:
                    # Use process_document method from our CodeProcessor
                    parsed_nodes = self.code_processor.process_document(doc)
                    logger.info(f"Successfully processed code document {doc.doc_id} with CodeProcessor")
                    
                    # Add node type metadata for enrichment
                    for node in parsed_nodes:
                        node.metadata["node_type"] = "code"
                    
                    # Add to code nodes for later enrichment
                    code_nodes.extend(parsed_nodes)
                    all_nodes.extend(parsed_nodes)
                except Exception as e:
                    logger.error(f"Error processing code document {doc.doc_id}: {str(e)}")
                    # Create a simple text node as fallback
                    fallback_node = TextNode(text=doc.text, metadata=doc.metadata)
                    fallback_node.metadata["node_type"] = "unknown"
                    all_nodes.append(fallback_node)
        
        # Enrich document nodes with metadata if enricher is available
        if document_nodes and self.document_metadata_enricher:
            try:
                logger.info(f"Enriching {len(document_nodes)} document nodes with metadata")
                self.document_metadata_enricher.enrich(document_nodes)
                logger.info("Document metadata enrichment completed")
            except Exception as e:
                logger.error(f"Error during document metadata enrichment: {str(e)}")
        
        # Enrich code nodes with metadata if enricher is available
        if code_nodes and self.code_metadata_enricher:
            try:
                logger.info(f"Enriching {len(code_nodes)} code nodes with metadata")
                self.code_metadata_enricher.enrich(code_nodes)
                logger.info("Code metadata enrichment completed")
            except Exception as e:
                logger.error(f"Error during code metadata enrichment: {str(e)}")
        
        return all_nodes
