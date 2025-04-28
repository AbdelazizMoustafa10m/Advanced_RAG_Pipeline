# core/models.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from .config import DocumentType, CodeLanguage, DocumentFormat # Import enums

@dataclass
class QAPair:
    """Represents a single Question-Answer pair."""
    question: str
    answer: Optional[str] = None

@dataclass
class ProcessedChunk:
    """
    Standardized data model for a processed chunk of content
    (from code or documents) after metadata enrichment.
    """
    # --- Core Content & ID ---
    chunk_id: str
    content: str
    # --- Source Provenance Metadata ---
    source_uri: str # Typically the full file path
    
    # --- Fields with default values ---
    embedding: Optional[List[float]] = None
    document_format: DocumentFormat = DocumentFormat.UNKNOWN
    content_type: DocumentType = DocumentType.UNKNOWN # 'code' or 'document'

    # --- Structural Metadata ---
    parent_id: Optional[str] = None
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    page_label: Optional[str] = None # Doc specific
    headings: Optional[List[str]] = field(default_factory=list) # Doc specific
    code_language: Optional[CodeLanguage] = None # Code specific
    docling_label: Optional[str] = None # Doc specific (from Docling)
    start_line: Optional[int] = None # Line number in original file
    end_line: Optional[int] = None   # Line number in original file

    # --- Enriched Metadata (Standardized Output) ---
    title: Optional[str] = None          # Standardized title field
    summary: Optional[str] = None        # Standardized summary field
    questions: Optional[List[QAPair]] = field(default_factory=list) # Standardized Q&A
    keywords: Optional[List[str]] = field(default_factory=list)

    # --- Raw/Other Metadata ---
    other_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_llama_index_node(self) -> 'TextNode':
        """Converts this ProcessedChunk into a LlamaIndex TextNode."""
        from llama_index.core.schema import TextNode, RelatedNodeInfo # Local import

        # Prepare metadata dict for TextNode
        metadata_dict = {
            "source_uri": self.source_uri,
            "document_format": self.document_format.value,
            "content_type": self.content_type.value,
            "page_label": self.page_label,
            "headings": self.headings,
            "code_language": self.code_language.value if self.code_language else None,
            "docling_label": self.docling_label,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "title": self.title, # Use standardized key
            "summary": self.summary, # Use standardized key
            # Flatten questions for easier metadata storage/filtering
            "questions": [q.question for q in self.questions] if self.questions else [],
            "keywords": self.keywords,
            **self.other_metadata # Include other metadata
        }
        # Filter out None values
        metadata_dict = {k: v for k, v in metadata_dict.items() if v is not None and v != []}

        # Prepare relationships
        relationships_li = {
            RelatedNodeInfo.from_string(key): [RelatedNodeInfo(node_id=nid) for nid in id_list]
            for key, id_list in self.relationships.items()
        }

        return TextNode(
            id_=self.chunk_id,
            text=self.content,
            metadata=metadata_dict,
            embedding=self.embedding,
            relationships=relationships_li,
            # Define excluded keys if needed for embedding/LLM
            # excluded_embed_metadata_keys=["source_uri", ...],
            # excluded_llm_metadata_keys=["source_uri", ...],
        )