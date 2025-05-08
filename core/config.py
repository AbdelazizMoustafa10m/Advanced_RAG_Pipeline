"""
Pydantic models for Advanced RAG Pipeline configuration.

This module contains Pydantic models that replace the original dataclasses
from core/config.py, providing better validation, documentation, and flexibility.
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
import os
import logging
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, AnyHttpUrl

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Enum for document types."""
    CODE = "code"
    DOCUMENT = "document"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class CodeLanguage(str, Enum):
    """Enum for code languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    RUST = "rust"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    OTHER = "other"
    UNKNOWN = "unknown"


class DocumentFormat(str, Enum):
    """Enum for document formats."""
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    HTML = "html"
    TXT = "txt"
    OTHER = "other"
    UNKNOWN = "unknown"


class FilterOperator(str, Enum):
    """Operators for metadata filters."""
    EQ = "eq"  # Equal
    NE = "ne"  # Not equal
    GT = "gt"  # Greater than
    GTE = "gte"  # Greater than or equal
    LT = "lt"  # Less than
    LTE = "lte"  # Less than or equal
    IN = "in"  # In list
    NIN = "nin"  # Not in list
    CONTAINS = "contains"  # String contains
    STARTSWITH = "startswith"  # String starts with
    ENDSWITH = "endswith"  # String ends with


class FilterCondition(str, Enum):
    """Conditions for combining multiple filters."""
    AND = "and"
    OR = "or"


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Logging format string"
    )
    file_path: Optional[str] = Field(default=None, description="Path to log file")
    console: bool = Field(default=True, description="Whether to log to console")


class ParallelConfig(BaseModel):
    """Configuration for parallel processing."""
    enabled: bool = Field(default=True, description="Whether parallel processing is enabled")
    max_workers: int = Field(
        default=4, 
        description="Maximum number of worker processes or threads",
        ge=1,
        le=32
    )


class DetectorConfig(BaseModel):
    """Configuration for document type detection."""
    use_file_extension: bool = Field(
        default=True, 
        description="Use file extension for detection"
    )
    use_content_analysis: bool = Field(
        default=False, 
        description="Use content analysis for detection"
    )
    use_magic_numbers: bool = Field(
        default=True, 
        description="Use magic numbers for detection"
    )
    parallel_processing: bool = Field(
        default=True,
        description="Process detection in parallel"
    )

    # File extension mappings
    code_extensions: List[str] = Field(
        default=[
            ".py", ".js", ".ts", ".java", ".cpp", ".go", ".cs", ".rb",
            ".php", ".rs", ".swift", ".kt", ".c", ".h", ".hpp", ".scala"
        ],
        description="List of code file extensions"
    )

    document_extensions: List[str] = Field(
        default=[
            ".pdf", ".docx", ".md", ".html", ".txt", ".rtf", ".odt", ".csv",
            ".json", ".xml", ".yaml", ".yml", ".pptx", ".ppt"
        ],
        description="List of document file extensions"
    )

    # Data file extensions
    data_extensions: List[str] = Field(
        default=[
            ".csv", ".json", ".xml", ".yaml", ".yml"
        ],
        description="List of data file extensions"
    )

    # Presentation file extensions
    presentation_extensions: List[str] = Field(
        default=[
            ".pptx", ".ppt", ".odp", ".key"
        ],
        description="List of presentation file extensions"
    )

    # Web file extensions
    web_extensions: List[str] = Field(
        default=[
            ".html", ".htm", ".css", ".js"
        ],
        description="List of web file extensions"
    )

    # Confidence thresholds for detection
    min_confidence: float = Field(
        default=0.7, 
        description="Minimum confidence threshold",
        ge=0.0,
        le=1.0
    )
    high_confidence: float = Field(
        default=0.9, 
        description="High confidence threshold",
        ge=0.0,
        le=1.0
    )
    medium_confidence: float = Field(
        default=0.7, 
        description="Medium confidence threshold",
        ge=0.0,
        le=1.0
    )
    low_confidence: float = Field(
        default=0.4, 
        description="Low confidence threshold",
        ge=0.0,
        le=1.0
    )

    model_config = ConfigDict(
        validate_default=True
    )

    @model_validator(mode="after")
    def validate_confidence_thresholds(self) -> "DetectorConfig":
        """Validate that confidence thresholds are in the correct order."""
        if not (self.low_confidence <= self.medium_confidence <= self.high_confidence):
            raise ValueError(
                "Confidence thresholds must be in ascending order: "
                "low_confidence <= medium_confidence <= high_confidence"
            )
        return self


class LoaderConfig(BaseModel):
    """Configuration for document loading."""
    recursive: bool = Field(
        default=True, 
        description="Recursively load documents from subdirectories"
    )
    exclude_patterns: List[str] = Field(
        default=[
            ".*", "node_modules", "__pycache__", "venv", ".git"
        ],
        description="Patterns to exclude during loading"
    )
    include_patterns: List[str] = Field(
        default=[],
        description="Patterns to include during loading (empty means include all)"
    )

    # File extraction configurations
    file_extractors: Dict[str, str] = Field(
        default={},
        description="Custom file extractors by extension (e.g., {'.pdf': 'PdfExtractor'})"
    )


class CodeProcessorConfig(BaseModel):
    """Configuration for code processing."""
    # Chunking parameters
    chunk_lines: int = Field(
        default=60, 
        description="Number of lines per chunk",
        gt=0
    )
    chunk_overlap_lines: int = Field(
        default=15, 
        description="Number of overlapping lines between chunks",
        ge=0
    )
    max_chars: int = Field(
        default=2000, 
        description="Maximum characters per chunk",
        gt=0
    )
    chunk_size: int = Field(
        default=512, 
        description="Token size for Chonkie chunker",
        gt=0
    )

    # Chunking strategy options
    chunking_strategies: List[str] = Field(
        default=[
            "chonkie_ast",  # AST-based chunking with Chonkie (highest priority)
            "llamaindex_ast",  # AST-based chunking with LlamaIndex
            "semantic_line",  # Semantic line-based chunking
            "basic_line"  # Basic line-based chunking with overlap
        ],
        description="Chunking strategies in order of preference"
    )

    # Language detection
    language_detection: str = Field(
        default="auto", 
        description="Code language detection method ('auto' or specific language)"
    )

    # Embedding parameters
    include_imports: bool = Field(
        default=True, 
        description="Include import statements in code chunks"
    )
    include_comments: bool = Field(
        default=True, 
        description="Include comments in code chunks"
    )
    include_docstrings: bool = Field(
        default=True, 
        description="Include docstrings in code chunks"
    )


class DoclingConfig(BaseModel):
    """Configuration for Docling document processing."""
    # OCR settings
    do_ocr: bool = Field(
        default=True, 
        description="Enable OCR for documents"
    )

    # Table extraction settings
    do_table_structure: bool = Field(
        default=True, 
        description="Extract table structure from documents"
    )
    table_structure_mode: str = Field(
        default="ACCURATE", 
        description="Table structure extraction mode ('ACCURATE' or 'FAST')"
    )

    # Enrichment settings
    do_code_enrichment: bool = Field(
        default=True, 
        description="Enrich code blocks in documents"
    )
    do_formula_enrichment: bool = Field(
        default=True, 
        description="Enrich formula blocks in documents"
    )

    # Chunking settings
    chunking_strategy: str = Field(
        default="semantic", 
        description="Chunking strategy ('semantic', 'fixed', or 'hierarchical')"
    )
    chunk_size: int = Field(
        default=1000, 
        description="Maximum chunk size in characters",
        gt=0
    )
    chunk_overlap: int = Field(
        default=200, 
        description="Overlap between chunks in characters",
        ge=0
    )

    @field_validator("table_structure_mode")
    @classmethod
    def validate_table_structure_mode(cls, v: str) -> str:
        """Validate table structure mode."""
        if v not in ["ACCURATE", "FAST"]:
            raise ValueError("Table structure mode must be 'ACCURATE' or 'FAST'")
        return v

    @field_validator("chunking_strategy")
    @classmethod
    def validate_chunking_strategy(cls, v: str) -> str:
        """Validate chunking strategy."""
        if v not in ["semantic", "fixed", "hierarchical"]:
            raise ValueError("Chunking strategy must be 'semantic', 'fixed', or 'hierarchical'")
        return v


class LLMSettings(BaseModel):
    """Configuration for a specific LLM instance."""
    model_name: Optional[str] = Field(
        default=None, 
        description="Model name for the LLM"
    )
    provider: str = Field(
        default="groq", 
        description="LLM provider name"
    )
    api_key_env_var: Optional[str] = Field(
        default=None, 
        description="Environment variable name for API key"
    )
    enabled: bool = Field(
        default=True, 
        description="Whether this LLM is enabled"
    )
    temperature: float = Field(
        default=0.1, 
        description="Temperature for LLM sampling",
        ge=0.0,
        le=1.0
    )
    max_tokens: Optional[int] = Field(
        default=None, 
        description="Maximum tokens for LLM response"
    )
    request_timeout: int = Field(
        default=60, 
        description="Timeout for LLM requests in seconds",
        gt=0
    )
    max_retries: int = Field(
        default=3, 
        description="Maximum number of retries for LLM requests",
        ge=0
    )
    api_base: Optional[str] = Field(
        default=None, 
        description="Base URL for the API (e.g., http://localhost:11434 for Ollama)"
    )
    # Additional provider-specific args
    additional_kwargs: Dict[str, Any] = Field(
        default={}, 
        description="Additional provider-specific arguments"
    )

    @property
    def api_key(self) -> Optional[str]:
        """Get API key from environment variable if specified."""
        if self.api_key_env_var:
            return os.getenv(self.api_key_env_var)
        return None


class LLMConfig(BaseModel):
    """Configuration holder for different LLM roles."""
    # Define settings for each role
    metadata_llm: LLMSettings = Field(
        default_factory=lambda: LLMSettings(
            model_name=os.getenv("METADATA_LLM_MODEL", "llama-3.1-8b-instant"),
            provider=os.getenv("METADATA_LLM_PROVIDER", "groq"),
            api_key_env_var=os.getenv("METADATA_LLM_API_KEY_ENV_VAR", "GROQ_API_KEY")
        )
    )
    query_llm: LLMSettings = Field(
        default_factory=lambda: LLMSettings(
            model_name=os.getenv("QUERY_LLM_MODEL", "llama-3.1-70b-versatile"),
            provider=os.getenv("QUERY_LLM_PROVIDER", "groq"),
            api_key_env_var=os.getenv("QUERY_LLM_API_KEY_ENV_VAR", "GROQ_API_KEY")
        )
    )
    coding_llm: LLMSettings = Field(
        default_factory=lambda: LLMSettings(
            model_name=os.getenv("CODING_LLM_MODEL", "llama-3.1-70b-instant"),
            provider=os.getenv("CODING_LLM_PROVIDER", "groq"),
            api_key_env_var=os.getenv("CODING_LLM_API_KEY_ENV_VAR", "GROQ_API_KEY")
        )
    )
    # Metadata enrichment controls by document type
    enrich_documents: bool = Field(
        default=True, 
        description="Controls enrichment for PDF, DOCX, etc. via Docling"
    )
    enrich_code: bool = Field(
        default=True, 
        description="Controls enrichment for code files via CodeSplitter"
    )
    use_cache: bool = Field(
        default=True, 
        description="Whether to cache LLM responses"
    )
    cache_dir: str = Field(
        default="./.cache/llm", 
        description="Directory for LLM cache"
    )


class EmbedderConfig(BaseModel):
    """Configuration for node embedding using LlamaIndex embedding models."""
    # Embedding provider and model
    provider: str = Field(
        default="huggingface", 
        description="Embedding provider (huggingface, openai, cohere, vertex, bedrock, etc.)"
    )
    model_name: str = Field(
        default="BAAI/bge-small-en-v1.5", 
        description="Embedding model name"
    )

    # API configuration for hosted models
    api_key_env_var: Optional[str] = Field(
        default=None, 
        description="Environment variable name for API key"
    )
    api_base: Optional[str] = Field(
        default=None, 
        description="Base URL for API"
    )

    # Performance settings
    embed_batch_size: int = Field(
        default=10, 
        description="Number of texts to embed in a single batch",
        gt=0
    )

    # Caching settings
    use_cache: bool = Field(
        default=True, 
        description="Cache embeddings to avoid recomputing"
    )
    cache_dir: str = Field(
        default="./.cache/embeddings", 
        description="Directory for embedding cache"
    )

    # Advanced settings for specific providers
    additional_kwargs: Dict[str, Any] = Field(
        default={}, 
        description="Additional kwargs for specific providers"
    )

    # Fallback settings
    fallback_provider: Optional[str] = Field(
        default=None, 
        description="Fallback provider if primary fails"
    )
    fallback_model: Optional[str] = Field(
        default=None, 
        description="Fallback model if primary fails"
    )

    @model_validator(mode="after")
    def create_cache_dir(self) -> "EmbedderConfig":
        """Create cache directory if caching is enabled."""
        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        return self

    @property
    def api_key(self) -> Optional[str]:
        """Get API key from environment variable if specified."""
        if self.api_key_env_var:
            return os.getenv(self.api_key_env_var)
        return None


class VectorStoreConfig(BaseModel):
    """Configuration for vector storage."""
    # General settings
    engine: str = Field(
        default="chroma", 
        description="Vector store engine (chroma, qdrant, simple)"
    )
    collection_name: str = Field(
        default="unified_knowledge", 
        description="Vector store collection name"
    )
    distance_metric: str = Field(
        default="cosine", 
        description="Distance metric for vector similarity (cosine, euclid, dot)"
    )

    # ChromaDB specific settings
    vector_db_path: str = Field(
        default="./vector_db", 
        description="Local path for ChromaDB"
    )
    persist_directory: Optional[str] = Field(
        default=None, 
        description="Optional alternative persist directory"
    )

    # Qdrant specific settings
    qdrant_location: str = Field(
        default="local", 
        description="Qdrant location (local or cloud)"
    )
    qdrant_url: Optional[str] = Field(
        default="RAG_VECTORSTORECONFIG_QDRANT_URL", 
        description="URL for cloud Qdrant"
    )
    qdrant_api_key: Optional[str] = Field(
        default=None, 
        description="API key for cloud Qdrant"
    )
    qdrant_api_key_env_var: Optional[str] = Field(
        default="QDRANT_API_KEY",
        description="Environment variable name for Qdrant API key"
    )
    qdrant_local_path: str = Field(
        default="./qdrant_db", 
        description="Path for local Qdrant storage"
    )
    qdrant_grpc_port: int = Field(
        default=6334, 
        description="gRPC port for local Qdrant"
    )
    qdrant_prefer_grpc: bool = Field(
        default=False, 
        description="Whether to prefer gRPC over HTTP"
    )

    # Advanced Qdrant settings
    qdrant_timeout: float = Field(
        default=10.0, 
        description="Timeout for Qdrant operations in seconds"
    )
    qdrant_vector_size: Optional[int] = Field(
        default=None, 
        description="Vector size, determined from first vector if None"
    )
    qdrant_shard_number: Optional[int] = Field(
        default=None, 
        description="Number of shards for Qdrant collection"
    )
    qdrant_replication_factor: Optional[int] = Field(
        default=None, 
        description="Replication factor for Qdrant collection"
    )
    qdrant_write_consistency_factor: Optional[int] = Field(
        default=None, 
        description="Write consistency factor"
    )
    qdrant_on_disk_payload: bool = Field(
        default=True, 
        description="Whether to store payload on disk"
    )

    model_config = ConfigDict(
        validate_default=True
    )

    @model_validator(mode="after")
    def validate_config(self) -> "VectorStoreConfig":
        """Validate configuration after initialization."""
        # Validate engine
        if self.engine not in ["chroma", "qdrant", "simple"]:
            raise ValueError(
                f"Unsupported vector store engine: {self.engine}. "
                "Supported engines: chroma, qdrant, simple"
            )

        # Validate Qdrant configuration if selected
        if self.engine == "qdrant":
            if self.qdrant_location not in ["local", "cloud"]:
                raise ValueError(
                    f"Unsupported Qdrant location: {self.qdrant_location}. "
                    "Supported locations: local, cloud"
                )

            if self.qdrant_location == "cloud":
                # Handle Qdrant URL from environment variables
                if self.qdrant_url and self.qdrant_url.startswith("RAG_"):
                    # The URL is specified as an environment variable name
                    env_url = os.environ.get(self.qdrant_url)
                    if env_url:
                        self.qdrant_url = env_url
                    else:
                        logger.warning(f"Environment variable {self.qdrant_url} not found for Qdrant URL")
                
                # Validate that we have a URL
                if not self.qdrant_url or self.qdrant_url.startswith("RAG_"):
                    raise ValueError("Qdrant cloud location requires qdrant_url to be set or to point to a valid environment variable")
                
                # Try to get API key from environment variable if not directly set
                if not self.qdrant_api_key and self.qdrant_api_key_env_var:
                    env_api_key = os.environ.get(self.qdrant_api_key_env_var)
                    if env_api_key:
                        self.qdrant_api_key = env_api_key
                    
                # Still validate that we have an API key one way or another
                if not self.qdrant_api_key:
                    raise ValueError("Qdrant cloud location requires qdrant_api_key to be set or qdrant_api_key_env_var to point to a valid environment variable")

        # Create directories
        if self.engine == "chroma":
            os.makedirs(self.vector_db_path, exist_ok=True)
        elif self.engine == "qdrant" and self.qdrant_location == "local":
            os.makedirs(self.qdrant_local_path, exist_ok=True)

        return self


class QueryConfig(BaseModel):
    """Configuration for query processing."""
    similarity_top_k: int = Field(
        default=5, 
        description="Number of similar documents to retrieve",
        gt=0
    )
    response_mode: str = Field(
        default="compact", 
        description="Response mode (compact, tree, refine, etc.)"
    )

    # Filtering settings
    default_filter_mode: str = Field(
        default="AND", 
        description="Default filter mode (AND or OR)"
    )

    # Language detection for code queries
    detect_language_in_query: bool = Field(
        default=True, 
        description="Detect programming language in query"
    )

    @field_validator("default_filter_mode")
    @classmethod
    def validate_filter_mode(cls, v: str) -> str:
        """Validate filter mode."""
        if v not in ["AND", "OR"]:
            raise ValueError("Default filter mode must be 'AND' or 'OR'")
        return v


class RegistryConfig(BaseModel):
    """Configuration for document registry."""
    enabled: bool = Field(
        default=True, 
        description="Whether document registry is enabled"
    )
    db_path: Optional[str] = Field(
        default=None, 
        description="Path to registry database (default: output_dir/document_registry.db)"
    )
    reset_stalled_after_seconds: int = Field(
        default=3600, 
        description="Reset stalled docs after this many seconds",
        gt=0
    )


class QueryTransformationConfig(BaseModel):
    """Configuration for query transformation."""
    # Enable/disable features
    enable_query_expansion: bool = Field(
        default=True, 
        description="Enable query expansion"
    )
    enable_query_rewriting: bool = Field(
        default=True, 
        description="Enable query rewriting"
    )

    # HyDE (Hypothetical Document Embeddings) settings
    use_hyde: bool = Field(
        default=True, 
        description="Use Hypothetical Document Embeddings"
    )
    hyde_prompt_template: str = Field(
        default="Please write a passage that answers the question: {query}",
        description="Prompt template for HyDE"
    )

    # Query expansion settings
    expansion_limit: int = Field(
        default=3, 
        description="Maximum number of expansions",
        gt=0
    )
    expansion_technique: str = Field(
        default="llm", 
        description="Query expansion technique (llm, keyword, hybrid)"
    )

    # Query decomposition settings
    enable_decomposition: bool = Field(
        default=False, 
        description="Enable query decomposition"
    )
    decomposition_mode: str = Field(
        default="step", 
        description="Query decomposition mode (step, sub_question)"
    )
    max_sub_questions: int = Field(
        default=5, 
        description="Maximum number of sub-questions",
        gt=0
    )

    # Query rewriting settings
    rewriting_technique: str = Field(
        default="instruct", 
        description="Query rewriting technique (instruct, reflexion, step_back)"
    )
    query_iterations: int = Field(
        default=1, 
        description="Number of query rewrites to attempt",
        ge=1
    )

    @field_validator("expansion_technique")
    @classmethod
    def validate_expansion_technique(cls, v: str) -> str:
        """Validate expansion technique."""
        if v not in ["llm", "keyword", "hybrid"]:
            raise ValueError("Expansion technique must be 'llm', 'keyword', or 'hybrid'")
        return v

    @field_validator("decomposition_mode")
    @classmethod
    def validate_decomposition_mode(cls, v: str) -> str:
        """Validate decomposition mode."""
        if v not in ["step", "sub_question"]:
            raise ValueError("Decomposition mode must be 'step' or 'sub_question'")
        return v

    @field_validator("rewriting_technique")
    @classmethod
    def validate_rewriting_technique(cls, v: str) -> str:
        """Validate rewriting technique."""
        if v not in ["instruct", "reflexion", "step_back"]:
            raise ValueError("Rewriting technique must be 'instruct', 'reflexion', or 'step_back'")
        return v


class RetrievalConfig(BaseModel):
    """Configuration for retrieval components."""
    # General retrieval settings
    retriever_strategy: str = Field(
        default="vector", 
        description="Retriever strategy (vector, keyword, hybrid, ensemble)"
    )
    similarity_top_k: int = Field(
        default=5, 
        description="Number of similar documents to retrieve",
        gt=0
    )
    use_filter_cache: bool = Field(
        default=True, 
        description="Use filter cache"
    )

    # Hybrid search settings
    use_hybrid_search: bool = Field(
        default=False, 
        description="Use hybrid search"
    )
    hybrid_mode: str = Field(
        default="OR", 
        description="Hybrid search mode (AND or OR)"
    )
    hybrid_alpha: float = Field(
        default=0.5, 
        description="Weight between vector (alpha) and keyword (1-alpha) scores",
        ge=0.0,
        le=1.0
    )

    # Keyword search settings
    keyword_top_k: int = Field(
        default=10, 
        description="Number of documents to retrieve with keyword search",
        gt=0
    )
    use_bm25: bool = Field(
        default=True, 
        description="Use BM25 for keyword search"
    )
    use_splade: bool = Field(
        default=False, 
        description="Use SPLADE for keyword search"
    )

    @field_validator("retriever_strategy")
    @classmethod
    def validate_retriever_strategy(cls, v: str) -> str:
        """Validate retriever strategy."""
        if v not in ["vector", "keyword", "hybrid", "ensemble"]:
            raise ValueError("Retriever strategy must be 'vector', 'keyword', 'hybrid', or 'ensemble'")
        return v

    @field_validator("hybrid_mode")
    @classmethod
    def validate_hybrid_mode(cls, v: str) -> str:
        """Validate hybrid mode."""
        if v not in ["AND", "OR"]:
            raise ValueError("Hybrid mode must be 'AND' or 'OR'")
        return v


class RerankerConfig(BaseModel):
    """Configuration for reranking components."""
    enable_reranking: bool = Field(
        default=True, 
        description="Enable reranking"
    )
    reranker_type: str = Field(
        default="semantic", 
        description="Reranker type (semantic, llm, fusion, cohere)"
    )
    rerank_top_n: int = Field(
        default=10, 
        description="Number of results to rerank",
        gt=0
    )
    batch_size: int = Field(
        default=5, 
        description="Batch size for processing nodes",
        gt=0
    )

    # Semantic reranking settings
    semantic_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", 
        description="Semantic reranking model"
    )

    # LLM reranking settings
    rerank_prompt_template: str = Field(
        default="""
        Query: {query}

        Document: {context}

        On a scale of 1 to 10, how relevant is this document to the query?
        Respond with just a number.
        """,
        description="Prompt template for LLM reranking"
    )

    # Cohere reranking settings
    cohere_api_key: str = Field(
        default="",
        description="Cohere API key for reranking"
    )

    model_config = ConfigDict(
        validate_default=True,
        extra="allow"
    )

    @model_validator(mode="after")
    def set_cohere_api_key(self) -> "RerankerConfig":
        """Set Cohere API key from environment variable if not provided."""
        if self.reranker_type == "cohere" and not self.cohere_api_key:
            self.cohere_api_key = os.environ.get("COHERE_API_KEY", "")
        return self

    @field_validator("reranker_type")
    @classmethod
    def validate_reranker_type(cls, v: str) -> str:
        """Validate reranker type."""
        if v not in ["semantic", "llm", "fusion", "cohere"]:
            raise ValueError("Reranker type must be 'semantic', 'llm', 'fusion', or 'cohere'")
        return v


class SynthesisConfig(BaseModel):
    """Configuration for answer synthesis."""
    synthesis_strategy: str = Field(
        default="refine", 
        description="Synthesis strategy (refine, tree, compact, simple)"
    )
    streaming: bool = Field(
        default=False, 
        description="Enable streaming responses"
    )
    use_async: bool = Field(
        default=False, 
        description="Use async mode"
    )

    # Refine strategy settings
    refine_prompt_template: str = Field(
        default="""
        The original query is: {query}

        We have provided an existing answer: {existing_answer}

        We have the opportunity to refine the existing answer with new context: {context}

        Given the new context, refine the original answer. If the context isn't useful, return the original answer.
        """,
        description="Prompt template for refine strategy"
    )

    # Tree strategy settings
    tree_width: int = Field(
        default=3, 
        description="Number of chunks to process at once in the tree",
        gt=0
    )

    # Response formatting
    include_citations: bool = Field(
        default=True, 
        description="Include citations in response"
    )
    include_metadata: bool = Field(
        default=True, 
        description="Include metadata in response"
    )
    structured_answer_filtering: bool = Field(
        default=False, 
        description="Enable structured answer filtering"
    )

    @field_validator("synthesis_strategy")
    @classmethod
    def validate_synthesis_strategy(cls, v: str) -> str:
        """Validate synthesis strategy."""
        if v not in ["refine", "tree", "compact", "simple"]:
            raise ValueError("Synthesis strategy must be 'refine', 'tree', 'compact', or 'simple'")
        return v


class QueryPipelineConfig(BaseModel):
    """Configuration for the query pipeline."""
    # Component configurations
    transformation: QueryTransformationConfig = Field(
        default_factory=QueryTransformationConfig,
        description="Query transformation configuration"
    )
    retrieval: RetrievalConfig = Field(
        default_factory=RetrievalConfig,
        description="Retrieval configuration"
    )
    reranker: RerankerConfig = Field(
        default_factory=RerankerConfig,
        description="Reranker configuration"
    )
    synthesis: SynthesisConfig = Field(
        default_factory=SynthesisConfig,
        description="Synthesis configuration"
    )

    # Pipeline settings
    timeout_seconds: int = Field(
        default=60, 
        description="Timeout for pipeline execution in seconds",
        gt=0
    )
    async_mode: bool = Field(
        default=False, 
        description="Use async mode"
    )
    streaming_response: bool = Field(
        default=False, 
        description="Enable streaming responses"
    )
    cache_results: bool = Field(
        default=True, 
        description="Cache query results"
    )
    cache_dir: str = Field(
        default="./.cache/query_results", 
        description="Directory for query cache"
    )
    verbose: bool = Field(
        default=False, 
        description="Enable verbose logging"
    )

    model_config = ConfigDict(
        validate_default=True
    )

    @model_validator(mode="after")
    def create_cache_dir(self) -> "QueryPipelineConfig":
        """Create cache directory if caching is enabled."""
        if self.cache_results and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        return self


class ApplicationEnvironment(str, Enum):
    """Application environment."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class UnifiedConfig(BaseModel):
    """Main configuration for the unified document parsing system."""
    # --- Global settings ---
    input_directory: str = Field(
        ...,  # Required parameter
        description="Input directory path"
    )
    project_name: str = Field(
        default="unified-parser", 
        description="Project name"
    )
    output_dir: str = Field(
        default="./output", 
        description="Output directory path"
    )
    environment: ApplicationEnvironment = Field(
        default=ApplicationEnvironment.DEVELOPMENT,
        description="Application environment"
    )

    # --- Component configs ---
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )
    parallel: ParallelConfig = Field(
        default_factory=ParallelConfig,
        description="Parallel processing configuration"
    )
    detector: DetectorConfig = Field(
        default_factory=DetectorConfig,
        description="Document detection configuration"
    )
    loader: LoaderConfig = Field(
        default_factory=LoaderConfig,
        description="Document loader configuration"
    )
    code_processor: CodeProcessorConfig = Field(
        default_factory=CodeProcessorConfig,
        description="Code processor configuration"
    )
    docling: DoclingConfig = Field(
        default_factory=DoclingConfig,
        description="Docling configuration"
    )
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration"
    )
    embedder: EmbedderConfig = Field(
        default_factory=EmbedderConfig,
        description="Embedder configuration"
    )
    vector_store: VectorStoreConfig = Field(
        default_factory=VectorStoreConfig,
        description="Vector store configuration"
    )
    query: QueryConfig = Field(
        default_factory=QueryConfig,
        description="Query configuration"
    )
    registry: RegistryConfig = Field(
        default_factory=RegistryConfig,
        description="Document registry configuration"
    )
    query_pipeline: QueryPipelineConfig = Field(
        default_factory=QueryPipelineConfig,
        description="Query pipeline configuration"
    )

    model_config = ConfigDict(
        validate_default=True
    )

    @model_validator(mode="after")
    def validate_and_setup(self) -> "UnifiedConfig":
        """Validate configuration and set up directories after initialization."""
        # Input directory validation
        input_path = Path(self.input_directory)
        if not input_path.exists() or not input_path.is_dir():
            raise ValueError(f"Input directory does not exist: {self.input_directory}")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create registry directory if needed
        if self.registry.enabled and not self.registry.db_path:
            registry_path = os.path.join(self.output_dir, "document_registry.db")
            self.registry.db_path = registry_path
            os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        
        # Create cache directories
        if self.llm.use_cache and self.llm.cache_dir:
            os.makedirs(self.llm.cache_dir, exist_ok=True)
        
        # Ensure embedding consistency
        if self.embedder.provider != "huggingface" and self.embedder.provider == self.query_pipeline.retrieval.retriever_strategy:
            # If using the same provider for embedding and retrieval, ensure model consistency
            if hasattr(self.query_pipeline.retrieval, "model_name") and self.query_pipeline.retrieval.model_name != self.embedder.model_name:
                # Log warning about potential embedding space mismatch
                print(f"WARNING: Using different embedding models for indexing ({self.embedder.model_name}) and retrieval ({self.query_pipeline.retrieval.model_name})")
        
        return self